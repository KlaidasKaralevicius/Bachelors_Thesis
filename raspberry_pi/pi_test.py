from picamera2 import Picamera2
import cv2
import time
import psutil
import subprocess
import threading
import numpy as np
from ultralytics import YOLO

video_path = "video.mp4"
model_path = "yolo26x-trained_openvino_model"
video_metrics = "video_metrics.txt"
live_metrics = "live_metrics.txt"
warmup_frames = 10
target_samples = 20

# ===== HELPERS ======
# execute commands
def run_cmd(cmd):
    try:
        return subprocess.check_output(
            cmd,
            shell=True
        ).decode().strip()
    except:
        return ""
    
# CPU temp
def get_cpu_temp():
    out = run_cmd("vcgencmd measure_temp")
    try:
        return float(out.replace("temp=", "").replace("'C", ""))
    except:
        return -1

# CPU frequency
def get_cpu_freq():
    f = psutil.cpu_freq()
    return f.current if f else 0

# check if CPU throttled
def is_throttled():
    out = run_cmd("vcgencmd get_throttled")
    try:
        val = int(out.split("=")[1], 16)
        return (val & 0x4) != 0
    except:
        return False

# RAM used
def get_ram_gb():
    mem = psutil.virtual_memory()
    return round(mem.used / (1024 ** 3), 4)

# ===== SAMPLING ======
samples = []
sampling = False

# collect multiple values for each inference
def sampler(interval):
    global samples
    global sampling

    while sampling:
        samples.append({
            "cpu": psutil.cpu_percent(interval=None),
            "freq": get_cpu_freq(),
            "ram": get_ram_gb(),
            "temp": get_cpu_temp(),
            "throttle": is_throttled()
        })
        time.sleep(interval)

# ===== AGGREGATE ======
# average metrics for each inference step
def aggregate(samples):
    if len(samples) == 0:
        return {}
    return {
        "cpu_avg":
            np.mean([s["cpu"] for s in samples]),
        "cpu_max":
            np.max([s["cpu"] for s in samples]),
        "freq_avg":
            np.mean([s["freq"] for s in samples]),
        "ram_used":
            samples[-1]["ram"],
        "temp_max":
			np.max([s["temp"] for s in samples]),
        "throttle_count":
            sum(1 for s in samples if s["throttle"])
    }

# ===== VIDEO INFERENCE ======
def run_video_inference():
    global samples
    global sampling
    
    model = YOLO(model_path, task='detect')
    cap = cv2.VideoCapture(video_path)
    latencies = []

    for i in range(warmup_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # use starting frames to calculate average inference
        start = time.time()
        _ = model(frame, task='detection')
        latency = (time.time() - start) * 1000
        latencies.append(latency)

    # calculate waiting time before each sample
    avg_latency = np.mean(latencies)
    sampling_interval = (avg_latency / target_samples) / 1000
    sampling_interval = max(0.001,min(sampling_interval, 0.02))
    
    print('\n\n\n======= START INFERENCE =======\n\n\n')
    
    # reset video
    cap.release()
    cap = cv2.VideoCapture(video_path)
    total_start = time.time()
    with open(video_metrics, "w") as f:
        f.write(
            "frame,"
            "latency_ms,"
            "cpu_avg,"
            "cpu_max,"
            "freq_avg,"
            "ram_used,"
            "temp_max,"
            "throttle_count\n"
        )

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # sample values for each frame
            samples = []
            sampling = True

            t = threading.Thread(
                target=sampler,
                args=(sampling_interval,)
            )
            t.start()
            start = time.time()
            _ = model(frame)
            latency = (time.time() - start) * 1000

            # write averaged values for frame
            sampling = False
            t.join()
            stats = aggregate(samples)
            line = (
                f"{frame_id},"
                f"{latency:.2f},"
                f"{stats['cpu_avg']:.2f},"
                f"{stats['cpu_max']:.2f},"
                f"{stats['freq_avg']:.2f},"
                f"{stats['ram_used']},"
                f"{stats['temp_max']:.2f},"
                f"{stats['throttle_count']}\n"
            )
            f.write(line)
            frame_id += 1

    # finish whole video inference
    total_time = time.time() - total_start
    print("\n========================")
    print(f"Total runtime: {total_time:.2f} sec")
    print("========================\n")
    cap.release()
    cv2.destroyAllWindows()

# ===== LIVE INFERENCE ======
def run_live_inference():
    global samples
    global sampling

    model = YOLO(model_path, task='detect')
    # initialized Rpi camera
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (640, 640), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    # calculate average latency
    latency_samples = []
    while len(latency_samples) < 2:
        frame = picam2.capture_array()
        start = time.time()
        _ = model(frame)
        latency = (time.time() - start) * 1000
        latency_samples.append(latency)
        
    # find sampling interval
    avg_latency = np.mean(latency_samples)
    sampling_interval = (avg_latency / target_samples) / 1000
    sampling_interval = max(0.001, min(sampling_interval, 0.02))

    print('\n\n\n======= START INFERENCE =======\n\n\n')
    
    benchmark_start = time.time()
    processed_frames = 0
    inference_latencies = []

    with open(live_metrics, "w") as f:
        f.write("frame,latency_ms,cpu_avg,cpu_max,freq_avg,ram_used,throttle_count\n")
        # inference current frame, only run for 30 seconds (but finish last inference)
        while (time.time() - benchmark_start) < 30:
            frame = picam2.capture_array()
            # start sampling
            samples = []
            sampling = True
            t = threading.Thread(
                target=sampler,
                args=(sampling_interval,)
            )
            t.start()
            start = time.time()
            _ = model(frame)
            inference_time = time.time() - start

            # average values per frame
            sampling = False
            t.join()
            latency_ms = inference_time * 1000
            inference_latencies.append(latency_ms)
            stats = aggregate(samples)
            line = (
                f"{processed_frames},"
                f"{latency_ms:.2f},"
                f"{stats['cpu_avg']:.2f},"
                f"{stats['cpu_max']:.2f},"
                f"{stats['freq_avg']:.2f},"
                f"{stats['ram_used']},"
                f"{stats['throttle_count']}\n"
            )
            f.write(line)
            processed_frames += 1

    # finish inference
    total_runtime = time.time() - benchmark_start
    print("\n========================")
    print(f"Total runtime: {total_runtime:.2f} sec")
    print("========================\n")
    picam2.stop()
    cv2.destroyAllWindows()

# ===== MAIN ======

# video benchmark
#run_video_inference()

# live benchmark
run_live_inference()