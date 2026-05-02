import cv2
import time
import psutil
import subprocess
import threading
import numpy as np
from ultralytics import YOLO

global samples, sampling
VIDEO_PATH = "video.mp4"
MODEL_PATH = "yolo26x-trained_openvino_model"
OUTPUT_FILE = "metrics.txt"

# ---------- utils ----------

def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True).decode().strip()
    except:
        return ""

def get_cpu_temp():
    out = run_cmd("vcgencmd measure_temp")
    try:
        return float(out.replace("temp=", "").replace("'C", ""))
    except:
        return -1

def get_cpu_freq():
    f = psutil.cpu_freq()
    return f.current if f else 0

def is_throttled():
    out = run_cmd("vcgencmd get_throttled")
    try:
        val = int(out.split("=")[1], 16)
        return (val & 0x4) != 0
    except:
        return False

# ---------- sampling ----------

samples = []
sampling = False

def get_ram_gb():
    mem = psutil.virtual_memory()
    return round(mem.used / (1024**3), 4)

def sampler(interval):
    while sampling:
        samples.append({
            "cpu": psutil.cpu_percent(interval=None),
            "freq": get_cpu_freq(),
            "ram": get_ram_gb(),
            "temp": get_cpu_temp(),
            "throttle": is_throttled()
        })
        time.sleep(interval)

# ---------- aggregation ----------

def aggregate(samples):
    if len(samples) == 0:
        return {}

    return {
        "cpu_avg": np.mean([s["cpu"] for s in samples]),
        "cpu_max": np.max([s["cpu"] for s in samples]),
        "freq_avg": np.mean([s["freq"] for s in samples]),
        "ram_used": samples[-1]["ram"],
        "temp_max": np.max([s["temp"] for s in samples]),
        "throttle_count": sum(1 for s in samples if s["throttle"])
    }

# ---------- main ----------
WARMUP_FRAMES = 10
latencies = []
	
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

for i in range(WARMUP_FRAMES):
	ret, frame = cap.read()
	if not ret:
		break

	start = time.time()
	_ = model(frame)
	lat = (time.time() - start) * 1000
	latencies.append(lat)

avg_latency = np.mean(latencies)

TARGET_SAMPLES = 20
sampling_interval = (avg_latency / TARGET_SAMPLES) / 1000  # to seconds

	# clamp to safe bounds
sampling_interval = max(0.001, min(sampling_interval, 0.02))
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')

with open(OUTPUT_FILE, "w") as f:
    f.write("frame,latency_ms,cpu_avg,cpu_max,freq_avg,ram_used,temp_max,throttle_count\n")

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        samples = []
        sampling = True

        t = threading.Thread(target=sampler, args=(sampling_interval,))
        t.start()

        start = time.time()
        _ = model(frame)
        latency = (time.time() - start) * 1000

        sampling = False
        t.join()

        stats = aggregate(samples)

        line = f"{frame_id},{latency:.2f}," \
               f"{stats['cpu_avg']:.2f},{stats['cpu_max']:.2f}," \
               f"{stats['freq_avg']:.2f},{stats['ram_used']}," \
               f"{stats['temp_max']:.2f}," \
               f"{stats['throttle_count']}\n"

        print(line.strip())
        f.write(line)

        frame_id += 1

cap.release()

