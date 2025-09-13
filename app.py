import cv2, time, threading
import numpy as np
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# -------------------
# Camera Config
# -------------------
camera_sources = {
    0: 'video1.mp4',
    1: 'video2.mp4',
    2: 'video3.mp4',
    3: 'video4.mp4'
}
caps = {k: cv2.VideoCapture(v) for k, v in camera_sources.items()}

# Initialize frames with first frame (thumbnail)
frames = {}
for i, cap in caps.items():
    ret, frame = cap.read()
    if not ret:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        frame = cv2.resize(frame, (640, 480))
    frames[i] = frame

# Traffic signal config
base_time, extra_time = 30, 15
signal_index, signal_timer, last_update = 0, base_time, time.time()
vehicle_counts = {i: 0 for i in camera_sources.keys()}
states = {i: "RED" for i in camera_sources.keys()}

# -------------------
# Helper Functions
# -------------------
def process_frame_simple(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th1 = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)
    _, th2 = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.bitwise_or(th1, th2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area, ar = w * h, w / h if h > 0 else 0
        if 200 < area < 80000 and 0.3 < ar < 8.0:
            count += 1
    return count

def draw_traffic_light(frame, state):
    colors = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0)}
    x, y, radius = 50, 50, 15
    cv2.circle(frame, (x, y), radius, colors["RED"] if state == "RED" else (50, 50, 50), -1)
    cv2.circle(frame, (x, y + 40), radius, colors["YELLOW"] if state == "YELLOW" else (50, 50, 50), -1)
    cv2.circle(frame, (x, y + 80), radius, colors["GREEN"] if state == "GREEN" else (50, 50, 50), -1)
    for shift in [0, 40, 80]:
        cv2.circle(frame, (x, y + shift), radius, (255, 255, 255), 2)
    return frame

# -------------------
# Thread to update frames
# -------------------
def update_frames():
    global frames, vehicle_counts, states, signal_index, signal_timer, last_update
    while True:
        new_frames, counts = {}, {}

        # Green/Red rotation (2 green, 2 red)
        if signal_index == 0:
            green_cams, red_cams = [0, 1], [2, 3]
        else:
            green_cams, red_cams = [2, 3], [0, 1]

        # Process frames
        for i, cap in caps.items():
            if i in red_cams:
                new_frames[i] = frames[i]   # freeze (pause)
                counts[i] = vehicle_counts[i]
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480))
            count = process_frame_simple(frame.copy())
            counts[i] = count
            new_frames[i] = frame

        elapsed = int(time.time() - last_update)
        remaining = max(0, signal_timer - elapsed)

        for i in frames.keys():
            if i in green_cams:
                states[i] = "YELLOW" if remaining <= 5 else "GREEN"
            else:
                states[i] = "RED"
            new_frames[i] = draw_traffic_light(new_frames[i], states[i])

        # Change cycle
        if elapsed >= signal_timer:
            signal_index = 1 - signal_index
            signal_timer = base_time
            last_update = time.time()

        frames = new_frames
        vehicle_counts.update(counts)
        time.sleep(0.05)

# -------------------
# Video Feeds
# -------------------
def gen_video(cam_id):
    global frames
    while True:
        frame = frames[cam_id]
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    if cam_id not in caps:
        return "Camera not found", 404
    return Response(gen_video(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------
# Signals API for frontend
# -------------------
@app.route("/api/signals")
def api_signals():
    elapsed = int(time.time() - last_update)
    remaining = max(0, signal_timer - elapsed)

    signals = []
    if signal_index == 0:
        green_cams, red_cams = [0, 1], [2, 3]
    else:
        green_cams, red_cams = [2, 3], [0, 1]

    for i in caps.keys():
        if i in green_cams:
            signals.append({
                "lane": f"Lane {i+1}",
                "vehicles": vehicle_counts[i],
                "remaining": remaining,
                "status": states[i]
            })
        else:
            # RED lanes → same countdown as GREEN (no +signal_timer)
            signals.append({
                "lane": f"Lane {i+1}",
                "vehicles": vehicle_counts[i],
                "remaining": remaining,
                "status": states[i]
            })
    return jsonify(signals)

# -------------------
# Home route
# -------------------
@app.route("/")
def index():
    return render_template("index.html")

# -------------------
# Start
# -------------------
if __name__ == "__main__":
    t = threading.Thread(target=update_frames, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=8000, debug=True)