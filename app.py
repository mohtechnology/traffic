import cv2
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import UniqueConstraint

# -------------------
# Flask + DB Setup
# -------------------
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///traffic.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# -------------------
# DB Model
# -------------------
class VehicleLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    lane = db.Column(db.Integer, nullable=False)       # stored as 1..N
    vehicles = db.Column(db.Integer, nullable=False)   # snapshot vehicle count
    timestamp = db.Column(db.DateTime, nullable=False)

    __table_args__ = (UniqueConstraint("lane", "timestamp", name="unique_lane_timestamp"),)

with app.app_context():
    db.create_all()

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

# -------------------
# Shared State
# -------------------
frames = {}
for i, cap in caps.items():
    ret, frame = cap.read()
    if not ret:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        frame = cv2.resize(frame, (640, 480))
    frames[i] = frame

vehicle_counts = {i: 0 for i in camera_sources.keys()}       # snapshot count
states = {i: "RED" for i in camera_sources.keys()}

# Signal timing
base_time = 30
signal_index = 0
signal_timer = base_time
last_update = time.time()

# Thread safety
lock = threading.Lock()

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
        area = w * h
        ar = w / h if h > 0 else 0
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
# Frame update thread
# -------------------
def update_frames():
    global frames, vehicle_counts, states, signal_index, signal_timer, last_update
    while True:
        new_frames = {}
        counts = {}

        # lane groups
        if signal_index == 0:
            green_cams, red_cams = [0, 1], [2, 3]
        else:
            green_cams, red_cams = [2, 3], [0, 1]

        # process frames
        for i, cap in caps.items():
            if i in red_cams:
                with lock:
                    new_frames[i] = frames[i].copy()
                counts[i] = vehicle_counts[i]
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

            frame = cv2.resize(frame, (640, 480))
            count = process_frame_simple(frame.copy())

            counts[i] = count
            new_frames[i] = frame

        elapsed = int(time.time() - last_update)
        remaining = max(0, signal_timer - elapsed)

        # assign states
        for i in list(frames.keys()):
            if i in green_cams:
                states[i] = "YELLOW" if remaining <= 5 else "GREEN"
            else:
                states[i] = "RED"
            target_frame = new_frames.get(i, frames[i]).copy()
            target_frame = draw_traffic_light(target_frame, states[i])
            new_frames[i] = target_frame

        # ---- Adaptive Signal Timing ----
        if elapsed >= signal_timer:
            with lock:
                green_total = sum(vehicle_counts[i] for i in green_cams)
                red_total = sum(vehicle_counts[i] for i in red_cams)

            if green_total > 0 and red_total > 0:
                diff = abs(green_total - red_total) / min(green_total, red_total)
                if diff >= 0.0125:  # 1.25% more traffic
                    signal_timer = base_time + 15
                else:
                    signal_timer = base_time
            else:
                signal_timer = base_time

            signal_index = 1 - signal_index
            last_update = time.time()

        with lock:
            frames = new_frames
            vehicle_counts.update(counts)

        time.sleep(0.05)

# -------------------
# Logging thread
# -------------------
def log_vehicle_counts():
    while True:
        now = datetime.now()
        next_minute = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
        wait_seconds = (next_minute - now).total_seconds()
        time.sleep(wait_seconds)

        ts = datetime.now().replace(second=0, microsecond=0)

        with app.app_context():
            with lock:
                for lane_zero_index, count in vehicle_counts.items():
                    log = VehicleLog(
                        lane=lane_zero_index + 1,
                        vehicles=int(count),
                        timestamp=ts
                    )
                    try:
                        db.session.add(log)
                        db.session.commit()
                    except Exception:
                        db.session.rollback()

# -------------------
# Video streaming
# -------------------
def gen_video(cam_id):
    global frames
    while True:
        with lock:
            frame = frames.get(cam_id)
            if frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                ret, buffer = cv2.imencode('.jpg', blank)
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    if cam_id not in caps:
        return "Camera not found", 404
    return Response(gen_video(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------
# API: signals
# -------------------
@app.route("/api/signals")
def api_signals():
    elapsed = int(time.time() - last_update)
    remaining = max(0, signal_timer - elapsed)

    signals = []
    with lock:
        for i in sorted(caps.keys()):
            signals.append({
                "lane": f"Lane {i + 1}",
                "vehicles": int(vehicle_counts.get(i, 0)),
                "remaining": remaining,
                "status": states.get(i, "RED")
            })
    return jsonify(signals)

# -------------------
# API: logs
# -------------------
@app.route("/api/logs")
def api_logs():
    logs = VehicleLog.query.order_by(VehicleLog.timestamp.desc()).limit(100).all()
    return jsonify([
        {
            "lane": log.lane,
            "vehicles": log.vehicles,
            "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
        for log in logs
    ])

# -------------------
# API: stats (for cards)
# -------------------
@app.route("/api/stats")
def api_stats():
    with lock:
        total_intersections = len(caps)
        active_systems = sum(1 for _, cap in caps.items() if cap.isOpened())
        avg_wait_time = signal_timer
        efficiency = max(50, 100 - int(np.mean(list(vehicle_counts.values())))) if vehicle_counts else 100

    return jsonify({
        "total_intersections": total_intersections,
        "active_systems": active_systems,
        "avg_wait_time": avg_wait_time,
        "efficiency": efficiency
    })

# -------------------
# Routes
# -------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    logs = VehicleLog.query.order_by(VehicleLog.timestamp.desc()).limit(50).all()
    return render_template("dashboard.html", logs=logs)

# -------------------
# Start
# -------------------
if __name__ == "__main__":
    t_frames = threading.Thread(target=update_frames, daemon=True)
    t_frames.start()

    t_log = threading.Thread(target=log_vehicle_counts, daemon=True)
    t_log.start()

    app.run(host="0.0.0.0", port=8000, debug=True)
