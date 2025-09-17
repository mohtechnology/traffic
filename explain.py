import cv2   # OpenCV - वीडियो/इमेज प्रोसेसिंग के लिए
import time  # टाइमिंग और डिले लगाने के लिए
import threading  # मल्टीथ्रेडिंग (background में processes चलाने के लिए)
import numpy as np  # NumPy - arrays और matrix operations के लिए
from datetime import datetime, timedelta  # डेट और टाइम calculations के लिए
from flask import Flask, render_template, Response, jsonify  # Flask web server और API के लिए
from flask_sqlalchemy import SQLAlchemy  # Flask के साथ SQLAlchemy ORM
from sqlalchemy import UniqueConstraint  # DB constraints लगाने के लिए

# -------------------
# Flask + DB Setup
# -------------------
app = Flask(__name__)  # Flask app बनाना
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///traffic.db'  # SQLite DB file का path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Extra overhead बंद करना
db = SQLAlchemy(app)  # DB को app से connect करना

# -------------------
# DB Model
# -------------------
class VehicleLog(db.Model):  # एक database टेबल define कर रहे हैं
    id = db.Column(db.Integer, primary_key=True)  # Primary key (auto increment ID)
    lane = db.Column(db.Integer, nullable=False)  # कौनसी लेन (1,2,3...)
    vehicles = db.Column(db.Integer, nullable=False)  # उस समय vehicles की count
    timestamp = db.Column(db.DateTime, nullable=False)  # कब record हुआ

    # एक ही लेन और एक ही टाइम पर duplicate entry ना बने
    __table_args__ = (UniqueConstraint("lane", "timestamp", name="unique_lane_timestamp"),)

# DB को initialize करना (अगर tables exist नहीं करते तो बना देगा)
with app.app_context():
    db.create_all()

# -------------------
# Camera Config
# -------------------
camera_sources = {
    0: 'video.mp4',
    1: 'video.mp4',
    2: 'video.mp4',
    3: 'video.mp4'
}
# हर source के लिए cv2.VideoCapture object बनाना
caps = {k: cv2.VideoCapture(v) for k, v in camera_sources.items()}

# -------------------
# Shared State
# -------------------
frames = {}  # हर कैमरे का current frame store होगा
for i, cap in caps.items():
    ret, frame = cap.read()  # पहला frame read करना
    if not ret:  # अगर frame नहीं मिला
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # blank black frame
    else:
        frame = cv2.resize(frame, (640, 480))  # frame resize
    frames[i] = frame

# हर कैमरे की initial vehicle count और signal state
vehicle_counts = {i: 0 for i in camera_sources.keys()}  # शुरू में 0 count
states = {i: "RED" for i in camera_sources.keys()}  # शुरू में सभी signals RED

# Signal timing setup
base_time = 30       # Base time per signal (30 sec)
signal_index = 0     # कौनसी लेन group green है (0 या 1)
signal_timer = base_time  # current timer
last_update = time.time()  # आखरी बार कब switch हुआ

# Thread safety के लिए lock
lock = threading.Lock()

# -------------------
# Helper Functions
# -------------------
def process_frame_simple(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # frame को grayscale में बदलना
    blur = cv2.GaussianBlur(gray, (3, 3), 0)  # Blur लगाना noise कम करने के लिए
    _, th1 = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)  # Threshold1
    _, th2 = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)  # Threshold2
    thresh = cv2.bitwise_or(th1, th2)  # दोनों thresholds को combine करना

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Morphology kernel
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Close gaps
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)   # Remove noise

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Contours निकालना
    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # Bounding box
        area = w * h  # Area निकालना
        ar = w / h if h > 0 else 0  # Aspect ratio
        # अगर object का area और ratio valid है तो उसे vehicle मान लेना
        if 200 < area < 80000 and 0.3 < ar < 8.0:
            count += 1
    return count  # Vehicle count return करना

# Frame पर traffic light draw करना
def draw_traffic_light(frame, state):
    colors = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0)}
    x, y, radius = 50, 50, 15
    # Red light
    cv2.circle(frame, (x, y), radius, colors["RED"] if state == "RED" else (50, 50, 50), -1)
    # Yellow light
    cv2.circle(frame, (x, y + 40), radius, colors["YELLOW"] if state == "YELLOW" else (50, 50, 50), -1)
    # Green light
    cv2.circle(frame, (x, y + 80), radius, colors["GREEN"] if state == "GREEN" else (50, 50, 50), -1)
    # Outer circles (white border)
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

        # Lane groups: signal_index=0 -> lanes 0,1 green; lanes 2,3 red
        if signal_index == 0:
            green_cams, red_cams = [0, 1], [2, 3]
        else:
            green_cams, red_cams = [2, 3], [0, 1]

        # Process each camera
        for i, cap in caps.items():
            if i in red_cams:  # अगर red signal है तो पुराना frame दिखाना
                with lock:
                    new_frames[i] = frames[i].copy()
                counts[i] = vehicle_counts[i]
                continue

            ret, frame = cap.read()
            if not ret:  # अगर वीडियो खत्म हो गया
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # फिर से शुरू करो
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

            frame = cv2.resize(frame, (640, 480))  # Frame resize
            count = process_frame_simple(frame.copy())  # Vehicles count करना

            counts[i] = count
            new_frames[i] = frame

        elapsed = int(time.time() - last_update)  # Last update से कितना time हुआ
        remaining = max(0, signal_timer - elapsed)  # कितना time बचा है

        # हर lane को state देना
        for i in list(frames.keys()):
            if i in green_cams:
                states[i] = "YELLOW" if remaining <= 5 else "GREEN"  # आखरी 5 sec में yellow
            else:
                states[i] = "RED"
            target_frame = new_frames.get(i, frames[i]).copy()
            target_frame = draw_traffic_light(target_frame, states[i])
            new_frames[i] = target_frame

        # Adaptive signal timing (traffic load के हिसाब से)
        if elapsed >= signal_timer:
            with lock:
                green_total = sum(vehicle_counts[i] for i in green_cams)
                red_total = sum(vehicle_counts[i] for i in red_cams)

            if green_total > 0 and red_total > 0:
                diff = abs(green_total - red_total) / min(green_total, red_total)
                if diff >= 0.0125:  # अगर 1.25% ज्यादा traffic है
                    signal_timer = base_time + 15
                else:
                    signal_timer = base_time
            else:
                signal_timer = base_time

            signal_index = 1 - signal_index  # Lane group switch करना
            last_update = time.time()

        with lock:
            frames = new_frames
            vehicle_counts.update(counts)

        time.sleep(0.05)  # हर 50ms बाद update

# -------------------
# Logging thread
# -------------------
def log_vehicle_counts():
    while True:
        now = datetime.now()
        next_minute = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
        wait_seconds = (next_minute - now).total_seconds()
        time.sleep(wait_seconds)  # अगले minute तक wait करो

        ts = datetime.now().replace(second=0, microsecond=0)  # Timestamp set करो

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
                        db.session.rollback()  # Error आए तो rollback

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
        total_intersections = len(caps)  # कितने cameras/intersections हैं
        active_systems = sum(1 for _, cap in caps.items() if cap.isOpened())  # कितने चल रहे हैं
        avg_wait_time = signal_timer  # average wait time
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
    return render_template("index.html")  # homepage
d
@app.route("/dashboard")
def dashboard():
    logs = VehicleLog.query.order_by(VehicleLog.timestamp.desc()).limit(50).all()
    return render_template("dashboard.html", logs=logs)  # dashboard page

# -------------------
# Start
# -------------------
if __name__ == "__main__":
    t_frames = threading.Thread(target=update_frames, daemon=True)  # Frame update thread
    t_frames.start()

    t_log = threading.Thread(target=log_vehicle_counts, daemon=True)  # Logging thread
    t_log.start()

    app.run(host="0.0.0.0", port=8000, debug=True)  # Flask server start