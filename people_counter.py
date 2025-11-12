# app.py
# Usage:
#   1) Activate venv
#   2) python app.py
#   3) POST a video file to http://localhost:5000/upload (form field 'file')
#
# Returns JSON with counts and a URL to download the annotated video.



# app.py
# Usage:
#   1) Activate venv
#   2) python app.py
#   3) POST a video file to http://localhost:5000/upload (form field 'file')
#
# Returns JSON with counts and a URL to download the annotated video.

import os
import time
import uuid
from flask import Flask, request, jsonify, send_from_directory, abort
from flask import render_template_string

from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort   # pip install sort-tracker

# -------- CONFIG ----------
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
ALLOWED_EXT = {"mp4", "avi", "mov", "mkv"}
MODEL_WEIGHTS = "yolov8n.pt"
CONF_THRESH = 0.45
IOU_NMS = 0.45
LINE_POSITION = 0.5  # fraction of frame height
MIN_BOX_AREA = 500
MAX_PROCESS_SECONDS = 300  # safety timeout per request (optional)
# --------------------------

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
app = Flask(__name__)

# Load model once (global) to speed requests
model = YOLO(MODEL_WEIGHTS)
model.conf = CONF_THRESH
model.iou = IOU_NMS

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def centroid_from_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def process_video(input_path, output_path, show_window=False):
    """
    Process the input video, write annotated output, and return a report dict.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    line_y = int(height * LINE_POSITION)
    counts = {"in": 0, "out": 0}
    track_last_centroid_y = {}
    track_events = {}  # track_id -> list of events {"frame":..,"time":..,"event":"enter"/"exit"}

    frame_idx = 0
    start_time = time.time()
    last_frame_time = start_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # run detector
        detections = []
        results = model(frame)  # single call returns batch of 1
        for box in results[0].boxes:
            cls = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            if cls != 0:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_BOX_AREA:
                continue
            detections.append([x1, y1, x2, y2, conf])

        dets_np = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

        # tracking
        tracked = tracker.update(dets_np)

        for t in tracked:
            x1, y1, x2, y2, track_id = t
            x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
            cx, cy = centroid_from_box((x1, y1, x2, y2))
            prev_y = track_last_centroid_y.get(track_id, None)

            # check crossing
            if prev_y is not None:
                if prev_y < line_y and cy >= line_y:
                    # entering direction (downwards)
                    counts["in"] += 1
                    track_events.setdefault(track_id, []).append({
                        "frame": frame_idx,
                        "time": time.time() - start_time,
                        "event": "enter"
                    })
                elif prev_y > line_y and cy <= line_y:
                    counts["out"] += 1
                    track_events.setdefault(track_id, []).append({
                        "frame": frame_idx,
                        "time": time.time() - start_time,
                        "event": "exit"
                    })

            track_last_centroid_y[track_id] = cy

            # draw annotations
            cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 200, 64), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        # draw UI
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
        cv2.putText(frame, f"In: {counts['in']}  Out: {counts['out']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out.write(frame)

        if show_window:
            cv2.imshow("annot", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # optional safety timeout
        if MAX_PROCESS_SECONDS and (time.time() - start_time) > MAX_PROCESS_SECONDS:
            break

    cap.release()
    out.release()
    if show_window:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    fps_processed = frame_idx / elapsed if elapsed > 0 else 0.0

    # Build report
    report = {
        "source": os.path.basename(input_path),
        "output_video": os.path.basename(output_path),
        "frames_processed": frame_idx,
        "elapsed_seconds": round(elapsed, 2),
        "avg_fps": round(fps_processed, 2),
        "counts": counts,
        "track_events": track_events
    }
    return report

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file provided (form field 'file')"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"file type not allowed, allowed: {ALLOWED_EXT}"}), 400

    filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex[:8]
    saved_name = f"{unique_id}_{filename}"
    input_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(input_path)

    # create output path
    out_name = f"annotated_{unique_id}.mp4"
    output_path = os.path.join(OUTPUT_DIR, out_name)

    try:
        report = process_video(input_path, output_path, show_window=False)
    except Exception as e:
        # clean up files on error
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        return jsonify({"error": str(e)}), 500

    # Return JSON with counts and link to download annotated video
    response = {
        "workflow_name": "Object Detection and Counting Pipeline",
        "input_file": os.path.basename(input_path),
        "annotated_video": f"/download/{out_name}",
        "report": report
    }
    return jsonify(response), 200



import os
import time
import uuid
from flask import Flask, request, jsonify, send_from_directory, abort

from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort   # pip install sort-tracker

# -------- CONFIG ----------
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
ALLOWED_EXT = {"mp4", "avi", "mov", "mkv"}
MODEL_WEIGHTS = "yolov8n.pt"
CONF_THRESH = 0.45
IOU_NMS = 0.45
LINE_POSITION = 0.5  # fraction of frame height
MIN_BOX_AREA = 500
MAX_PROCESS_SECONDS = 300  # safety timeout per request (optional)
# --------------------------

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
app = Flask(__name__)

# Load model once (global) to speed requests
model = YOLO(MODEL_WEIGHTS)
model.conf = CONF_THRESH
model.iou = IOU_NMS

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def centroid_from_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def process_video(input_path, output_path, show_window=False):
    """
    Process the input video, write annotated output, and return a report dict.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    line_y = int(height * LINE_POSITION)
    counts = {"in": 0, "out": 0}
    track_last_centroid_y = {}
    track_events = {}  # track_id -> list of events {"frame":..,"time":..,"event":"enter"/"exit"}

    frame_idx = 0
    start_time = time.time()
    last_frame_time = start_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # run detector
        detections = []
        results = model(frame)  # single call returns batch of 1
        for box in results[0].boxes:
            cls = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            if cls != 0:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_BOX_AREA:
                continue
            detections.append([x1, y1, x2, y2, conf])

        dets_np = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

        # tracking
        tracked = tracker.update(dets_np)

        for t in tracked:
            x1, y1, x2, y2, track_id = t
            x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
            cx, cy = centroid_from_box((x1, y1, x2, y2))
            prev_y = track_last_centroid_y.get(track_id, None)

            # check crossing
            if prev_y is not None:
                if prev_y < line_y and cy >= line_y:
                    # entering direction (downwards)
                    counts["in"] += 1
                    track_events.setdefault(track_id, []).append({
                        "frame": frame_idx,
                        "time": time.time() - start_time,
                        "event": "enter"
                    })
                elif prev_y > line_y and cy <= line_y:
                    counts["out"] += 1
                    track_events.setdefault(track_id, []).append({
                        "frame": frame_idx,
                        "time": time.time() - start_time,
                        "event": "exit"
                    })

            track_last_centroid_y[track_id] = cy

            # draw annotations
            cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 200, 64), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        # draw UI
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
        cv2.putText(frame, f"In: {counts['in']}  Out: {counts['out']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out.write(frame)

        if show_window:
            cv2.imshow("annot", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # optional safety timeout
        if MAX_PROCESS_SECONDS and (time.time() - start_time) > MAX_PROCESS_SECONDS:
            break

    cap.release()
    out.release()
    if show_window:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    fps_processed = frame_idx / elapsed if elapsed > 0 else 0.0

    # Build report
    report = {
        "source": os.path.basename(input_path),
        "output_video": os.path.basename(output_path),
        "frames_processed": frame_idx,
        "elapsed_seconds": round(elapsed, 2),
        "avg_fps": round(fps_processed, 2),
        "counts": counts,
        "track_events": track_events
    }
    return report

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file provided (form field 'file')"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"file type not allowed, allowed: {ALLOWED_EXT}"}), 400

    filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex[:8]
    saved_name = f"{unique_id}_{filename}"
    input_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(input_path)

    # create output path
    out_name = f"annotated_{unique_id}.mp4"
    output_path = os.path.join(OUTPUT_DIR, out_name)

    try:
        report = process_video(input_path, output_path, show_window=False)
    except Exception as e:
        # clean up files on error
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        return jsonify({"error": str(e)}), 500

    # Return JSON with counts and link to download annotated video
    response = {
        "workflow_name": "Object Detection and Counting Pipeline",
        "input_file": os.path.basename(input_path),
        "annotated_video": f"/download/{out_name}",
        "report": report
    }
    return jsonify(response), 200

@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    # secure: ensure file exists in OUTPUT_DIR
    if not os.path.exists(os.path.join(OUTPUT_DIR, filename)):
        abort(404)
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)