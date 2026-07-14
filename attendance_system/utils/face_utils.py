"""
face_utils.py
-------------
All computer-vision logic for the Automatic Attendance System.

Uses OpenCV's built-in Haar Cascade for face DETECTION and the
LBPH (Local Binary Patterns Histograms) algorithm for face RECOGNITION.

Both are shipped inside opencv-contrib-python, so the system needs
NO internet access, NO cloud API, and NO GPU. This makes it realistic
to deploy on an ordinary desktop/laptop with a simple USB webcam in a
rural school that has little or no internet connectivity.
"""

import os
import cv2
import json
import time
import numpy as np

from utils import db_utils

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_FILE = os.path.join(TRAINER_DIR, "labels.json")

FACE_SIZE = (200, 200)
SAMPLES_PER_STUDENT = 40
CONFIDENCE_THRESHOLD = 70  # LBPH: LOWER value = more confident match

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# progress tracker shared with the frontend while a capture is in progress
capture_progress = {"count": 0, "total": SAMPLES_PER_STUDENT, "done": False}
attendance_log = []  # rolling list of most-recently recognised names this session


def _open_camera():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cam


def gen_register_frames(student_id):
    """
    Generator used by Flask to stream MJPEG video while capturing
    face samples for a NEW student. Saves cropped grayscale face
    images to dataset/<student_id>/.
    """
    global capture_progress
    capture_progress = {"count": 0, "total": SAMPLES_PER_STUDENT, "done": False}

    student_dir = os.path.join(DATASET_DIR, student_id)
    os.makedirs(student_dir, exist_ok=True)

    cam = _open_camera()
    count = 0
    last_capture_time = 0

    try:
        while count < SAMPLES_PER_STUDENT:
            ok, frame = cam.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = _face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                # only save a new sample every ~150ms to get varied poses
                if time.time() - last_capture_time > 0.15:
                    face_img = gray[y:y + h, x:x + w]
                    face_img = cv2.resize(face_img, FACE_SIZE)
                    count += 1
                    cv2.imwrite(os.path.join(student_dir, f"{count}.jpg"), face_img)
                    last_capture_time = time.time()
                    capture_progress["count"] = count
                break  # only handle the largest/first face per frame

            cv2.putText(
                frame, f"Captured: {count}/{SAMPLES_PER_STUDENT}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2,
            )
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

            if count >= SAMPLES_PER_STUDENT:
                break
    finally:
        cam.release()
        capture_progress["done"] = True
        db_utils.update_photo_count(student_id, count)


def train_model():
    """
    Trains the LBPH recognizer on every image inside dataset/<student_id>/...
    Returns (success: bool, message: str, num_students: int, num_images: int)
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}  # numeric label -> student_id
    next_label = 0

    if not os.path.isdir(DATASET_DIR):
        return False, "No dataset found. Please register students first.", 0, 0

    student_folders = [
        d for d in sorted(os.listdir(DATASET_DIR))
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]

    if not student_folders:
        return False, "No students registered yet.", 0, 0

    for student_id in student_folders:
        folder = os.path.join(DATASET_DIR, student_id)
        images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
        if not images:
            continue
        label_map[next_label] = student_id
        for img_name in images:
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, FACE_SIZE)
            faces.append(img)
            labels.append(next_label)
        next_label += 1

    if not faces:
        return False, "No valid face images found to train on.", 0, 0

    recognizer.train(faces, np.array(labels))
    recognizer.write(TRAINER_FILE)
    with open(LABELS_FILE, "w") as f:
        json.dump(label_map, f)

    return True, "Model trained successfully.", len(label_map), len(faces)


def _load_recognizer():
    if not os.path.exists(TRAINER_FILE) or not os.path.exists(LABELS_FILE):
        return None, None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)
    with open(LABELS_FILE) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    return recognizer, label_map


def gen_attendance_frames():
    """
    Generator used by Flask to stream MJPEG video while running LIVE
    face recognition. Marks attendance in the database automatically
    the first time each student is recognised each day.
    """
    global attendance_log
    attendance_log = []

    recognizer, label_map = _load_recognizer()
    if recognizer is None:
        # produce a single frame explaining the model isn't trained yet
        blank = 255 * np.ones((320, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Please train the model first (Students > Train Model)",
                    (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        ret, buffer = cv2.imencode(".jpg", blank)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        return

    cam = _open_camera()
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = _face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))

            for (x, y, w, h) in faces:
                face_img = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
                label, confidence = recognizer.predict(face_img)

                if confidence < CONFIDENCE_THRESHOLD and label in label_map:
                    student_id = label_map[label]
                    student = db_utils.get_student_by_student_id(student_id)
                    if student:
                        newly_marked = db_utils.mark_attendance(
                            student["student_id"], student["name"], student["class"]
                        )
                        display_name = f"{student['name']}"
                        color = (0, 200, 0)
                        if newly_marked:
                            entry = f"{student['name']} ({student['class']}) marked Present"
                            if entry not in attendance_log:
                                attendance_log.insert(0, entry)
                                attendance_log[:] = attendance_log[:8]
                    else:
                        display_name = "Unknown"
                        color = (0, 0, 220)
                else:
                    display_name = "Unknown"
                    color = (0, 0, 220)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, display_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        cam.release()
