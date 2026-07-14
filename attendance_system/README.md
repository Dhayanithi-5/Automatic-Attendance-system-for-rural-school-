# Automatic Attendance System for Rural Schools

A complete, working, offline-first attendance system that uses a webcam and
face recognition to automatically mark student attendance — no internet
connection or paid services required after setup.

## Why this design fits rural schools

| Constraint in rural schools            | How this project addresses it                          |
|-----------------------------------------|----------------------------------------------------------|
| Unreliable / no internet                | Runs 100% locally. No cloud APIs, no external calls.     |
| Limited budget                          | Uses only a normal PC/laptop + any USB webcam.            |
| Limited technical staff                 | Simple browser-based UI, one-click "Train Model" button. |
| Need for paper-free records             | SQLite database + one-click CSV export.                  |
| Power/hardware limitations              | Lightweight (OpenCV Haar Cascade + LBPH), no GPU needed. |

## How it works (technical overview)

1. **Registration** — Teacher enters a student's details, then the webcam
   captures ~40 face images (`utils/face_utils.py::gen_register_frames`),
   automatically cropped and saved to `dataset/<student_id>/`.
2. **Training** — Clicking "Train Recognition Model" builds an LBPH
   (Local Binary Patterns Histogram) face-recognition model from every
   captured photo (`utils/face_utils.py::train_model`) and saves it to
   `trainer/trainer.yml`.
3. **Attendance** — On the "Take Attendance" page, the webcam stream is
   analysed frame-by-frame: faces are detected with a Haar Cascade,
   then identified with the trained LBPH model. The first time a student
   is recognised each day, they're automatically marked "Present" in the
   database with a timestamp.
4. **Reports** — Attendance can be filtered by date/class and exported to
   CSV at any time for printing, sharing, or archiving.

## Project structure

```
attendance_system/
├── app.py                     # Flask application & routes
├── requirements.txt
├── README.md
├── utils/
│   ├── db_utils.py            # SQLite database helpers
│   └── face_utils.py          # Face detection / training / recognition
├── templates/                 # HTML pages (Jinja2, no external CDN needed)
│   ├── base.html, login.html, dashboard.html, students.html,
│   │   register.html, capture.html, attendance.html, reports.html
├── static/css/style.css       # Self-contained styling (offline)
├── dataset/                   # Captured face photos per student (auto-created)
└── trainer/                   # Trained recognition model (auto-created)
```

## Requirements

- Python 3.9+
- A webcam (built-in laptop camera or any USB webcam)
- ~200MB free disk space

## Installation

```bash
# 1. Extract this zip, then open a terminal inside the folder
cd attendance_system

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> Note: `opencv-contrib-python` includes the `cv2.face` module used for
> LBPH recognition. Do not install plain `opencv-python` instead, or
> face recognition training will fail with an AttributeError.

## Running the system

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in a browser on the same computer.

**Default login:** `admin` / `admin123`
(Change these constants near the top of `app.py` before real deployment.)

## Usage walkthrough

1. **Log in** with the admin credentials.
2. Go to **Students → Register New Student**, fill in the student's
   details and click "Save & Continue to Face Capture".
3. On the capture page, let the webcam collect ~40 photos of the
   student's face (turn the head slightly left/right for variety, keep
   the room well-lit). This takes about 15–20 seconds per student.
4. Repeat for every student in the class.
5. Click **Students → Train Recognition Model**. This only needs to be
   redone when you add/remove students.
6. Go to **Take Attendance** each morning/period. Students simply look
   at the camera for a moment; recognised faces are marked "Present"
   automatically and appear in the "Just Marked" list.
7. Go to **Reports** to filter by date/class and **Export CSV** for your
   records.

## Tips for best recognition accuracy

- Capture faces in good, even lighting (avoid strong backlight).
- Capture 30–50 images per student for a robust model.
- Re-train the model any time new students are added.
- Keep the camera at roughly eye level, 40–80 cm from the student.
- The recognition confidence threshold can be tuned in
  `utils/face_utils.py` (`CONFIDENCE_THRESHOLD`, default `70`; lower =
  stricter matching, fewer false positives).

## Extending this project

Ideas for future scope (useful for a project report / viva):

- **SMS/USSD alerts** to guardians using a local GSM modem (no internet
  needed) when a child is absent — the `guardian_phone` field is already
  captured for this purpose.
- **Multi-classroom sync** — periodically copy `attendance.db` to a USB
  drive or sync over a local Wi-Fi hotspot when multiple computers are
  used across classrooms.
- **Biometric fallback** — add fingerprint or RFID card scanning as a
  backup for days when lighting conditions are poor.
- **Solar-powered kiosk mode** — run on a low-power single-board
  computer (e.g. Raspberry Pi) for off-grid schools.

## Troubleshooting

| Problem                                   | Fix                                                              |
|--------------------------------------------|-------------------------------------------------------------------|
| `AttributeError: module 'cv2' has no attribute 'face'` | Install `opencv-contrib-python`, not `opencv-python`.       |
| Webcam not detected / black video          | Check camera permissions and that no other app is using the camera. |
| Poor recognition accuracy                  | Recapture photos in better lighting; recapture more samples; retrain. |
| "Please train the model first" message     | Register at least one student and click "Train Recognition Model". |

## License

This is an educational project template — free to use, modify, and
extend for coursework, demonstrations, or real deployments.
