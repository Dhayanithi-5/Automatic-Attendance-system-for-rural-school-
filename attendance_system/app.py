"""
Automatic Attendance System for Rural Schools
-----------------------------------------------
A self-contained Flask web application that uses a normal USB webcam plus
OpenCV face detection/recognition to take student attendance automatically.

Designed for low-resource environments:
  - No internet connection required after installation.
  - No cloud services, subscriptions, or paid APIs.
  - Runs on an ordinary laptop / desktop PC.
  - All data stored locally in a single SQLite file (attendance.db).
  - Attendance can be exported to CSV for record keeping / sharing over
    SMS-friendly channels or on a USB drive.

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import os
import io
import csv
from functools import wraps
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, Response, jsonify, send_file
)

from utils import db_utils, face_utils

app = Flask(__name__)
app.secret_key = "rural-school-attendance-secret-key-change-me"

# ---- simple hard-coded admin login (change for real deployment) ----
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


# --------------------------------------------------------------------- auth
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------- dashboard
@app.route("/")
@login_required
def dashboard():
    stats = db_utils.get_dashboard_stats()
    today_attendance = db_utils.get_today_attendance()
    return render_template("dashboard.html", stats=stats, today_attendance=today_attendance)


# ----------------------------------------------------------------- students
@app.route("/students")
@login_required
def students():
    all_students = db_utils.get_all_students()
    trained = os.path.exists(face_utils.TRAINER_FILE)
    return render_template("students.html", students=all_students, trained=trained)


@app.route("/students/delete/<student_id>")
@login_required
def delete_student(student_id):
    db_utils.delete_student(student_id)
    flash(f"Student {student_id} deleted.", "success")
    return redirect(url_for("students"))


@app.route("/train")
@login_required
def train():
    success, message, num_students, num_images = face_utils.train_model()
    flash(message + (f" ({num_students} students, {num_images} images)" if success else ""),
          "success" if success else "error")
    return redirect(url_for("students"))


# ------------------------------------------------------------ registration
@app.route("/register", methods=["GET", "POST"])
@login_required
def register():
    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip()
        name = request.form.get("name", "").strip()
        class_name = request.form.get("class_name", "").strip()
        roll_no = request.form.get("roll_no", "").strip()
        guardian_phone = request.form.get("guardian_phone", "").strip()

        if not student_id or not name:
            flash("Student ID and Name are required.", "error")
            return redirect(url_for("register"))

        if db_utils.student_id_exists(student_id):
            flash("A student with this ID already exists.", "error")
            return redirect(url_for("register"))

        db_utils.add_student(student_id, name, class_name, roll_no, guardian_phone)
        return redirect(url_for("capture", student_id=student_id))

    return render_template("register.html")


@app.route("/capture/<student_id>")
@login_required
def capture(student_id):
    student = db_utils.get_student_by_student_id(student_id)
    if not student:
        flash("Student not found.", "error")
        return redirect(url_for("register"))
    return render_template("capture.html", student=student)


@app.route("/video_feed_register/<student_id>")
@login_required
def video_feed_register(student_id):
    return Response(
        face_utils.gen_register_frames(student_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/capture_progress")
@login_required
def capture_progress_route():
    return jsonify(face_utils.capture_progress)


# ------------------------------------------------------------- attendance
@app.route("/attendance")
@login_required
def attendance():
    trained = os.path.exists(face_utils.TRAINER_FILE)
    return render_template("attendance.html", trained=trained)


@app.route("/video_feed_attendance")
@login_required
def video_feed_attendance():
    return Response(
        face_utils.gen_attendance_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/attendance_log")
@login_required
def attendance_log_route():
    return jsonify(face_utils.attendance_log)


# ---------------------------------------------------------------- reports
@app.route("/reports")
@login_required
def reports():
    date_filter = request.args.get("date", "")
    class_filter = request.args.get("class_name", "")
    records = db_utils.get_attendance_filtered(
        date_filter or None, class_filter or None
    )
    all_classes = sorted({s["class"] for s in db_utils.get_all_students() if s["class"]})
    return render_template(
        "reports.html",
        records=records,
        all_classes=all_classes,
        date_filter=date_filter,
        class_filter=class_filter,
    )


@app.route("/export_csv")
@login_required
def export_csv():
    date_filter = request.args.get("date", "")
    class_filter = request.args.get("class_name", "")
    records = db_utils.get_attendance_filtered(date_filter or None, class_filter or None)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student ID", "Name", "Class", "Date", "Time", "Status"])
    for r in records:
        writer.writerow([r["student_id"], r["name"], r["class"], r["date"], r["time"], r["status"]])

    mem = io.BytesIO(output.getvalue().encode("utf-8"))
    filename = f"attendance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name=filename)


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    db_utils.init_db()
    print("=" * 60)
    print(" Automatic Attendance System for Rural Schools")
    print(" Open your browser at: http://127.0.0.1:5000")
    print(" Default login -> username: admin | password: admin123")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
