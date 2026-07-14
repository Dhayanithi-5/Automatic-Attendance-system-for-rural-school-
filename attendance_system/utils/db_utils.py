"""
db_utils.py
------------
Handles all SQLite database operations for the Automatic Attendance System.
Uses plain sqlite3 (no external DB server needed) so the whole system can run
on a low-cost PC in a rural school with no internet connection.
"""

import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "attendance.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they do not already exist."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            class TEXT,
            roll_no TEXT,
            guardian_phone TEXT,
            registered_on TEXT,
            photos_captured INTEGER DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            name TEXT,
            class TEXT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT DEFAULT 'Present',
            UNIQUE(student_id, date)
        )
        """
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- students
def add_student(student_id, name, class_name, roll_no, guardian_phone):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO students (student_id, name, class, roll_no, guardian_phone, registered_on)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (student_id, name, class_name, roll_no, guardian_phone,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def update_photo_count(student_id, count):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE students SET photos_captured = ? WHERE student_id = ?", (count, student_id))
    conn.commit()
    conn.close()


def get_all_students():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM students ORDER BY class, name").fetchall()
    conn.close()
    return rows


def get_student_by_student_id(student_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM students WHERE student_id = ?", (student_id,)).fetchone()
    conn.close()
    return row


def get_student_by_db_id(db_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM students WHERE id = ?", (db_id,)).fetchone()
    conn.close()
    return row


def delete_student(student_id):
    conn = get_connection()
    conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
    conn.commit()
    conn.close()


def student_id_exists(student_id):
    return get_student_by_student_id(student_id) is not None


# --------------------------------------------------------------- attendance
def mark_attendance(student_id, name, class_name):
    """Marks attendance for today. Returns True if newly marked, False if already marked."""
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")
    conn = get_connection()
    cur = conn.cursor()
    existing = cur.execute(
        "SELECT id FROM attendance WHERE student_id = ? AND date = ?", (student_id, today)
    ).fetchone()
    if existing:
        conn.close()
        return False
    cur.execute(
        """INSERT INTO attendance (student_id, name, class, date, time, status)
           VALUES (?, ?, ?, ?, ?, 'Present')""",
        (student_id, name, class_name, today, now_time),
    )
    conn.commit()
    conn.close()
    return True


def get_today_attendance():
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM attendance WHERE date = ? ORDER BY time DESC", (today,)
    ).fetchall()
    conn.close()
    return rows


def get_attendance_filtered(date_filter=None, class_filter=None):
    query = "SELECT * FROM attendance WHERE 1=1"
    params = []
    if date_filter:
        query += " AND date = ?"
        params.append(date_filter)
    if class_filter:
        query += " AND class = ?"
        params.append(class_filter)
    query += " ORDER BY date DESC, time DESC"
    conn = get_connection()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def get_dashboard_stats():
    conn = get_connection()
    total_students = conn.execute("SELECT COUNT(*) c FROM students").fetchone()["c"]
    today = datetime.now().strftime("%Y-%m-%d")
    present_today = conn.execute(
        "SELECT COUNT(*) c FROM attendance WHERE date = ?", (today,)
    ).fetchone()["c"]
    total_classes = conn.execute(
        "SELECT COUNT(DISTINCT class) c FROM students WHERE class IS NOT NULL AND class != ''"
    ).fetchone()["c"]
    conn.close()
    absent_today = max(total_students - present_today, 0)
    return {
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": absent_today,
        "total_classes": total_classes,
    }
