from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from chatbot import HealthChatbot
import logging
import os
import sys
import smtplib
from email.message import EmailMessage
from contextlib import contextmanager
from datetime import datetime
import csv

# ---------------- Logging Setup ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)

# ---------------- Suppress prints ----------------
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# ---------------- Flask Setup ----------------
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "../frontend/templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "../frontend/static")
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your_secret_key_here")

# ---------------- Paths Setup ----------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
USER_FILE = os.path.join(DATA_DIR, "users.csv")
APPOINTMENT_FILE = os.path.join(DATA_DIR, "appointments.csv")
DOCTORS_FILE = os.path.join(DATA_DIR, "doctors.csv")

json_files = [os.path.join(DATA_DIR, f) for f in ["diseases4.json", "diseases3.json"]]
ml_model_file = os.path.join(MODEL_DIR, "best_model.pkl")
train_csv = os.path.join(DATA_DIR, "Training.csv")
test_csv = os.path.join(DATA_DIR, "Testing.csv")

# ---------------- Ensure JSON datasets exist ----------------
existing_json_files = [f for f in json_files if os.path.exists(f)]
if not existing_json_files:
    logging.warning("⚠️ No dataset could be loaded. Chatbot will only use ML fallback.")

# ---------------- Initialize Chatbot ----------------
with suppress_stdout():
    chatbot = HealthChatbot(
        json_files=existing_json_files,
        ml_model_file=ml_model_file,
        train_csv=train_csv,
        test_csv=test_csv,
        tts_enabled=False
    )

for file in existing_json_files:
    logging.info(f"✅ Loaded dataset: {file}")

# ---------------- Helper: send login alert ----------------
def send_login_email(user_email: str):
    try:
        EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
        EMAIL_PASSWORD = os.environ.get("EMAIL_PASS")
        if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
            logging.warning("⚠️ Email credentials not set. Skipping email.")
            return

        msg = EmailMessage()
        msg["Subject"] = "New Login Alert"
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = user_email
        msg.set_content(
            f"Hello,\n\nYour account was just logged in.\nIf this wasn't you, secure your account."
        )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)

        logging.info(f"✅ Login alert sent to {user_email}")
    except Exception as e:
        logging.error(f"❌ Failed to send login email: {e}")

# ---------------- Helper: doctors loader ----------------
def load_doctors():
    doctors = []
    if not os.path.exists(DOCTORS_FILE):
        logging.warning(f"⚠️ Doctors file not found at {DOCTORS_FILE}")
        return doctors

    try:
        with open(DOCTORS_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("name") and row.get("specialization") and row.get("hospital") and row.get("address"):
                    doctors.append({
                        "name": row["name"].strip(),
                        "specialization": row["specialization"].strip(),
                        "hospital": row["hospital"].strip(),
                        "address": row["address"].strip()
                    })
        logging.info(f"✅ Loaded {len(doctors)} doctors from {DOCTORS_FILE}")
        if not doctors:
            logging.warning("⚠️ Doctors file is empty or only has header.")
    except Exception as e:
        logging.error(f"❌ Error reading doctors file {DOCTORS_FILE}: {e}")
    return doctors

# ---------------- API: Get doctors for frontend ----------------
@app.route("/api/doctors")
def api_doctors():
    doctors = load_doctors()
    return jsonify(doctors)

# ---------------- User management ----------------
def load_users():
    users = {}
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r", newline="", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    email, password = parts[0], parts[1]
                    profile = {"age": "", "gender": "", "name": "", "health_details": ""}
                    if len(parts) > 2:
                        keys = list(profile.keys())
                        for i, key in enumerate(keys):
                            if i + 2 < len(parts):
                                profile[key] = parts[i + 2]
                    users[email] = {"password": password, "profile": profile}
    return users

def save_user(email: str, password: str, profile: dict = None):
    os.makedirs(os.path.dirname(USER_FILE), exist_ok=True)
    profile_data = [
        profile.get("age",""), profile.get("gender",""),
        profile.get("name",""), profile.get("health_details","")
    ] if profile else [""]*4
    with open(USER_FILE, "a", newline="", encoding="utf-8") as f:
        f.write(",".join([email, password] + profile_data) + "\n")

def update_user_profile(email: str, profile: dict) -> bool:
    users = load_users()
    if email in users:
        users[email]["profile"] = profile
        with open(USER_FILE, "w", newline="", encoding="utf-8") as f:
            for e, data in users.items():
                line = [
                    e, data["password"], data["profile"]["age"],
                    data["profile"]["gender"], data["profile"]["name"],
                    data["profile"]["health_details"]
                ]
                f.write(",".join(line) + "\n")
        return True
    return False

# ---------------- Appointment management ----------------
def load_appointments():
    appointments = []
    if os.path.exists(APPOINTMENT_FILE):
        with open(APPOINTMENT_FILE, "r", newline="", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split(",")
                if len(parts) >= 5:
                    appointments.append({
                        "id": idx + 1,
                        "patient": parts[0],
                        "doctor": parts[1],
                        "date": parts[2],
                        "time": parts[3],
                        "status": parts[4]
                    })
    return appointments

def overwrite_appointments(appointments):
    os.makedirs(os.path.dirname(APPOINTMENT_FILE), exist_ok=True)
    with open(APPOINTMENT_FILE, "w", newline="", encoding="utf-8") as f:
        for appt in appointments:
            f.write(",".join([
                appt.get("patient",""),
                appt.get("doctor",""),
                appt.get("date",""),
                appt.get("time",""),
                appt.get("status","upcoming")
            ]) + "\n")

def save_appointment(patient, doctor, date, time, status="upcoming"):
    os.makedirs(os.path.dirname(APPOINTMENT_FILE), exist_ok=True)
    with open(APPOINTMENT_FILE, "a", newline="", encoding="utf-8") as f:
        f.write(",".join([patient, doctor, date, time, status]) + "\n")

def auto_update_status(appointments):
    now = datetime.now()
    for appt in appointments:
        try:
            if appt.get("status") == "upcoming":
                appt_dt_str = f"{appt.get('date','')} {appt.get('time','')}"
                try:
                    appt_dt = datetime.strptime(appt_dt_str, "%Y-%m-%d %H:%M")
                except ValueError:
                    appt_dt = datetime.strptime(appt_dt_str, "%Y-%m-%d %H:%M:%S")
                if now > appt_dt:
                    appt["status"] = "expired"
        except Exception as e:
            logging.debug(f"Couldn't parse appointment datetime '{appt.get('date')} {appt.get('time')}' : {e}")
    return appointments

# ---------------- Routes ----------------
@app.route("/")
def index():
    if "user_email" in session:
        users = load_users()
        profile = users.get(session["user_email"], {}).get("profile", {})
        return render_template("index.html", email=session["user_email"], profile=profile)
    return redirect(url_for("login"))

@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        password = (request.form.get("password") or "").strip()
        if not email or not password:
            flash("Please provide email and password", "error")
            return redirect(url_for("signup"))
        users = load_users()
        if email in users:
            flash("Email already registered!", "error")
            return redirect(url_for("signup"))
        save_user(email, password)
        flash("Account created successfully! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        password = (request.form.get("password") or "").strip()
        users = load_users()
        if email in users and users[email]["password"] == password:
            session["user_email"] = email
            send_login_email(email)
            flash("Login successful!", "success")
            return redirect(url_for("index"))
        flash("Invalid email or password", "error")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully", "success")
    return redirect(url_for("login"))

@app.route("/settings", methods=["GET","POST"])
def settings():
    if "user_email" not in session:
        return redirect(url_for("login"))
    users = load_users()
    email = session["user_email"]
    profile = users.get(email, {}).get("profile", {"name":"","age":"","gender":"","health_details":""})
    if request.method == "POST":
        profile["name"] = request.form.get("name","").strip()
        profile["age"] = request.form.get("age","").strip()
        profile["gender"] = request.form.get("gender","").strip()
        profile["health_details"] = request.form.get("health_details","").strip()
        update_user_profile(email, profile)
        flash("Profile updated successfully!", "success")
        return redirect(url_for("settings"))
    return render_template("settings.html", profile=profile)

@app.route("/appointment", methods=["GET","POST"])
def appointment():
    if "user_email" not in session:
        return redirect(url_for("login"))
    doctors = load_doctors()
    if request.method == "POST":
        patient = (request.form.get("patient") or "").strip()
        doctor = (request.form.get("doctor") or "").strip()
        date = (request.form.get("date") or "").strip()
        time = (request.form.get("time") or "").strip()
        if not (patient and doctor and date and time):
            flash("Please fill all fields", "error")
            return redirect(url_for("appointment"))
        save_appointment(patient, doctor, date, time)
        flash("Appointment booked successfully!", "success")
        return redirect(url_for("dashboard"))
    return render_template("appointment.html", doctors=doctors)

@app.route("/dashboard")
def dashboard():
    if "user_email" not in session:
        return redirect(url_for("login"))
    appointments = load_appointments()
    appointments = auto_update_status(appointments)
    overwrite_appointments(appointments)
    return render_template("dashboard.html", appointments=appointments)

@app.route("/cancel/<int:appt_id>")
def cancel(appt_id):
    if "user_email" not in session:
        return redirect(url_for("login"))
    appointments = load_appointments()
    for appt in appointments:
        if appt.get("id") == appt_id and appt.get("status") == "upcoming":
            appt["status"] = "cancelled"
            break
    overwrite_appointments(appointments)
    flash("Appointment updated.", "success")
    return redirect(url_for("dashboard"))

@app.route("/attend/<int:appt_id>")
def attend(appt_id):
    if "user_email" not in session:
        return redirect(url_for("login"))
    appointments = load_appointments()
    for appt in appointments:
        if appt.get("id") == appt_id and appt.get("status") == "upcoming":
            appt["status"] = "attended"
            break
    overwrite_appointments(appointments)
    flash("Appointment updated.", "success")
    return redirect(url_for("dashboard"))

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    user_input = data.get("message", "").strip()
    session_id = session.get("user_email", "default")
    if not user_input:
        return jsonify({"status":"error","reply":"Please enter a valid message."})
    response = chatbot.respond(user_input, session_id=session_id)
    return jsonify({"status":"success","reply":response})

# ---------------- Run Server ----------------
if __name__ == "__main__":
    logging.info("🚀 Starting HealthChatbot Server at http://127.0.0.1:5000 ...")
    app.run(debug=True, host="127.0.0.1", port=5000)
