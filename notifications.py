import os
import smtplib
from email.message import EmailMessage
import logging

def send_login_email(user_email: str):
    """
    Send email alert when a user logs in.
    Make sure EMAIL_ADDRESS and EMAIL_PASS are set in environment variables.
    """
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
