"""
email_notifier.py
------------------
Sends the report by email to your mailbox (e.g. Outlook/Hotmail).

IMPORTANT - why we do NOT send FROM your Outlook account:
Microsoft removed basic authentication (username+password) for sending
via SMTP from Outlook.com / Microsoft 365 accounts (fully rejected since
April 2026). The official alternative (registering an OAuth app in
Microsoft Entra) is disproportionate for a personal project.

Solution: send FROM another SMTP provider (Gmail with an "app password",
or Brevo, which is free) TO your Outlook mailbox. The result is the
same: the report lands in your Outlook inbox.

Required environment variables (see .env.example):
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_TO, EMAIL_FROM
"""
import os
import smtplib
import ssl
from email.message import EmailMessage


def send_email(subject: str, body: str) -> None:
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASSWORD"]
    to_addr = os.environ["EMAIL_TO"]
    from_addr = os.environ.get("EMAIL_FROM", user)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls(context=context)
        server.login(user, password)
        server.send_message(msg)
