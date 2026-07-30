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
import html
import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional


def send_email(subject: str, body: str, image: Optional[bytes] = None) -> None:
    """Send the report. When `image` is given (a PNG from report_chart),
    the message becomes multipart/alternative: a plain-text part that is
    exactly what it always was, plus an HTML part with the chart embedded
    inline.

    Inline via Content-ID rather than a link on purpose -- a linked image
    needs somewhere to host it and most clients block remote images by
    default, so the chart would arrive as a broken box. The plain-text
    part means a client that refuses HTML entirely still gets the whole
    report, just without the picture."""
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

    if image:
        cid = "report-chart"
        msg.add_alternative(_html_body(body, cid), subtype="html")
        # add_related must target the HTML part, not the message root, or
        # the image lands as a sibling attachment and `cid:` never resolves.
        msg.get_payload()[-1].add_related(
            image, maintype="image", subtype="png", cid=f"<{cid}>",
        )

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls(context=context)
        server.login(user, password)
        server.send_message(msg)


def _html_body(body: str, cid: str) -> str:
    """Deliberately plain and inline-styled. Email clients strip <style>
    blocks and ignore most of a stylesheet, so anything not written as a
    style attribute here would simply not apply. `body` is LLM output and
    is escaped before it ever reaches the markup."""
    safe = html.escape(body)
    return f"""\
<html><body style="margin:0;padding:24px;background:#f4f5f7;">
  <div style="max-width:680px;margin:0 auto;background:#ffffff;border-radius:12px;
              padding:26px 28px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
    <div style="font-size:13px;font-weight:600;color:#F7931A;letter-spacing:.06em;
                text-transform:uppercase;margin-bottom:14px;">Crypto Agent</div>
    <div style="font-size:16px;line-height:1.65;color:#1c2330;white-space:pre-wrap;">{safe}</div>
    <img src="cid:{cid}" alt="Contexto de mercado"
         style="display:block;width:100%;max-width:624px;height:auto;margin-top:22px;border-radius:10px;" />
    <div style="margin-top:20px;font-size:11.5px;line-height:1.6;color:#8a93a3;">
      Este informe solo describe datos de mercado. No es una se&ntilde;al de compra o venta:
      todas las decisiones y ejecuciones son manuales.
    </div>
  </div>
</body></html>"""
