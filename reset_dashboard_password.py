"""One-off: reset the dashboard login password via the Supabase admin API,
bypassing the (unimplemented) email-recovery flow. Run locally, once:

    .venv/bin/python reset_dashboard_password.py

Prompts for email and new password with getpass (hidden input, never
printed, never passed as a CLI arg) so neither ends up in shell history.
Uses SUPABASE_KEY (service_role) from .env -- delete this file after use.
"""
from getpass import getpass

from dotenv import load_dotenv
import os

load_dotenv()

from supabase import create_client

url = os.environ["SUPABASE_URL"]
service_key = os.environ["SUPABASE_KEY"]

email = input("Email del dashboard: ").strip()
new_password = getpass("Contraseña nueva: ")
confirm = getpass("Repetila: ")

if new_password != confirm:
    raise SystemExit("Las contraseñas no coinciden.")
if len(new_password) < 6:
    raise SystemExit("Supabase exige al menos 6 caracteres.")

client = create_client(url, service_key)

users = client.auth.admin.list_users()
match = next((u for u in users if u.email == email), None)
if match is None:
    raise SystemExit(f"No hay ningún usuario con el email {email!r}.")

client.auth.admin.update_user_by_id(match.id, {"password": new_password})
print("Listo. Ya podés loguearte en el dashboard con la contraseña nueva.")
