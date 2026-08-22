"""
services/email_service.py
Web3Forms integration for the contact form.
"""

import os
import requests
import structlog

logger = structlog.get_logger()

WEB3FORMS_URL = "https://api.web3forms.com/submit"


def send_contact_email(name: str, email: str, message: str) -> tuple[bool, str]:
    """
    Send a contact form submission via Web3Forms.

    Returns: (success: bool, message: str)
    """
    key = os.getenv("WEB3FORMS_KEY")
    if not key:
        logger.warning("web3forms_key_missing")
        return False, "Email service is not configured. Please try again later."

    payload = {
        "access_key": key,
        "name": name,
        "email": email,
        "message": message,
        "subject": f"Autolysis Contact: Message from {name}",
        "from_name": "Autolysis Contact Form",
        "botcheck": "",  # Honeypot field — must be empty
    }

    try:
        resp = requests.post(WEB3FORMS_URL, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("success"):
            logger.info("contact_email_sent", from_email=email)
            return True, "Your message has been sent successfully!"
        else:
            logger.warning("web3forms_failure", response=data)
            return False, data.get("message", "Failed to send message. Please try again.")

    except requests.exceptions.Timeout:
        logger.error("web3forms_timeout")
        return False, "Request timed out. Please try again."
    except Exception as e:
        logger.error("web3forms_error", error=str(e))
        return False, "An error occurred. Please try again later."
