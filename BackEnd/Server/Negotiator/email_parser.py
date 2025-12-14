import os
import email
from email.header import decode_header
from config import ATTACHMENT_DIR

def decode_header_value(header):
    try:
        decoded_header, encoding = decode_header(header)[0]
        return decoded_header.decode(encoding) if encoding else str(decoded_header)
    except:
        return header

def extract_email_content(msg):
    subject = decode_header_value(msg.get("Subject", ""))
    sender = decode_header_value(msg.get("From", ""))
    body = ""
    attachments = []
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in content_disposition:
                body = part.get_payload(decode=True).decode()
            elif "attachment" in content_disposition:
                filename = decode_header_value(part.get_filename())
                if filename:
                    attachments.append(save_attachment(part, filename))
    else:
        body = msg.get_payload(decode=True).decode()

    return subject, sender, body, attachments

def save_attachment(part, filename):
    os.makedirs(ATTACHMENT_DIR, exist_ok=True)
    filepath = os.path.join(ATTACHMENT_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(part.get_payload(decode=True))
    return filepath