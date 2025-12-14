import time
from email_client import connect_to_mail
from email_parser import extract_email_content

def negotiate_contract(sender_email, sender_password, receiver_email):
    mail = connect_to_mail(sender_email, sender_password)
    if not mail:
        return False

    last_uid = 0
    while True:
        print("\nChecking for replies...")
        msg, new_uid = fetch_latest_email(mail, receiver_email, last_uid)
        
        if msg:
            last_uid = new_uid
            process_email_response(msg)
            
            if is_contract_accepted(msg):
                print("\n✅ Contract finalized!")
                return True
        time.sleep(30)
def fetch_latest_email(mail, recipient_email, last_uid):
    mail.select("inbox")
    status, messages = mail.search(None, f'FROM "{recipient_email}" UNSEEN')
    if status != "OK":
        return None, last_uid

    email_ids = messages[0].split()
    for email_id in reversed(email_ids):
        try:
            email_id_int = int(email_id)
            if email_id_int > last_uid:
                status, msg_data = mail.fetch(email_id, "(RFC822)")
                if status == "OK":
                    return email.message_from_bytes(msg_data[0][1]), email_id_int
        except ValueError:
            continue
    return None, last_uid

def process_email_response(msg):
    subject, sender, body, attachments = extract_email_content(msg)
    print(f"\nNew Email From: {sender}")
    print(f"Subject: {subject}")
    print(f"Message: {body.strip()}")
    if attachments:
        print(f"Attachments: {attachments}")

def is_contract_accepted(msg):
    body = extract_email_content(msg)[2].lower()
    return any(word in body for word in ["accept", "approved", "agreed"])