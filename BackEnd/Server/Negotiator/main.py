from email_client import send_email
from negotiation_handler import negotiate_contract

def gather_negotiation_details():
    return {
        'sender_name': input("Enter your name: "),
        'sender_company': input("Enter company name: "),
        'sender_email': input("Enter your email: "),
        'sender_password': input("Enter email password: "),
        'receiver_email': input("Enter recipient email: "),
        'contract_type': input("Enter contract type: ")
    }

def create_initial_draft(details):
    return f"""Dear Recipient,

We propose a {details['contract_type']} agreement between {details['sender_company']} and your organization.
Please review the attached draft and respond with your feedback.

Best regards,
{details['sender_name']}
"""

def start_negotiation():
    details = gather_negotiation_details()
    message = create_initial_draft(details)
    
    if send_email(details['sender_email'], details['sender_password'],
                 details['receiver_email'], "Contract Draft", message,
                 input("Attachment path (Enter to skip): ")):
        negotiate_contract(details['sender_email'], details['sender_password'],
                          details['receiver_email'])

if __name__ == "__main__":
    start_negotiation()