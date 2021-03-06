import email, smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class EmailSender():
    my_logger = None
    
    def __init__(self, my_logger):
        self.my_logger = my_logger
        
    def send(self, config_json):
        subject = "An email with attachment from Python"
        body = "This is an email with attachment sent from Python"
        sender_email = "adib.mosharrof@gmail.com"
        receiver_email = "adib.mosharrof@gmail.com"
        password = "Main@100192"
        
        # Create a multipart message and set headers
        message = MIMEMultipart()
        message["From"] = config_json["sender"]
        message["To"] = ", ".join(config_json["receivers"])
        message["Subject"] = subject
        message["Bcc"] = ", ".join(config_json["receivers"])  # Recommended for mass emails
        
        # Add body to email
        message.attach(MIMEText(body, "plain"))
        
        filename = self.my_logger.handlers[0].baseFilename
        
        # Open PDF file in binary mode
        with open(filename, "rb") as attachment:
            # Add file as application/octet-stream
            # Email client can usually download this automatically as attachment
            log_file = attachment.read()
            part = MIMEBase("application", "octet-stream")
            part.set_payload(log_file)
        
        # Encode file in ASCII characters to send by email    
        encoders.encode_base64(part)
        
        # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {filename}",
        )
        
        # Add attachment to message and convert message to string
        message.attach(part)
        text = message.as_string()
        
        # Log in to server using secure context and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, text)
            
            