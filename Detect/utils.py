# import tensorflow as tf
# from tensorflow.keras import models, layers
# import numpy as np

from django.core.mail import EmailMessage
import json
from Utilities.checkLab.src import imageCheck as check


def SendMail(data):
    
    with open('Detect\mail-template.JSON', 'r') as file:
        mail_data = json.load(file)
    
    mail_data['message'] = mail_data['message'].replace('{email}', data.user.email)
    mail_data['message'] = mail_data['message'].replace('{confidence}', str(data.confidence))  # Convert to string if needed
    mail_data['message'] = mail_data['message'].replace('{name}', data.user.first_name + " " + data.user.last_name)
    mail_data['message'] = mail_data['message'].replace('{phone}', data.user.phone)
    mail_data['subject'] = mail_data['subject'].replace('{name}', data.user.first_name + " " + data.user.last_name)
    subject = mail_data['subject']
    message = mail_data['message']
    from_email = 'admin@ImageDetection.com'
    recipient_list = [mail_data['mail-to']]

    email = EmailMessage(subject, message, from_email, recipient_list)

    file_path = 'media/'+str(data.image)
    email.attach_file(file_path)
    email.attach_file(file_path.replace('imageToPredict', 'result-image'))

    email.send(fail_silently=False)



def predict(image):
    check.setPath(image)
    result = check.check()
    return result