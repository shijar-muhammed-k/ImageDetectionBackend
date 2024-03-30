from django.core.mail import send_mail

def SendMail(data):
    print(data)
    send_mail(f'Reply Mail From Image Detection Admin - {data['date']}', data['message'], 'admin@codeanalyzer.com', [data['mail']], fail_silently=False)