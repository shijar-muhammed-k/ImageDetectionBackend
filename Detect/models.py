from django.db import models
from User.models import Profile

# Create your models here.
class PredictionImage(models.Model):
    user = models.ForeignKey(Profile, on_delete=models.CASCADE)
    image = models.FileField(upload_to='imageToPredict')
    date = models.DateField(auto_now_add=True)
    prediction = models.CharField(max_length=50, null=True, blank=True)
    confidence = models.CharField(max_length=10, null=True, blank=True)
    reported = models.BooleanField(default=False)
    reported_date = models.DateField(null=True, blank=True)

    def __str__(self):
        return self.user.first_name
         