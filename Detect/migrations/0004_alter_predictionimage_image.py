# Generated by Django 5.0.2 on 2024-03-28 15:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("Detect", "0003_predictionimage_reported_date"),
    ]

    operations = [
        migrations.AlterField(
            model_name="predictionimage",
            name="image",
            field=models.FileField(upload_to="imageToPredict"),
        ),
    ]
