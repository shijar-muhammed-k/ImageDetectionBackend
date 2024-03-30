# Generated by Django 5.0.2 on 2024-03-28 15:16

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("Detect", "0004_alter_predictionimage_image"),
        ("User", "0002_profile_created_date_profile_credit_points_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="predictionimage",
            name="user",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE, to="User.profile"
            ),
        ),
    ]