# Generated by Django 5.0.2 on 2024-03-09 05:07

import django.contrib.auth.models
import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='Profile',
            fields=[
                ('user_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to=settings.AUTH_USER_MODEL)),
                ('phone', models.CharField(blank=True, max_length=15, null=True)),
                ('profile_image', models.ImageField(blank=True, null=True, upload_to='profile_image/')),
                ('profession', models.CharField(blank=True, max_length=25, null=True)),
                ('dob', models.DateField(blank=True, null=True)),
                ('gender', models.IntegerField(blank=True, choices=[(1, 'male'), (2, 'female'), (3, 'others')], null=True)),
            ],
            options={
                'verbose_name': 'user',
                'verbose_name_plural': 'users',
                'abstract': False,
            },
            bases=('auth.user',),
            managers=[
                ('objects', django.contrib.auth.models.UserManager()),
            ],
        ),
    ]
