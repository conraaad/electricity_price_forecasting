# Generated by Django 5.1.3 on 2025-06-13 17:23

import uuid
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('email', models.EmailField(max_length=254, unique=True)),
                ('api_key', models.CharField(default=uuid.uuid4, max_length=64, unique=True)),
            ],
        ),
    ]
