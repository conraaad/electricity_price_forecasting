
import uuid
from django.db import models

class User(models.Model):
    email = models.EmailField(unique=True)
    api_key = models.CharField(max_length=64, unique=True, default=uuid.uuid4)

    def __str__(self):
        return self.email