from django.urls import path
from . import views

urlpatterns = [
    path('health', views.health_check, name='health'),
    path('register', views.RegisterView.as_view(), name='register'),
    path('predict', views.PredictView.as_view(), name='predict'),
    path('features', views.FeaturesView.as_view(), name='features'),  # Add this line
]
