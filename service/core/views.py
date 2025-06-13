import os
import sys
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
import json

from core.utils import get_random_date_2023
from src.predict_final.predict_random_forest import predict_from_model_and_date


from .models import User

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def health_check(request):
    health_status = {
        'status': 'OK',
        'service': 'electricity-price-forecasting',
        'version': '1.0.0',
    }
    return JsonResponse(health_status, status=200)

@method_decorator(csrf_exempt, name='dispatch')
class RegisterView(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
            
        email = data.get('email')

        if not email:
            return JsonResponse({'error': 'Email field is required'}, status=400)
        
        if not email.strip():
            return JsonResponse({'error': 'Email cannot be empty'}, status=400)

        try:
            validate_email(email)
        except ValidationError:
            return JsonResponse({'error': 'Invalid email format'}, status=400)

        user, created = User.objects.get_or_create(email=email)
        return JsonResponse({
            'email': user.email,
            'api_key': user.api_key
        }, status=201)

class PredictView(View):
    def get(self, request):
        api_key = request.headers.get('Authorization')

        if not api_key or not User.objects.filter(api_key=api_key).exists():
            return JsonResponse({'error': 'Unauthorized: API key is required'}, status=401)

        try:
            # date = '2023-10-01'

            #get random date within 2023
            date = get_random_date_2023()
            
            # Call the prediction function
            model_path = os.path.join(ROOT_DIR, '../model/src/predict_final/models/random_forest_model.joblib')

            predictions = predict_from_model_and_date(model_path, date)
            print(f"Model path: {model_path}")
            
            return JsonResponse(predictions, status=200)
            
        except Exception as e:
            return JsonResponse({'error': f'Prediction failed: {str(e)}'}, status=500)
