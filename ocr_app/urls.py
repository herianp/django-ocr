from django.urls import path
from . import views

app_name = 'ocr_app'  # Optional: for namespacing URLs

urlpatterns = [
    path('api/process-image/', views.ocr_image_endpoint, name='ocr_image_api'),
]