from django.urls import path
from . import views
from .views import OCRImageAPIView, OCRMultipleImagesAPIView, OCRVideoAPIView, OCRVideoStatusAPIView

app_name = 'ocr_app'  # Optional: for namespacing URLs

urlpatterns = [
    path('api/process-image/', OCRImageAPIView.as_view(), name='ocr_image_api'),
    path('api/process-multiple-images/', OCRMultipleImagesAPIView.as_view(), name='ocr_multiple_images_api'),
    path('api/process-video/', OCRVideoAPIView.as_view(), name='ocr_video_api'),
    path('api/process-video/<str:job_id>/status/', OCRVideoStatusAPIView.as_view(), name='ocr_video_status_api'),
]
