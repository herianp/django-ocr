# image_ocr_project/urls.py
from django.contrib import admin
from django.urls import path, include # Make sure include is imported
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('ocr/', include('ocr_app.urls')), # Include your app's URLs under the 'ocr/' prefix
]

# To serve media files during development (if you decide to save and serve them)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)