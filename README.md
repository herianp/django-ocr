# Django OCR Application

A simple Django application for Optical Character Recognition (OCR) using PaddleOCR.

## Features

- Upload images for OCR processing
- Extract text from images using PaddleOCR
- Process videos from URLs and extract text from frames
- RESTful API for OCR processing
- Background job processing for long-running tasks
- API documentation with Swagger UI

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install djangorestframework drf-spectacular paddleocr==2.8.0 paddlepaddle==2.6.0 opencv-python requests
   ```
3. Install ffmpeg (required for video processing):
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`
4. Run migrations:
   ```
   python manage.py migrate
   ```
5. Start the development server:
   ```
   python manage.py runserver
   ```

## Usage

### API Endpoints

- OCR Processing (Single Image): `POST /ocr/api/process-image/`
  - Upload a single image file to extract text
  - Request: Form data with key `image`
  - Response: JSON with extracted text

- OCR Processing (Multiple Images): `POST /ocr/api/process-multiple-images/`
  - Upload multiple image files to extract text from each
  - Request: Form data with key `images[]`
  - Response: JSON array with extracted text for each image

- OCR Processing (Video): `POST /ocr/api/process-video/`
  - Process a video from a URL and extract text from frames
  - Request: JSON with `video_url` and optional `interval` (seconds between frames, default: 8)
  - Response: JSON with `job_id` for checking status
  - Example request:
    ```json
    {
      "video_url": "https://example.com/video.mp4",
      "interval": 8
    }
    ```

- OCR Video Job Status: `GET /ocr/api/process-video/{job_id}/status/`
  - Check the status of a video OCR job
  - Response: JSON with job status, progress, and results (if completed)

### API Documentation

- API Schema: `/api/schema/`
- Swagger UI: `/api/schema/swagger-ui/`
- ReDoc: `/api/schema/redoc/`

## Updates

### PaddleOCR and PaddlePaddle Update

The application has been updated to use newer versions of PaddleOCR and PaddlePaddle:
- paddleocr: 2.7.0.3 -> 2.8.0
- paddlepaddle: 2.5.1 -> 2.6.0

These updates may improve Czech language support and diacritics handling. To update your existing installation:

1. Update the packages:
   ```
   pip install --upgrade paddleocr==2.8.0 paddlepaddle==2.6.0
   ```

2. Restart the application:
   ```
   python manage.py runserver
   ```

If you encounter any issues after updating, you can revert to the previous versions:
   ```
   pip install paddleocr==2.7.0.3 paddlepaddle==2.5.1
   ```
