# Django OCR Application

A simple Django application for Optical Character Recognition (OCR) using PaddleOCR.

## Features

- Upload images for OCR processing
- Extract text from images using PaddleOCR
- RESTful API for OCR processing
- API documentation with Swagger UI

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install djangorestframework drf-spectacular paddleocr opencv-python
   ```
3. Run migrations:
   ```
   python manage.py migrate
   ```
4. Start the development server:
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

### API Documentation

- API Schema: `/api/schema/`
- Swagger UI: `/api/schema/swagger-ui/`
- ReDoc: `/api/schema/redoc/`
