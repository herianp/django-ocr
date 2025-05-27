from django.conf import settings  # For MEDIA_ROOT
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR  # Keep PaddleOCR from paddleocr library
import time

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiExample
from drf_spectacular.types import OpenApiTypes

# --- PaddleOCR Instance Initialization (from previous Django example) ---
OCR_ENGINE_INSTANCE = None

try:
    print("Initializing PaddleOCR (Django App)...")
    OCR_ENGINE_INSTANCE = PaddleOCR(use_textline_orientation=True, lang='cs', use_gpu=False, show_log=False)
    print("PaddleOCR initialized successfully.")
except ValueError as e_init:
    if 'use_gpu' in str(e_init).lower() or 'show_log' in str(e_init).lower():
        print(f"PaddleOCR init failed due to unsupported params ({e_init}), trying fallback...")
        try:
            OCR_ENGINE_INSTANCE = PaddleOCR(use_textline_orientation=True, lang='cs')  # Fallback
            print("PaddleOCR initialized successfully (fallback).")
        except Exception as e_fallback:
            print(f"CRITICAL: PaddleOCR fallback initialization failed: {e_fallback}")
    else:
        print(f"CRITICAL: PaddleOCR initialization failed (non-param error): {e_init}")
except Exception as e_general:
    print(f"CRITICAL: General PaddleOCR initialization error: {e_general}")


def process_single_image_with_ocr(image_cv_array, ocr_engine):
    """
    Processes a single image (as a NumPy array) with the given OCR engine.
    Returns a list of recognized text strings and their details.
    """
    if image_cv_array is None:
        print("Error: Input image array is None.")
        return None

    if ocr_engine is None:
        print("Error: OCR Engine not initialized.")
        return None

    ocr_raw_result = ocr_engine.predict(image_cv_array)
    return ocr_raw_result


class OCRImageAPIView(APIView):
    """
    API endpoint for OCR processing of uploaded images.
    """
    parser_classes = (MultiPartParser, FormParser)

    @extend_schema(
        description="Process an image with OCR to extract text",
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'image': {
                        'type': 'string',
                        'format': 'binary'
                    }
                },
                'required': ['image']
            }
        },
        responses={
            200: {
                'description': 'OCR processing successful',
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'},
                    'filename': {'type': 'string'},
                    'rec_texts': {
                        'type': 'array',
                        'items': {
                            'type': 'string'
                        }
                    }
                }
            },
            400: {
                'description': 'Bad request',
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            },
            405: {
                'description': 'Method not allowed',
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            },
            500: {
                'description': 'Server error',
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            },
            503: {
                'description': 'OCR engine not initialized',
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            }
        }
    )
    def post(self, request, *args, **kwargs):
        """
        Process an uploaded image with OCR and return the extracted text.
        """
        if OCR_ENGINE_INSTANCE is None:
            return Response(
                {'error': 'OCR engine is not initialized. Check server logs.'}, 
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        print(request.FILES)
        if 'image' not in request.FILES:
            return Response(
                {'error': 'No image file found. Ensure key is "image".'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        uploaded_file = request.FILES['image']

        # Basic file type validation
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        _filename, ext = os.path.splitext(uploaded_file.name)  # Use _filename to avoid clash
        if ext.lower() not in allowed_extensions:
            return Response(
                {'error': f'Invalid file type: {ext}. Allowed: {", ".join(allowed_extensions)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            image_data = uploaded_file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_cv is None:
                return Response(
                    {'error': 'Could not decode image data.'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            ocr_start_time = time.time()
            recognized_data = process_single_image_with_ocr(img_cv, OCR_ENGINE_INSTANCE)
            ocr_end_time = time.time()

            print(f"OCR processing for uploaded image took: {ocr_end_time - ocr_start_time:.2f} seconds")
            print(f"recognized_data: {recognized_data}")

            # Extract rec_texts from recognized_data based on the structure in the issue description
            rec_texts = []
            try:
                if recognized_data and isinstance(recognized_data, list) and len(recognized_data) > 0:
                    # Based on the issue description, rec_texts is a direct field in the object
                    if isinstance(recognized_data[0], dict):
                        if 'rec_texts' in recognized_data[0]:
                            rec_texts = recognized_data[0]['rec_texts']
                        else:
                            for key, value in recognized_data[0].items():
                                if key == 'rec_texts':
                                    rec_texts = value
                                    break
                                elif isinstance(value, dict) and 'rec_texts' in value:
                                    rec_texts = value['rec_texts']
                                    break
            except Exception as e:
                print(f"Error extracting rec_texts: {e}")
                # If we can't extract rec_texts, return the entire recognized_data for debugging
                return Response({
                    'message': 'OCR successful but could not extract rec_texts',
                    'filename': uploaded_file.name,
                    'error': str(e),
                    'recognized_data': str(recognized_data)[:1000]  # Limit the size for readability
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response({
                'message': 'OCR successful',
                'filename': uploaded_file.name,
                'rec_texts': rec_texts  # Only return the extracted text
            }, status=status.HTTP_200_OK)

        except Exception as e:
            import traceback
            print(f"Error during OCR processing: {e}\n{traceback.format_exc()}")
            return Response(
                {'error': f'An error occurred: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class OCRMultipleImagesAPIView(APIView):
    """
    API endpoint for OCR processing of multiple uploaded images.
    """
    parser_classes = (MultiPartParser, FormParser)

    @extend_schema(
        description="Process multiple images with OCR to extract text",
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'images': {
                        'type': 'array',
                        'items': {
                            'type': 'string',
                            'format': 'binary'
                        }
                    }
                },
                'required': ['images']
            }
        },
        responses={
            200: {
                'description': 'OCR processing successful',
                'type': 'object',
                'properties': {
                    'results': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'message': {'type': 'string'},
                                'filename': {'type': 'string'},
                                'rec_texts': {
                                    'type': 'array',
                                    'items': {
                                        'type': 'string'
                                    }
                                }
                            }
                        }
                    }
                }
            },
            400: {
                'description': 'Bad request',
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            },
            405: {
                'description': 'Method not allowed',
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            },
            500: {
                'description': 'Server error',
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            },
            503: {
                'description': 'OCR engine not initialized',
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            }
        }
    )
    def post(self, request, *args, **kwargs):
        """
        Process multiple uploaded images with OCR and return the extracted text for each image.
        """
        if OCR_ENGINE_INSTANCE is None:
            return Response(
                {'error': 'OCR engine is not initialized. Check server logs.'}, 
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        print(f"request.FILES: {request.FILES}")
        if not request.FILES:
            return Response(
                {'error': 'No image files found. Ensure keys are "images[]".'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        # Get all uploaded files
        uploaded_files = request.FILES.getlist('images')
        if not uploaded_files:
            # Try alternative key format
            uploaded_files = []
            for key in request.FILES:
                if key.startswith('images['):
                    uploaded_files.append(request.FILES[key])

            if not uploaded_files:
                return Response(
                    {'error': 'No image files found with key "images". Please check your request format.'},
                    status=status.HTTP_400_BAD_REQUEST
                )

        allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        results = []

        for uploaded_file in uploaded_files:
            # Basic file type validation
            _filename, ext = os.path.splitext(uploaded_file.name)
            if ext.lower() not in allowed_extensions:
                results.append({
                    'message': 'OCR failed',
                    'filename': uploaded_file.name,
                    'error': f'Invalid file type: {ext}. Allowed: {", ".join(allowed_extensions)}'
                })
                continue

            try:
                image_data = uploaded_file.read()
                nparr = np.frombuffer(image_data, np.uint8)
                img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img_cv is None:
                    results.append({
                        'message': 'OCR failed',
                        'filename': uploaded_file.name,
                        'error': 'Could not decode image data.'
                    })
                    continue

                ocr_start_time = time.time()
                recognized_data = process_single_image_with_ocr(img_cv, OCR_ENGINE_INSTANCE)
                ocr_end_time = time.time()

                print(f"OCR processing for {uploaded_file.name} took: {ocr_end_time - ocr_start_time:.2f} seconds")

                # Extract rec_texts from recognized_data
                rec_texts = []
                try:
                    if recognized_data and isinstance(recognized_data, list) and len(recognized_data) > 0:
                        if isinstance(recognized_data[0], dict):
                            if 'rec_texts' in recognized_data[0]:
                                rec_texts = recognized_data[0]['rec_texts']
                            else:
                                for key, value in recognized_data[0].items():
                                    if key == 'rec_texts':
                                        rec_texts = value
                                        break
                                    elif isinstance(value, dict) and 'rec_texts' in value:
                                        rec_texts = value['rec_texts']
                                        break
                except Exception as e:
                    print(f"Error extracting rec_texts for {uploaded_file.name}: {e}")
                    results.append({
                        'message': 'OCR successful but could not extract rec_texts',
                        'filename': uploaded_file.name,
                        'error': str(e)
                    })
                    continue

                results.append({
                    'message': 'OCR successful',
                    'filename': uploaded_file.name,
                    'rec_texts': rec_texts
                })

            except Exception as e:
                import traceback
                print(f"Error during OCR processing for {uploaded_file.name}: {e}\n{traceback.format_exc()}")
                results.append({
                    'message': 'OCR failed',
                    'filename': uploaded_file.name,
                    'error': str(e)
                })

        return Response({'results': results}, status=status.HTTP_200_OK)
