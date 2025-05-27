from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings  # For MEDIA_ROOT
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR  # Keep PaddleOCR from paddleocr library
import time
import json  # For potential output formatting if needed, though JsonResponse handles it

# --- PaddleOCR Instance Initialization (from previous Django example) ---
OCR_ENGINE_INSTANCE = None
try:
    print("Initializing PaddleOCR (Django App)...")
    # Using 'en' for English as a default, adjust if your script's 'cs' (Czech) is always needed
    # If your script used 'cs', change lang='cs' here.
    OCR_ENGINE_INSTANCE = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
    print("PaddleOCR initialized successfully.")
except ValueError as e_init:
    if 'use_gpu' in str(e_init).lower() or 'show_log' in str(e_init).lower():
        print(f"PaddleOCR init failed due to unsupported params ({e_init}), trying fallback...")
        try:
            OCR_ENGINE_INSTANCE = PaddleOCR(use_angle_cls=True, lang='en')  # Fallback
            print("PaddleOCR initialized successfully (fallback).")
        except Exception as e_fallback:
            print(f"CRITICAL: PaddleOCR fallback initialization failed: {e_fallback}")
    else:
        print(f"CRITICAL: PaddleOCR initialization failed (non-param error): {e_init}")
except Exception as e_general:
    print(f"CRITICAL: General PaddleOCR initialization error: {e_general}")


# --- Functions adapted from your script ---
# (seconds_to_time_format and extract_frame_info are less relevant for single image upload
# unless you pass that timing info separately)

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

    # PaddleOCR's ocr method returns a list like:
    # [[[points, (text, confidence)], [points, (text, confidence)], ...]]
    # For a single image, the outer list has one element.
    ocr_raw_result = ocr_engine.ocr(image_cv_array)  # Pass the numpy array

    processed_results = []
    if ocr_raw_result and ocr_raw_result[0]:  # Check results for the first (and only) image
        lines = ocr_raw_result[0]
        if lines:
            for line_info in lines:
                # line_info is [bounding_box, (text, confidence_score)]
                text_content = line_info[1][0]
                confidence = line_info[1][1]
                bounding_box = line_info[0]  # list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

                processed_results.append({
                    'text': text_content,
                    'confidence': float(confidence),
                    'bounding_box': bounding_box
                })
    else:
        print('No text detected in the image by process_single_image_with_ocr.')

    return processed_results


@csrf_exempt
def ocr_image_endpoint(request):
    if OCR_ENGINE_INSTANCE is None:
        return JsonResponse({'error': 'OCR engine is not initialized. Check server logs.'}, status=503)

    if request.method == 'POST':
        print(request.FILES)
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image file found. Ensure key is "image".'}, status=400)

        uploaded_file = request.FILES['image']

        # Basic file type validation
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        _filename, ext = os.path.splitext(uploaded_file.name)  # Use _filename to avoid clash
        if ext.lower() not in allowed_extensions:
            return JsonResponse({'error': f'Invalid file type: {ext}. Allowed: {", ".join(allowed_extensions)}'},
                                status=400)

        try:
            image_data = uploaded_file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_cv is None:
                return JsonResponse({'error': 'Could not decode image data.'}, status=400)

            ocr_start_time = time.time()
            recognized_data = process_single_image_with_ocr(img_cv, OCR_ENGINE_INSTANCE)
            ocr_end_time = time.time()

            print(f"OCR processing for uploaded image took: {ocr_end_time - ocr_start_time:.2f} seconds")

            # Optional: Save the uploaded image if needed for debugging or records
            # This part is from your script's concept of frames_dir
            # For a Django app, you'd typically use Django's file storage system
            # or a dedicated media handling strategy.
            # Example: save to MEDIA_ROOT/temp_uploads/
            # temp_save_dir = os.path.join(settings.MEDIA_ROOT, 'temp_uploads')
            # os.makedirs(temp_save_dir, exist_ok=True)
            # unique_filename = f"{int(time.time())}_{uploaded_file.name}"
            # temp_file_path = os.path.join(temp_save_dir, unique_filename)
            # with open(temp_file_path, 'wb') as f_out:
            #     f_out.write(image_data)
            # print(f"Uploaded image temporarily saved to: {temp_file_path}")

            # The structure of your 'create_frame_result' can be adapted here
            # Since it's a single image, 'frame_num', 'startTime', 'endTime'
            # might not be relevant unless passed as extra data.
            # We'll directly return the list of recognized texts and their details.

            return JsonResponse({
                'message': 'OCR successful',
                'filename': uploaded_file.name,
                'ocr_data': recognized_data  # This now contains text, confidence, bbox
            }, status=200)

        except Exception as e:
            import traceback
            print(f"Error during OCR processing: {e}\n{traceback.format_exc()}")
            return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)
    else:
        return JsonResponse({'error': 'Only POST method allowed.'}, status=405)