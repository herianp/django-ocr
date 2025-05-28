from django.conf import settings  # For MEDIA_ROOT
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR  # Keep PaddleOCR from paddleocr library
import time
import subprocess
import requests
import uuid
import json
from urllib.parse import urlparse

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
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


def seconds_to_time_format(seconds):
    """
    Convert seconds to 00:00:00 format.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def extract_frame_info(frame_file, interval_seconds):
    """
    Extract frame number and calculate start/end times from frame filename.
    Assumes frame filenames are like 'frame_0001.png', 'frame_0002.png', etc.
    """
    try:
        # Frame numbers from ffmpeg (like frame_0001.png) are 1-based
        frame_num = int(frame_file.split('_')[-1].split('.')[0])
        start_time = (frame_num - 1) * interval_seconds
        end_time = frame_num * interval_seconds
        return frame_num, start_time, end_time
    except (ValueError, IndexError) as e:
        print(f"Error: Could not extract frame number from {frame_file}: {e}")
        return None, None, None


def process_frame_with_ocr(frame_path, ocr_engine):
    """
    Process a single frame with OCR and return the recognized text.
    """
    img = cv2.imread(frame_path)
    if img is None:
        print(f"Error: Could not read image from {frame_path}")
        return None

    ocr_raw_result = ocr_engine.predict(img)

    # Extract rec_texts from the OCR result
    rec_texts = []
    try:
        if ocr_raw_result and isinstance(ocr_raw_result, list) and len(ocr_raw_result) > 0:
            if isinstance(ocr_raw_result[0], dict):
                if 'rec_texts' in ocr_raw_result[0]:
                    rec_texts = ocr_raw_result[0]['rec_texts']
                else:
                    for key, value in ocr_raw_result[0].items():
                        if key == 'rec_texts':
                            rec_texts = value
                            break
                        elif isinstance(value, dict) and 'rec_texts' in value:
                            rec_texts = value['rec_texts']
                            break
    except Exception as e:
        print(f"Error extracting rec_texts: {e}")
        return None

    return rec_texts


def create_frame_result(frame_num, start_time, end_time, recognized_texts):
    """
    Creates a result object for a frame with the given information.
    """
    return {
        "frame_num": frame_num,
        "chunk": {
            "startTime": seconds_to_time_format(start_time),
            "endTime": seconds_to_time_format(end_time)
        },
        "texts": recognized_texts
    }


def extract_frames(video_path, frames_dir, interval=8):
    """
    Extract frames from a video at the specified interval using ffmpeg.
    Returns True if successful, False otherwise.
    """
    print(f"Extracting frames from '{video_path}' to '{frames_dir}' at {interval}s interval...")

    # First, check if the video file exists and is not empty
    if not os.path.exists(video_path):
        print(f"Error: Video file does not exist: {video_path}")
        return False

    if os.path.getsize(video_path) == 0:
        print(f"Error: Video file is empty: {video_path}")
        return False

    # Get video information using ffprobe
    try:
        probe_command = [
            'ffprobe',
            '-v', 'error',
            '-show_format',
            '-show_streams',
            '-of', 'json',
            video_path
        ]
        probe_result = subprocess.run(probe_command, check=True, capture_output=True, text=True)
        video_info = json.loads(probe_result.stdout)
        print(f"Video info: {json.dumps(video_info, indent=2)}")
    except subprocess.CalledProcessError as e:
        print(f"Error getting video information: {e}")
        print(f"FFprobe stderr: {e.stderr}")
        # Continue with extraction even if probe fails
    except json.JSONDecodeError as e:
        print(f"Error parsing video information: {e}")
        # Continue with extraction even if JSON parsing fails
    except Exception as e:
        print(f"Unexpected error during video probing: {e}")
        # Continue with extraction even if there's an unexpected error

    # Extract frames using ffmpeg
    output_pattern = os.path.join(frames_dir, 'frame_%04d.png')
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps=1/{interval}',
        '-q:v', '2',
        output_pattern,
        '-loglevel', 'error'
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Frames extracted to '{frames_dir}'.")

        # Check if any frames were actually extracted
        frames = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not frames:
            print(f"Warning: No frames were extracted from the video. The video might be invalid or empty.")
            return False

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during frame extraction: {e}")
        print(f"FFmpeg stdout: {e.stdout}")
        print(f"FFmpeg stderr: {e.stderr}")

        # Provide more detailed error information based on common ffmpeg errors
        stderr = e.stderr.lower() if e.stderr else ""
        if "moov atom not found" in stderr:
            print("The video file is likely corrupted or incomplete. The 'moov atom' is missing, which is essential for MP4 playback.")
        elif "invalid data found when processing input" in stderr:
            print("The file does not appear to be a valid video file or is in a format ffmpeg cannot recognize.")
        elif "no such file or directory" in stderr:
            print(f"The video file path is invalid: {video_path}")
        elif "permission denied" in stderr:
            print(f"Permission denied when accessing the video file: {video_path}")

        return False
    except Exception as e:
        print(f"Unexpected error during frame extraction: {e}")
        return False


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


class OCRVideoAPIView(APIView):
    """
    API endpoint for OCR processing of videos from a URL.
    """
    parser_classes = (JSONParser,)

    @extend_schema(
        description="Process a video from a URL with OCR to extract text from frames",
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'video_url': {
                        'type': 'string',
                        'format': 'uri',
                        'description': 'URL of the video to process'
                    },
                    'interval': {
                        'type': 'integer',
                        'description': 'Interval in seconds between frames to extract (default: 8)',
                        'default': 8
                    }
                },
                'required': ['video_url']
            }
        },
        responses={
            200: {
                'description': 'OCR processing started',
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'},
                    'job_id': {'type': 'string'},
                    'status': {'type': 'string'}
                }
            },
            202: {
                'description': 'OCR processing in progress',
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'},
                    'job_id': {'type': 'string'},
                    'status': {'type': 'string'}
                }
            },
            400: {
                'description': 'Bad request',
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
        Process a video from a URL with OCR to extract text from frames.
        """
        if OCR_ENGINE_INSTANCE is None:
            return Response(
                {'error': 'OCR engine is not initialized. Check server logs.'}, 
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        # Validate request data
        if not request.data:
            return Response(
                {'error': 'No data provided. Expected JSON with video_url.'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        video_url = request.data.get('video_url')
        if not video_url:
            return Response(
                {'error': 'No video_url provided.'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        # Get interval parameter (default: 8 seconds)
        interval = request.data.get('interval', 8)
        try:
            interval = int(interval)
            if interval <= 0:
                return Response(
                    {'error': 'Interval must be a positive integer.'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
        except (ValueError, TypeError):
            return Response(
                {'error': 'Interval must be a valid integer.'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        # Generate a unique job ID
        job_id = str(uuid.uuid4())

        # Create directories for this job
        job_dir = os.path.join(settings.MEDIA_ROOT, 'video_jobs', job_id)
        frames_dir = os.path.join(job_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        # Start the video processing in a background thread
        import threading
        thread = threading.Thread(
            target=self.process_video_job,
            args=(job_id, video_url, interval, job_dir, frames_dir)
        )
        thread.daemon = True
        thread.start()

        return Response({
            'message': 'Video OCR processing started',
            'job_id': job_id,
            'status': 'processing'
        }, status=status.HTTP_202_ACCEPTED)

    def process_video_job(self, job_id, video_url, interval, job_dir, frames_dir):
        """
        Background process to handle video downloading, frame extraction, and OCR processing.
        """
        try:
            # Create a status file to track progress
            status_file = os.path.join(job_dir, 'status.json')
            with open(status_file, 'w') as f:
                json.dump({
                    'status': 'downloading',
                    'message': 'Downloading video from URL',
                    'progress': 0,
                    'job_id': job_id,
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f)

            # Download the video
            video_path = os.path.join(job_dir, f'video_{job_id}.mp4')
            try:
                self.download_video(video_url, video_path)

                # Verify the video file exists and has content
                if not os.path.exists(video_path):
                    raise ValueError(f"Video file was not created at {video_path}")

                if os.path.getsize(video_path) == 0:
                    raise ValueError(f"Downloaded video file is empty: {video_path}")

                # Log the file size for debugging
                file_size = os.path.getsize(video_path)
                print(f"Downloaded video file size: {file_size} bytes")

            except Exception as e:
                error_message = str(e)
                error_type = type(e).__name__

                # Provide more detailed error information
                if "Incomplete download" in error_message:
                    error_details = "The video download was incomplete. This could be due to network issues or the server disconnecting before the download finished."
                elif "not a valid video file" in error_message:
                    error_details = "The downloaded file is not a valid video file. The URL might not point to a video, or the video might be in an unsupported format."
                elif "Invalid URL format" in error_message:
                    error_details = "The provided URL is not valid. Please check the URL and try again."
                else:
                    error_details = "An unexpected error occurred during video download."

                with open(status_file, 'w') as f:
                    json.dump({
                        'status': 'failed',
                        'message': f'Failed to download video: {error_message}',
                        'error_type': error_type,
                        'error_details': error_details,
                        'job_id': job_id,
                        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
                    }, f)
                return

            # Update status
            with open(status_file, 'w') as f:
                json.dump({
                    'status': 'extracting',
                    'message': 'Extracting frames from video',
                    'progress': 25,
                    'job_id': job_id,
                    'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f)

            # Extract frames
            if not extract_frames(video_path, frames_dir, interval):
                # Check for common error patterns in the log files
                error_details = "Failed to extract frames from the video."

                # Try to determine the specific error
                if os.path.exists(video_path):
                    file_size = os.path.getsize(video_path)
                    if file_size == 0:
                        error_details = "The video file is empty (0 bytes)."
                    elif file_size < 1024:  # Less than 1KB
                        error_details = f"The video file is suspiciously small ({file_size} bytes). It might not be a valid video file."
                    else:
                        # Try to get more information about the file using file command if available
                        try:
                            file_info = subprocess.run(['file', video_path], capture_output=True, text=True, check=False)
                            if file_info.returncode == 0:
                                error_details = f"File information: {file_info.stdout.strip()}"
                        except (FileNotFoundError, subprocess.SubprocessError):
                            # file command not available or failed, try with ffprobe
                            try:
                                probe_command = [
                                    'ffprobe',
                                    '-v', 'error',
                                    '-show_format',
                                    '-show_streams',
                                    '-of', 'json',
                                    video_path
                                ]
                                probe_result = subprocess.run(probe_command, capture_output=True, text=True, check=False)
                                if probe_result.returncode == 0:
                                    try:
                                        video_info = json.loads(probe_result.stdout)
                                        if 'format' in video_info:
                                            format_info = video_info['format']
                                            error_details = f"The file appears to be a {format_info.get('format_name', 'unknown')} format, but frame extraction failed."
                                        else:
                                            error_details = "The file does not appear to be a valid video file (no format information)."
                                    except json.JSONDecodeError:
                                        error_details = "Could not parse video information from ffprobe."
                                else:
                                    error_details = f"ffprobe could not analyze the file: {probe_result.stderr}"
                            except (FileNotFoundError, subprocess.SubprocessError):
                                # Both file and ffprobe failed or are not available
                                pass
                else:
                    error_details = "The video file does not exist. It may have been deleted during processing."

                with open(status_file, 'w') as f:
                    json.dump({
                        'status': 'failed',
                        'message': 'Failed to extract frames from video',
                        'error_details': error_details,
                        'video_path': video_path,
                        'video_exists': os.path.exists(video_path),
                        'video_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0,
                        'job_id': job_id,
                        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
                    }, f)
                return

            # Update status
            with open(status_file, 'w') as f:
                json.dump({
                    'status': 'processing',
                    'message': 'Processing frames with OCR',
                    'progress': 50,
                    'job_id': job_id,
                    'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f)

            # Process frames with OCR
            results = []
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            total_frames = len(frame_files)

            for i, frame_file in enumerate(frame_files):
                frame_path = os.path.join(frames_dir, frame_file)
                frame_num, start_time, end_time = extract_frame_info(frame_file, interval)

                if frame_num is None:
                    continue

                # Process frame with OCR
                recognized_texts = process_frame_with_ocr(frame_path, OCR_ENGINE_INSTANCE)

                # Create and add result
                if recognized_texts:
                    frame_result = create_frame_result(frame_num, start_time, end_time, recognized_texts)
                    results.append(frame_result)

                # Update progress
                progress = 50 + int((i + 1) / total_frames * 50)
                with open(status_file, 'w') as f:
                    json.dump({
                        'status': 'processing',
                        'message': f'Processing frames with OCR ({i+1}/{total_frames})',
                        'progress': progress,
                        'job_id': job_id,
                        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
                    }, f)

            # Save results
            results_file = os.path.join(job_dir, 'results.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # Update status to completed
            with open(status_file, 'w') as f:
                json.dump({
                    'status': 'completed',
                    'message': 'OCR processing completed',
                    'progress': 100,
                    'job_id': job_id,
                    'results_file': results_file,
                    'total_frames': total_frames,
                    'processed_frames': len(results),
                    'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f)

        except Exception as e:
            import traceback
            error_message = f"Error during video processing: {str(e)}\n{traceback.format_exc()}"
            print(error_message)

            # Update status to failed
            with open(status_file, 'w') as f:
                json.dump({
                    'status': 'failed',
                    'message': f'An error occurred: {str(e)}',
                    'job_id': job_id,
                    'error': error_message,
                    'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f)

    def download_video(self, video_url, output_path):
        """
        Download a video from a URL to the specified path.
        """
        try:
            # Parse the URL to validate it
            parsed_url = urlparse(video_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid URL format")

            # Download the video in chunks
            response = requests.get(video_url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Get content type and check if it's a video
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith(('video/', 'application/octet-stream')):
                print(f"Warning: Content-Type '{content_type}' may not be a video. Proceeding anyway.")

            # Get content length if available
            content_length = response.headers.get('Content-Length')
            if content_length:
                expected_size = int(content_length)
            else:
                expected_size = None
                print("Warning: Content-Length header not available. Cannot verify complete download.")

            # Download the file
            downloaded_size = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

            # Verify download size if content length was provided
            if expected_size and downloaded_size < expected_size:
                raise ValueError(f"Incomplete download: Got {downloaded_size} bytes, expected {expected_size} bytes")

            # Validate the downloaded file
            if not self.validate_video_file(output_path):
                raise ValueError("Downloaded file is not a valid video file")

            return True
        except Exception as e:
            print(f"Error downloading video: {e}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    print(f"Removed invalid video file: {output_path}")
                except Exception as remove_error:
                    print(f"Failed to remove invalid video file: {remove_error}")
            raise

    def validate_video_file(self, video_path):
        """
        Validate that the file is a valid video file by checking its header.
        """
        try:
            # Check if file exists and has size > 0
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                print(f"Video file does not exist or is empty: {video_path}")
                return False

            # Use ffprobe to check if the file is a valid video
            command = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name',
                '-of', 'json',
                video_path
            ]

            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                # Parse the JSON output
                output = json.loads(result.stdout)
                # Check if there's a video stream
                if 'streams' in output and len(output['streams']) > 0:
                    print(f"Valid video file detected: {video_path}")
                    return True
                else:
                    print(f"No video streams found in file: {video_path}")
                    return False
            except subprocess.CalledProcessError as e:
                print(f"ffprobe error: {e.stderr}")
                return False
            except json.JSONDecodeError:
                print(f"Failed to parse ffprobe output")
                return False
        except Exception as e:
            print(f"Error validating video file: {e}")
            return False


class OCRVideoStatusAPIView(APIView):
    """
    API endpoint for checking the status of a video OCR job.
    """

    @extend_schema(
        description="Check the status of a video OCR job",
        parameters=[
            OpenApiParameter(
                name='job_id',
                description='ID of the OCR job to check',
                required=True,
                type=str,
                location=OpenApiParameter.PATH
            )
        ],
        responses={
            200: {
                'description': 'Job status',
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'message': {'type': 'string'},
                    'progress': {'type': 'integer'},
                    'job_id': {'type': 'string'},
                    'results': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'frame_num': {'type': 'integer'},
                                'chunk': {
                                    'type': 'object',
                                    'properties': {
                                        'startTime': {'type': 'string'},
                                        'endTime': {'type': 'string'}
                                    }
                                },
                                'texts': {
                                    'type': 'array',
                                    'items': {'type': 'string'}
                                }
                            }
                        }
                    }
                }
            },
            404: {
                'description': 'Job not found',
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
            }
        }
    )
    def get(self, request, job_id, *args, **kwargs):
        """
        Check the status of a video OCR job.
        """
        # Validate job_id
        try:
            uuid.UUID(job_id)  # Validate that job_id is a valid UUID
        except ValueError:
            return Response(
                {'error': 'Invalid job ID format.'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        # Check if job exists
        job_dir = os.path.join(settings.MEDIA_ROOT, 'video_jobs', job_id)
        status_file = os.path.join(job_dir, 'status.json')

        if not os.path.exists(status_file):
            return Response(
                {'error': f'Job with ID {job_id} not found.'}, 
                status=status.HTTP_404_NOT_FOUND
            )

        try:
            # Read status file
            with open(status_file, 'r') as f:
                job_status = json.load(f)

            # If job is completed, include results
            if job_status.get('status') == 'completed':
                results_file = os.path.join(job_dir, 'results.json')
                if os.path.exists(results_file):
                    with open(results_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    job_status['results'] = results

            return Response(job_status, status=status.HTTP_200_OK)

        except Exception as e:
            import traceback
            error_message = f"Error retrieving job status: {str(e)}\n{traceback.format_exc()}"
            print(error_message)
            return Response(
                {'error': f'An error occurred: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
