#!/usr/bin/env python3
"""
Video OCR Analysis Backend Server
Podporuje stahování videí, extrakci snímků a OCR analýzu
"""

import os
import uuid
import json
import time
import shutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import yt_dlp
import cv2
import ffmpeg
from paddleocr import PaddleOCR
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
import logging

# Konfigurace
app = Flask(__name__)
CORS(app)  # Povolí CORS pro frontend

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globální konfigurace
CONFIG = {
    'UPLOAD_FOLDER': 'temp_videos',
    'FRAMES_FOLDER': 'temp_frames',
    'MAX_VIDEO_SIZE': 500 * 1024 * 1024,  # 500MB
    'SUPPORTED_FORMATS': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'],
    'CLEANUP_AFTER_HOURS': 24,
    'MAX_CONCURRENT_JOBS': 3
}

# Globální proměnné
jobs_status = {}  # {job_id: status_dict}
ocr_engines = {}  # {lang: PaddleOCR_instance}
active_jobs = 0


def ensure_directories():
    """Vytvoří potřebné adresáře"""
    Path(CONFIG['UPLOAD_FOLDER']).mkdir(exist_ok=True)
    Path(CONFIG['FRAMES_FOLDER']).mkdir(exist_ok=True)


def get_ocr_engine(lang: str) -> PaddleOCR:
    """Vrátí OCR engine pro daný jazyk (singleton pattern)"""
    if lang not in ocr_engines:
        try:
            if lang == 'cs+en':
                ocr_engines[lang] = PaddleOCR(use_angle_cls=True, lang='cs', use_gpu=False)
            elif lang == 'cs':
                ocr_engines[lang] = PaddleOCR(use_angle_cls=True, lang='cs',
                                              use_gpu=False)  # PaddleOCR podporuje češtinu
            else:
                ocr_engines[lang] = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
            logger.info(f"Inicializován OCR engine pro jazyk: {lang}")
        except Exception as e:
            logger.error(f"Chyba při inicializaci OCR: {e}")
            raise
    return ocr_engines[lang]


def download_video(url: str, output_path: str) -> Dict[str, Any]:
    """Stáhne video z URL pomocí yt-dlp"""
    try:
        ydl_opts = {
            'outtmpl': output_path,
            'format': 'best[height<=720]',  # Omezí kvalitu pro rychlejší zpracování
            'noplaylist': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Získání informací o videu
            info = ydl.extract_info(url, download=False)
            duration = info.get('duration', 0)
            title = info.get('title', 'Unknown')

            # Kontrola velikosti
            filesize = info.get('filesize') or info.get('filesize_approx', 0)
            if filesize > CONFIG['MAX_VIDEO_SIZE']:
                raise ValueError(f"Video je příliš velké: {filesize / 1024 / 1024:.1f}MB")

            # Stažení
            ydl.download([url])

            return {
                'duration': duration,
                'title': title,
                'filesize': filesize
            }
    except Exception as e:
        logger.error(f"Chyba při stahování videa: {e}")
        raise


def extract_frames(video_path: str, output_dir: str, interval: int) -> List[str]:
    """Extrahuje snímky z videa pomocí ffmpeg"""
    try:
        # Vytvoření výstupního adresáře
        Path(output_dir).mkdir(exist_ok=True)

        # FFmpeg příkaz pro extrakci snímků
        output_pattern = os.path.join(output_dir, 'frame_%04d.jpg')

        (
            ffmpeg
            .input(video_path)
            .filter('fps', f'1/{interval}')
            .output(output_pattern, **{'q:v': 2})
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        # Získání seznamu vytvořených snímků
        frame_files = sorted([
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.startswith('frame_') and f.endswith('.jpg')
        ])

        logger.info(f"Extrahovano {len(frame_files)} snímků")
        return frame_files

    except ffmpeg.Error as e:
        logger.error(f"FFmpeg chyba: {e.stderr.decode()}")
        raise
    except Exception as e:
        logger.error(f"Chyba při extrakci snímků: {e}")
        raise


def analyze_frame_ocr(frame_path: str, ocr_engine: PaddleOCR) -> Dict[str, Any]:
    """Analyzuje jeden snímek pomocí OCR"""
    try:
        # OCR analýza
        result = ocr_engine.ocr(frame_path, cls=True)

        # Zpracování výsledků
        texts = []
        confidences = []

        # Debug výpis pro kontrolu struktury výsledků
        logger.debug(f"OCR result structure: {type(result)}")
        if result:
            logger.debug(f"First result item: {type(result[0]) if result else 'None'}")

        if result and result[0]:
            for line in result[0]:
                if len(line) >= 2:
                    # Zajištění správného zpracování textu s diakritikou
                    if isinstance(line[1], (list, tuple)):
                        text = line[1][0]  # Zachování původního formátu textu
                        confidence = line[1][1] if len(line[1]) > 1 else 0.0
                    else:
                        text = str(line[1])  # Explicitní konverze na string
                        confidence = 0.0

                    # Debug výpis pro kontrolu textu
                    logger.debug(f"Detected text: {text}")

                    texts.append(text)
                    confidences.append(confidence)

        combined_text = '\n'.join(texts) if texts else ''
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Debug výpis pro kontrolu výsledného textu
        logger.debug(f"Combined text: {combined_text}")

        return {
            'text': combined_text,
            'confidence': avg_confidence,
            'word_count': len(texts),
            'raw_texts': texts  # Přidání seznamu všech rozpoznaných textů
        }

    except Exception as e:
        logger.error(f"Chyba při OCR analýze snímku {frame_path}: {e}")
        return {
            'text': '',
            'confidence': 0.0,
            'word_count': 0,
            'error': str(e)
        }


def process_video_job(job_id: str, video_url: str, settings: Dict[str, Any]):
    """Hlavní funkce pro zpracování videa (běží v samostatném vlákně)"""
    global active_jobs

    try:
        active_jobs += 1
        jobs_status[job_id]['status'] = 'downloading'
        jobs_status[job_id]['progress'] = 10

        # Příprava cest
        video_filename = f"{job_id}.%(ext)s"
        video_dir = Path(CONFIG['UPLOAD_FOLDER']) / job_id
        frames_dir = Path(CONFIG['FRAMES_FOLDER']) / job_id
        video_dir.mkdir(exist_ok=True)
        frames_dir.mkdir(exist_ok=True)

        video_path_template = str(video_dir / video_filename)

        # 1. Stažení videa
        logger.info(f"Job {job_id}: Stahování videa z {video_url}")
        video_info = download_video(video_url, video_path_template)

        # Najít skutečný název souboru
        video_files = list(video_dir.glob("*"))
        if not video_files:
            raise FileNotFoundError("Video se nepodařilo stáhnout")
        video_path = str(video_files[0])

        jobs_status[job_id].update({
            'status': 'extracting_frames',
            'progress': 30,
            'video_info': video_info
        })

        # 2. Extrakce snímků
        logger.info(f"Job {job_id}: Extrakce snímků")
        frame_files = extract_frames(
            video_path,
            str(frames_dir),
            settings['frame_interval']
        )

        jobs_status[job_id].update({
            'status': 'ocr_analysis',
            'progress': 50,
            'total_frames': len(frame_files)
        })

        # 3. OCR analýza
        logger.info(f"Job {job_id}: OCR analýza {len(frame_files)} snímků")
        ocr_engine = get_ocr_engine(settings['language'])

        frame_results = []
        for i, frame_path in enumerate(frame_files):
            # Výpočet času snímku
            frame_time_seconds = i * settings['frame_interval']
            frame_time_formatted = f"{frame_time_seconds // 60}:{frame_time_seconds % 60:02d}"

            # OCR analýza
            ocr_result = analyze_frame_ocr(frame_path, ocr_engine)

            frame_results.append({
                'frame_number': i + 1,
                'time_seconds': frame_time_seconds,
                'time_formatted': frame_time_formatted,
                'text': ocr_result['text'],
                'confidence': ocr_result['confidence'],
                'word_count': ocr_result['word_count'],
                'raw_texts': ocr_result.get('raw_texts', [])  # Přidání seznamu všech rozpoznaných textů
            })

            # Aktualizace pokroku
            progress = 50 + int((i / len(frame_files)) * 45)
            jobs_status[job_id]['progress'] = progress
            jobs_status[job_id]['processed_frames'] = i + 1

        # Dokončení úlohy
        all_text = '\n\n'.join([
            f"[{result['time_formatted']}] {result['text']}"
            for result in frame_results
            if result['text'].strip()
        ])

        # Vytvoření alternativního textu s použitím raw_texts pro lepší zachování diakritiky
        all_raw_text = '\n\n'.join([
            f"[{result['time_formatted']}] {' | '.join(result.get('raw_texts', []))}"
            for result in frame_results
            if result.get('raw_texts')
        ])

        frames_with_text = sum(1 for r in frame_results if r['text'].strip())
        avg_confidence = sum(r['confidence'] for r in frame_results) / len(frame_results) if frame_results else 0

        jobs_status[job_id].update({
            'status': 'completed',
            'progress': 100,
            'results': {
                'frame_results': frame_results,
                'summary': {
                    'total_frames': len(frame_results),
                    'frames_with_text': frames_with_text,
                    'success_rate': frames_with_text / len(frame_results) if frame_results else 0,
                    'average_confidence': avg_confidence,
                    'all_text': all_text,
                    'all_raw_text': all_raw_text  # Přidání alternativního textu s lepší diakritikou
                }
            },
            'completed_at': datetime.now().isoformat()
        })

        logger.info(f"Job {job_id}: Dokončeno úspěšně")

    except Exception as e:
        logger.error(f"Job {job_id}: Chyba - {e}")
        jobs_status[job_id].update({
            'status': 'failed',
            'error': str(e),
            'failed_at': datetime.now().isoformat()
        })
    finally:
        active_jobs -= 1
        # Naplánování vyčištění souborů
        cleanup_timer = threading.Timer(
            CONFIG['CLEANUP_AFTER_HOURS'] * 3600,
            cleanup_job_files,
            args=[job_id]
        )
        cleanup_timer.start()


def cleanup_job_files(job_id: str):
    """Vyčistí soubory související s úlohou"""
    try:
        video_dir = Path(CONFIG['UPLOAD_FOLDER']) / job_id
        frames_dir = Path(CONFIG['FRAMES_FOLDER']) / job_id

        if video_dir.exists():
            shutil.rmtree(video_dir)
        if frames_dir.exists():
            shutil.rmtree(frames_dir)

        # Odstranění z jobs_status po delší době
        if job_id in jobs_status:
            del jobs_status[job_id]

        logger.info(f"Vyčištěny soubory pro job {job_id}")

    except Exception as e:
        logger.error(f"Chyba při čištění souborů pro job {job_id}: {e}")


# API Endpointy

@app.route('/api/health', methods=['GET'])
def health_check():
    """Kontrola stavu serveru"""
    return jsonify({
        'status': 'healthy',
        'active_jobs': active_jobs,
        'max_concurrent_jobs': CONFIG['MAX_CONCURRENT_JOBS'],
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/process-video', methods=['POST'])
def process_video():
    """Spustí zpracování videa"""
    try:
        # Kontrola limitu současných úloh
        if active_jobs >= CONFIG['MAX_CONCURRENT_JOBS']:
            return jsonify({
                'error': 'Server je přetížen, zkuste to později'
            }), 429

        # Validace vstupu
        data = request.get_json()
        if not data or 'video_url' not in data:
            raise BadRequest('Chybí video_url')

        video_url = data['video_url'].strip()
        if not video_url:
            raise BadRequest('Prázdná video_url')

        # Nastavení s výchozími hodnotami
        settings = {
            'frame_interval': data.get('frame_interval', 8),
            'ocr_quality': data.get('ocr_quality', 'medium'),
            'language': data.get('language', 'cs+en')
        }

        print(f"language: {settings['language']}")
        # Validace nastavení
        if not (1 <= settings['frame_interval'] <= 60):
            raise BadRequest('frame_interval musí být mezi 1-60 sekundami')

        if settings['ocr_quality'] not in ['high', 'medium', 'fast']:
            raise BadRequest('Neplatná hodnota ocr_quality')

        if settings['language'] not in ['cs', 'en', 'cs+en']:
            raise BadRequest('Nepodporovaný jazyk')

        # Vytvoření nové úlohy
        job_id = str(uuid.uuid4())
        jobs_status[job_id] = {
            'job_id': job_id,
            'status': 'queued',
            'progress': 0,
            'video_url': video_url,
            'settings': settings,
            'started_at': datetime.now().isoformat()
        }

        # Spuštění zpracování v novém vlákně
        thread = threading.Thread(
            target=process_video_job,
            args=(job_id, video_url, settings)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Zpracování zahájeno'
        })

    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Chyba v process_video: {e}")
        return jsonify({'error': 'Interní chyba serveru'}), 500


@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    """Vrátí stav zpracování úlohy"""
    if job_id not in jobs_status:
        return jsonify({'error': 'Úloha nenalezena'}), 404

    status = jobs_status[job_id].copy()
    # Neposíláme výsledky v status endpointu (mohou být velké)
    if 'results' in status:
        status['has_results'] = True
        del status['results']

    return jsonify(status)


@app.route('/api/results/<job_id>', methods=['GET'])
def get_job_results(job_id: str):
    """Vrátí výsledky zpracování úlohy"""
    if job_id not in jobs_status:
        return jsonify({'error': 'Úloha nenalezena'}), 404

    job = jobs_status[job_id]

    if job['status'] != 'completed':
        return jsonify({
            'error': 'Úloha ještě není dokončena',
            'current_status': job['status']
        }), 400

    return jsonify({
        'job_id': job_id,
        'results': job.get('results', {}),
        'completed_at': job.get('completed_at')
    })


@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """Vrátí seznam všech úloh"""
    jobs_list = []
    for job_id, job_data in jobs_status.items():
        job_summary = {
            'job_id': job_id,
            'status': job_data['status'],
            'progress': job_data.get('progress', 0),
            'started_at': job_data.get('started_at'),
            'video_url': job_data.get('video_url', '').split('/')[-1]  # Pouze název souboru
        }
        if job_data['status'] == 'completed':
            job_summary['completed_at'] = job_data.get('completed_at')
        elif job_data['status'] == 'failed':
            job_summary['failed_at'] = job_data.get('failed_at')
            job_summary['error'] = job_data.get('error')

        jobs_list.append(job_summary)

    return jsonify({
        'jobs': sorted(jobs_list, key=lambda x: x['started_at'], reverse=True),
        'total': len(jobs_list),
        'active': active_jobs
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint nenalezen'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Interní chyba serveru'}), 500


if __name__ == '__main__':
    # Inicializace
    ensure_directories()

    # Kontrola závislostí
    try:
        import yt_dlp
        import cv2
        import ffmpeg
        from paddleocr import PaddleOCR

        print("✅ Všechny závislosti jsou k dispozici")
    except ImportError as e:
        print(f"❌ Chybí závislost: {e}")
        print("Nainstalujte pomocí: pip install -r requirements.txt")
        exit(1)

    print("🚀 Spouštím Video OCR Backend Server...")
    print(f"📁 Dočasné soubory: {CONFIG['UPLOAD_FOLDER']}")
    print(f"🖼️ Snímky: {CONFIG['FRAMES_FOLDER']}")
    print(f"👥 Max současných úloh: {CONFIG['MAX_CONCURRENT_JOBS']}")

    # Spuštění Flask serveru
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
