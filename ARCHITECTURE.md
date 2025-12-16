# VideoLingo Architecture for Video Course Platform

## Overview

VideoLingo интегрируется в платформу видеокурсов (Django 5 + Next.js) как сервис обработки видео на GPU сервере с on-demand биллингом через immers.cloud.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MAIN SERVER (24/7, no GPU)                             │
│                      Django 5 + Next.js + PostgreSQL                        │
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Partner    │  │   Admin     │  │  Next.js    │  │  Celery Beat        │ │
│  │  Cabinet    │  │   Panel     │  │  Frontend   │  │  (scheduler)        │ │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘  └──────────┬──────────┘ │
│         │                │                                      │            │
│         ▼                ▼                                      ▼            │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │                         DJANGO BACKEND                                   ││
│  │  • Course management                                                     ││
│  │  • File storage (local → S3/MinIO)                                       ││
│  │  • OpenStack API client (shelve/unshelve)                                ││
│  │  • Kinescope.io integration                                              ││
│  └──────────────────────────────────────────────────────────────────────────┘│
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │                      REDIS + CELERY                                      ││
│  │  • Task Queue (video_processing_queue)                                   ││
│  │  • Celery Orchestrator Worker                                            ││
│  │  • Status tracking                                                       ││
│  └──────────────────────────────────────┬───────────────────────────────────┘│
└─────────────────────────────────────────┼───────────────────────────────────┘
                                          │
                            HTTP/SFTP     │   OpenStack API
                            (files)       │   (shelve/unshelve)
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GPU SERVER (immers.cloud, on-demand)                   │
│                      RTX 4090, shelved when idle                            │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │                    VIDEOLINGO WORKER                                     ││
│  │                    ~/projects/VideoLingo                                 ││
│  │                                                                          ││
│  │  • Celery Worker (listens to Redis on main server)                       ││
│  │  • Downloads video → Processes → Uploads result                          ││
│  │  • WhisperX + Demucs + Ollama + HTTP→Chatterbox + FFmpeg                 ││
│  └──────────────────────────────────┬───────────────────────────────────────┘│
│                                     │ HTTP :8001                             │
│                                     ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │                    CHATTERBOX TTS SERVICE                                ││
│  │                    ~/projects/chatterbox-service (systemd)               ││
│  │                    Isolated venv - no dependency conflicts               ││
│  └──────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │                    OLLAMA (qwen2.5:14b)                                  ││
│  │                    systemd service                                       ││
│  └──────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Workflow

```
1. Partner uploads course
         │
         ▼
2. Files saved LOCALLY on main server
         │
         ▼
3. Admin receives notification → Reviews content
         │
         ▼ (review passed)
4. Admin clicks "Generate voiceover"
         │
         ▼
5. Django creates task in Redis Queue
         │
         ▼
6. Celery Orchestrator sees task:
   - Calls OpenStack API → unshelve GPU server
   - Waits for server to boot (~1-2 min)
         │
         ▼
7. GPU Worker receives task:
   - Downloads video from main server
   - WhisperX → Transcription
   - Ollama → Translation
   - Chatterbox → TTS (via HTTP to isolated service)
   - FFmpeg → Merge audio with video
   - Uploads result back
         │
         ▼
8. Main server receives result
         │
         ▼
9. (If queue empty for 15 min)
   Celery Orchestrator → shelve GPU server
         │
         ▼
10. Admin receives notification → Reviews voiceover
         │
         ▼ (if OK)
11. Django → Kinescope.io API → Upload
         │
         ▼
12. Course available for purchase
```

---

## Why Two Isolated Services on GPU Server

### The Problem

VideoLingo and Chatterbox have **incompatible dependencies**:

| Package | VideoLingo (whisperx) | Chatterbox | Conflict |
|---------|----------------------|------------|----------|
| tokenizers | <0.16 | >=0.20 | ❌ |
| transformers | ~4.39 | ==4.46.3 | ❌ |
| torch | 2.0.x | 2.6.x | ❌ |
| librosa | 0.10.x | 0.11.0 | ❌ |

### The Solution

Two isolated venvs communicating via HTTP:

```
~/projects/
├── VideoLingo/              # Service 1
│   ├── venv/                # whisperx, demucs, etc.
│   └── core/tts_backend/
│       └── chatterbox_http.py   # HTTP client
│
└── chatterbox-service/      # Service 2
    ├── venv/                # chatterbox-tts with correct deps
    └── main.py              # FastAPI server on :8001
```

---

## GPU Server Directory Structure

```
/home/ubuntu/
├── projects/
│   ├── VideoLingo/              # Main VideoLingo
│   │   ├── venv/
│   │   ├── config.yaml
│   │   ├── worker.py            # Celery worker (future)
│   │   ├── install.sh           # Installation script
│   │   └── core/
│   │       └── tts_backend/
│   │           ├── chatterbox_http.py   # HTTP client
│   │           └── tts_main.py          # TTS dispatcher
│   │
│   └── chatterbox-service/      # Chatterbox HTTP Service
│       ├── venv/
│       ├── main.py              # FastAPI app
│       ├── requirements.txt
│       └── start.sh
│
└── data/                        # Shared data (if needed)
    ├── uploads/
    ├── processing/
    └── outputs/
```

---

## Implementation Phases

### Phase 1: Chatterbox Isolation (Current)

1. Create `~/projects/chatterbox-service/`
2. Install chatterbox-tts in isolated venv
3. Write FastAPI server (`main.py`)
4. Configure systemd for auto-start
5. Create HTTP client in VideoLingo
6. Update VideoLingo to remove chatterbox-tts from its venv

### Phase 2: VideoLingo Worker

1. Create `worker.py` - Celery worker
2. Create `tasks.py` - Task definitions
3. Add file transfer logic (download/upload)
4. Add status reporting to Redis

### Phase 3: Django Integration

1. Create `ProcessingJob` model
2. Create Celery tasks for orchestration
3. Implement OpenStack API client (shelve/unshelve)
4. Add admin UI for job management
5. Integrate Kinescope.io API

### Phase 4: Production Hardening

1. Error handling and retries
2. Monitoring and alerting
3. Log aggregation
4. Auto-scaling (if needed)

---

## Chatterbox HTTP Service API

### Endpoints

```
GET  /health
     → {"status": "healthy", "model_loaded": true, "device": "cuda", "vram_used_gb": 3.2}

POST /api/tts/generate
     Content-Type: application/json
     Body: {"text": "...", "language": "en", "exaggeration": 0.5, "cfg_weight": 0.4}
     → {"audio_base64": "...", "sample_rate": 24000, "duration_seconds": 2.5}

POST /api/tts/generate-file
     Content-Type: multipart/form-data
     Fields: text, language, exaggeration, cfg_weight, reference_audio (optional file)
     → audio/wav file
```

### systemd Service

```ini
# /etc/systemd/system/chatterbox-tts.service
[Unit]
Description=Chatterbox TTS Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/projects/chatterbox-service
ExecStart=/home/ubuntu/projects/chatterbox-service/venv/bin/python main.py
Restart=always
RestartSec=5
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
```

---

## Configuration

### VideoLingo config.yaml

```yaml
# TTS method - use HTTP client for isolated Chatterbox
tts_method: 'chatterbox_http'

# Chatterbox settings (passed to HTTP service)
chatterbox_tts:
  service_url: 'http://localhost:8001'
  voice_clone_mode: 2        # 1=no clone, 2=single ref, 3=per-segment
  exaggeration: 0.5          # Emotionality (0.0-1.0)
  cfg_weight: 0.4            # Cloning strength (0.3-0.5)

# Languages
source_language: 'Russian'
target_language: 'English'
```

---

## GPU Server Quick Setup

```bash
# 1. Clone VideoLingo
cd ~/projects
git clone git@github.com:carden-code/VideoLingo.git
cd VideoLingo

# 2. Install VideoLingo (without chatterbox in same venv)
python3 -m venv venv
source venv/bin/activate
./install.sh

# 3. Setup Chatterbox service
cd ~/projects
mkdir chatterbox-service && cd chatterbox-service
python3 -m venv venv
source venv/bin/activate
pip install chatterbox-tts fastapi uvicorn python-multipart soundfile
# Copy main.py from tz.md or create it

# 4. Start Chatterbox service
python main.py  # or use systemd

# 5. Test
curl http://localhost:8001/health

# 6. Start VideoLingo
cd ~/projects/VideoLingo
source venv/bin/activate
streamlit run st.py
```

---

## Expected Performance (RTX 4090)

| Metric | Value |
|--------|-------|
| Chatterbox model load | ~3-4 GB VRAM |
| Generation speed | ~80-100 it/s |
| 10 sec audio generation | ~2-3 sec |
| 10 min video full cycle | ~10-15 min |

---

## Files to Create

| File | Description |
|------|-------------|
| `~/projects/chatterbox-service/main.py` | FastAPI service |
| `~/projects/chatterbox-service/requirements.txt` | Dependencies |
| `~/projects/chatterbox-service/start.sh` | Startup script |
| `~/projects/VideoLingo/core/tts_backend/chatterbox_http.py` | HTTP client |
| `/etc/systemd/system/chatterbox-tts.service` | Systemd service |

---

## Related Docs

- [tz.md](./tz.md) - Original technical specification
- [immers.cloud API](https://immers.cloud/api) - OpenStack Shelve/Unshelve
- [Kinescope.io](https://kinescope.io) - Video hosting platform
