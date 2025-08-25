# ==============================================================================
# AI-Powered Photo Gallery FastAPI Application (Full Implementation)
#
# Features:
# - Secure API-driven media access.
# - Efficient, paginated "lazy loading" of media.
# - Bandwidth-saving two-stage upload with pre-flight hash checks.
# - AI face detection, age/gender estimation (InsightFace).
# - AI image captioning for text search (Salesforce/BLIP).
# - Advanced search endpoint combining text, person, and demographic filters.
# - Automatic creation of folder structure and fresh database setup.
# ==============================================================================
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# --- FastAPI & Pydantic ---
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --- Database ---
import aiosqlite

# --- Image & Video Processing ---
import cv2
from PIL import Image

# --- AI & Clustering ---
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# --- AI Captioning (New) ---
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

import uvicorn

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================

# --- File System ---
BASE_DIR = Path(__file__).parent
DATABASE_URL = BASE_DIR / "media_database.db"
AI_DATABASE_URL = BASE_DIR / "ai_results.db"
STAGING_DIR = BASE_DIR / "staging"
OUTPUT_DIR = BASE_DIR / "output_media"
ORIGINALS_SUBDIR = "originals"
THUMBNAILS_SUBDIR = "thumbnails"

# --- Concurrency & Performance ---
CPU_WORKERS = max(1, os.cpu_count() // 2)
AI_WORKERS = max(1, os.cpu_count() // 4)
MAX_CONCURRENT_UPLOAD_TASKS = CPU_WORKERS
AI_MAX_CONCURRENT_TASKS = AI_WORKERS

# --- Media Processing ---
THUMBNAIL_SIZE = (256, 256)
VIDEO_THUMBNAIL_TIME_SECONDS = 2.0
VIDEO_FRAME_INTERVAL_SECONDS = 5.0
VIDEO_FACE_SIMILARITY_THRESHOLD = 0.85

# --- AI Clustering ---
CLUSTER_EPSILON = 0.5
MIN_CLUSTER_SAMPLES = 2
CLUSTER_INTERVAL_SECONDS = 3600

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Enums for Statuses ---
class TaskStatusEnum(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class AIStatusEnum(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ==============================================================================
# --- 2. Pydantic Models (API Schemas) ---
# ==============================================================================

class TaskStatus(BaseModel):
    id: str
    original_filename: str
    status: TaskStatusEnum
    ai_status: Optional[AIStatusEnum] = None
    error_message: Optional[str] = None

class MediaItem(BaseModel):
    id: str
    original_filename: str
    media_url: str
    thumbnail_url: str
    face_count: int
    processed_at: datetime
    caption: Optional[str] = None

class Person(BaseModel):
    person_id: str
    name: Optional[str] = "Unknown Person"
    face_count: int
    cover_thumbnail_url: str

class HashCheckRequest(BaseModel):
    hashes: List[str]

class HashCheckResponse(BaseModel):
    needed_hashes: List[str]

# ==============================================================================
# --- 3. DATABASE SETUP & HELPERS ---
# ==============================================================================

@asynccontextmanager
async def get_media_db():
    db = await aiosqlite.connect(DATABASE_URL, timeout=30.0)
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()

@asynccontextmanager
async def get_ai_db():
    db = await aiosqlite.connect(AI_DATABASE_URL, timeout=30.0)
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()

async def init_databases():
    logger.info("Initializing databases...")
    # --- Media Processing Database ---
    async with get_media_db() as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS media_files (
                id TEXT PRIMARY KEY, original_filename TEXT NOT NULL, staging_path TEXT,
                status TEXT NOT NULL, ai_status TEXT NOT NULL DEFAULT 'PENDING',
                file_type TEXT NOT NULL, file_hash TEXT NOT NULL UNIQUE,
                relative_original_path TEXT, relative_thumbnail_path TEXT,
                error_message TEXT, created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
        await db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_file_hash ON media_files (file_hash)")
        await db.commit()

    # --- AI Results Database ---
    async with get_ai_db() as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS processed_files (
                id TEXT PRIMARY KEY, original_filename TEXT NOT NULL, relative_path TEXT NOT NULL,
                relative_thumbnail_path TEXT, face_count INTEGER NOT NULL,
                caption TEXT, -- New column for BLIP captions
                processed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id TEXT PRIMARY KEY, file_id TEXT NOT NULL, age INTEGER, gender TEXT,
                embedding BLOB NOT NULL, bbox TEXT, cluster_id INTEGER,
                FOREIGN KEY (file_id) REFERENCES processed_files(id) ON DELETE CASCADE,
                FOREIGN KEY (cluster_id) REFERENCES face_clusters(cluster_id)
            )""")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS face_clusters(
                cluster_id INTEGER PRIMARY KEY AUTOINCREMENT, person_id TEXT NOT NULL UNIQUE,
                name TEXT, notes TEXT
            )""")
        # --- Full-Text Search (FTS5) Virtual Table for Captions ---
        await db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS processed_files_fts USING fts5(
                id UNINDEXED, caption, content='processed_files', content_rowid='rowid'
            )""")
        # Trigger to keep FTS table in sync with processed_files
        await db.execute("""
            CREATE TRIGGER IF NOT EXISTS processed_files_after_insert
            AFTER INSERT ON processed_files BEGIN
                INSERT INTO processed_files_fts(rowid, id, caption)
                VALUES (new.rowid, new.id, new.caption);
            END;
        """)

        await db.execute(
            "INSERT OR IGNORE INTO face_clusters (cluster_id, person_id, name) VALUES (0, ?, ?)",
            (str(uuid.uuid5(uuid.NAMESPACE_DNS, 'noise.cluster')), 'Noise')
        )
        await db.commit()
    logger.info("Databases initialized successfully.")


# ==============================================================================
# --- 4. CORE PROCESSING LOGIC (CPU-Bound & AI) ---
# ==============================================================================

def calculate_hash(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192): h.update(chunk)
    return h.hexdigest()

def process_image_cpu(contents: bytes, original_path: Path, thumb_path: Path) -> None:
    image = Image.open(BytesIO(contents)).convert("RGB")
    image.save(original_path, format="JPEG", quality=95)
    image.thumbnail(THUMBNAIL_SIZE)
    image.save(thumb_path, format="JPEG", quality=85)

def process_video_thumb_cpu(original_path: Path, thumb_path: Path) -> bool:
    cap = cv2.VideoCapture(str(original_path))
    if not cap.isOpened(): return False
    cap.set(cv2.CAP_PROP_POS_MSEC, VIDEO_THUMBNAIL_TIME_SECONDS * 1000)
    success, frame = cap.read()
    if success: cv2.imwrite(str(thumb_path), frame)
    cap.release()
    return success

def analyze_image_faces(model: FaceAnalysis, file_path: Path) -> list[dict]:
    img = cv2.imread(str(file_path))
    if img is None: return [{"error": f"Could not read image: {file_path}"}]
    faces = model.get(img)
    return [{"age": f.age, "gender": "Male" if f.gender else "Female",
             "embedding": f.embedding.tobytes(), "bbox": json.dumps(f.bbox.astype(int).tolist())} for f in faces]

def analyze_video_faces(model: FaceAnalysis, file_path: Path) -> list[dict]:
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened(): return [{"error": f"Could not open video: {file_path}"}]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps * VIDEO_FRAME_INTERVAL_SECONDS)
    unique_faces, embeddings = [], []
    frame_count = 0
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        success, frame = cap.read()
        if not success: break
        for face in model.get(frame):
            sim = cosine_similarity(face.embedding.reshape(1, -1), np.array(embeddings)) if embeddings else [[0]]
            if not embeddings or np.max(sim) < VIDEO_FACE_SIMILARITY_THRESHOLD:
                embeddings.append(face.embedding)
                unique_faces.append({"age": face.age, "gender": "Male" if face.gender else "Female",
                                     "embedding": face.embedding.tobytes(), "bbox": json.dumps(face.bbox.astype(int).tolist())})
        frame_count += frame_interval
    cap.release()
    return unique_faces

def generate_caption(processor, model, file_path: Path) -> str:
    try:
        raw_image = Image.open(file_path).convert('RGB')
        inputs = processor(images=raw_image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)
        return processor.decode(out[0], skip_special_tokens=True).strip()
    except Exception as e:
        logger.error(f"Failed to generate caption for {file_path.name}: {e}")
        return ""


# ==============================================================================
# --- 5. BACKGROUND WORKER TASKS ---
# ==============================================================================

async def process_upload_queue(app: FastAPI):
    # This function remains largely the same, but with corrected transaction handling
    logger.info("Upload processor worker started.")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOAD_TASKS)
    loop = asyncio.get_running_loop()
    while True:
        async with get_media_db() as db:
            cursor = await db.execute("SELECT id FROM media_files WHERE status = ? LIMIT ?",
                                      (TaskStatusEnum.PENDING.value, MAX_CONCURRENT_UPLOAD_TASKS))
            task_ids = [row['id'] for row in await cursor.fetchall()]
        if not task_ids:
            await asyncio.sleep(5)
            continue

        async def process_task(task_id: str):
            async with semaphore:
                async with get_media_db() as db:
                    try:
                        await db.execute("UPDATE media_files SET status = ? WHERE id = ?",
                                         (TaskStatusEnum.PROCESSING.value, task_id))
                        await db.commit()
                        cursor = await db.execute("SELECT * FROM media_files WHERE id = ?", (task_id,))
                        task = await cursor.fetchone()
                        if not task: return

                        staging_path = Path(task["staging_path"])
                        date_folder = datetime.now().strftime("%Y-%m-%d")
                        originals_dir = OUTPUT_DIR / date_folder / ORIGINALS_SUBDIR
                        thumbnails_dir = OUTPUT_DIR / date_folder / THUMBNAILS_SUBDIR
                        originals_dir.mkdir(parents=True, exist_ok=True)
                        thumbnails_dir.mkdir(parents=True, exist_ok=True)

                        p = Path(task["original_filename"])
                        new_base = f"{p.stem}_{task_id[:8]}"
                        if task["file_type"] == 'image':
                            orig_path = originals_dir / f"{new_base}.jpg"
                            thumb_path = thumbnails_dir / f"{new_base}.jpg"
                            await loop.run_in_executor(app.state.process_pool, process_image_cpu,
                                                       staging_path.read_bytes(), orig_path, thumb_path)
                        else:
                            orig_path = originals_dir / f"{new_base}{p.suffix}"
                            thumb_path = thumbnails_dir / f"{new_base}.jpg"
                            shutil.move(str(staging_path), str(orig_path))
                            await loop.run_in_executor(app.state.process_pool, process_video_thumb_cpu,
                                                       orig_path, thumb_path)

                        staging_path.unlink(missing_ok=True)
                        await db.execute("UPDATE media_files SET status = ?, relative_original_path = ?, "
                                         "relative_thumbnail_path = ?, staging_path = NULL WHERE id = ?",
                                         (TaskStatusEnum.COMPLETED.value, str(orig_path.relative_to(OUTPUT_DIR)),
                                          str(thumb_path.relative_to(OUTPUT_DIR)), task_id))
                        await db.commit()
                        logger.info(f"Successfully processed upload {task_id}")
                    except Exception as e:
                        logger.error(f"Failed to process upload {task_id}: {e}", exc_info=False)
                        await db.execute("UPDATE media_files SET status = ?, error_message = ? WHERE id = ?",
                                         (TaskStatusEnum.FAILED.value, str(e), task_id))
                        await db.commit()
        await asyncio.gather(*(process_task(tid) for tid in task_ids))

async def process_ai_queue(app: FastAPI):
    logger.info("AI processor worker started.")
    semaphore = asyncio.Semaphore(AI_MAX_CONCURRENT_TASKS)
    loop = asyncio.get_running_loop()
    while True:
        async with get_media_db() as media_db:
            cursor = await media_db.execute("SELECT * FROM media_files WHERE status = ? AND ai_status = ? LIMIT ?",
                                           (TaskStatusEnum.COMPLETED.value, AIStatusEnum.PENDING.value, AI_MAX_CONCURRENT_TASKS))
            tasks = await cursor.fetchall()
        if not tasks:
            await asyncio.sleep(10)
            continue

        async def process_task(task_data):
            async with semaphore:
                task_id = task_data['id']
                async with get_media_db() as media_db, get_ai_db() as ai_db:
                    try:
                        await media_db.execute("UPDATE media_files SET ai_status = ? WHERE id = ?",
                                               (AIStatusEnum.PROCESSING.value, task_id))
                        await media_db.commit()

                        full_path = OUTPUT_DIR / task_data["relative_original_path"]
                        caption = ""
                        if task_data['file_type'] == 'image':
                            face_func = partial(analyze_image_faces, app.state.face_app, full_path)
                            caption = await loop.run_in_executor(app.state.ai_executor, generate_caption,
                                                                 app.state.blip_processor, app.state.blip_model, full_path)
                        else: # video
                            face_func = partial(analyze_video_faces, app.state.face_app, full_path)

                        face_results = await loop.run_in_executor(app.state.ai_executor, face_func)
                        if any('error' in r for r in face_results): raise RuntimeError(face_results[0]['error'])

                        await ai_db.execute(
                            "INSERT INTO processed_files (id, original_filename, relative_path, relative_thumbnail_path, face_count, caption) VALUES (?, ?, ?, ?, ?, ?)",
                            (task_id, task_data['original_filename'], task_data['relative_original_path'],
                             task_data['relative_thumbnail_path'], len(face_results), caption))
                        for face in face_results:
                            await ai_db.execute("INSERT INTO face_embeddings (id, file_id, age, gender, embedding, bbox) VALUES (?, ?, ?, ?, ?, ?)",
                                (str(uuid.uuid4()), task_id, face['age'], face['gender'], face['embedding'], face['bbox']))
                        await ai_db.commit()

                        await media_db.execute("UPDATE media_files SET ai_status = ? WHERE id = ?",
                                               (AIStatusEnum.COMPLETED.value, task_id))
                        await media_db.commit()
                        logger.info(f"AI processing completed for {task_id}")
                    except Exception as e:
                        logger.error(f"Failed AI processing for {task_id}: {e}", exc_info=False)
                        async with get_media_db() as db_err:
                            await db_err.execute("UPDATE media_files SET ai_status = ?, error_message = ? WHERE id = ?",
                                                 (AIStatusEnum.FAILED.value, str(e), task_id))
                            await db_err.commit()
        await asyncio.gather(*(process_task(task) for task in tasks))

async def run_face_clustering():
    # This function remains largely the same, but with corrected transaction handling
    logger.info("Face clustering worker started.")
    while True:
        try:
            async with get_ai_db() as db:
                cursor = await db.execute("SELECT COUNT(id) FROM face_embeddings WHERE cluster_id IS NULL")
                count = (await cursor.fetchone())[0]
                if count < MIN_CLUSTER_SAMPLES:
                    logger.info(f"Not enough new faces to cluster ({count}/{MIN_CLUSTER_SAMPLES}). Waiting.")
                else:
                    logger.info(f"Clustering {count} new faces...")
                    cursor = await db.execute("SELECT id, embedding FROM face_embeddings WHERE cluster_id IS NULL")
                    rows = await cursor.fetchall()
                    face_ids = [row['id'] for row in rows]
                    embeddings = np.array([np.frombuffer(row['embedding'], dtype=np.float32) for row in rows])
                    dbscan = DBSCAN(eps=CLUSTER_EPSILON, min_samples=MIN_CLUSTER_SAMPLES, metric='cosine', n_jobs=-1)
                    labels = dbscan.fit_predict(embeddings)

                    for label in set(labels):
                        if label == -1: continue
                        person_uuid = str(uuid.uuid4())
                        cursor = await db.execute("INSERT INTO face_clusters (person_id) VALUES (?)", (person_uuid,))
                        new_cluster_id = cursor.lastrowid
                        member_indices = np.where(labels == label)[0]
                        face_ids_to_update = [face_ids[i] for i in member_indices]
                        placeholders = ','.join('?' * len(face_ids_to_update))
                        await db.execute(f"UPDATE face_embeddings SET cluster_id = ? WHERE id IN ({placeholders})",
                                         [new_cluster_id] + face_ids_to_update)

                    noise_indices = np.where(labels == -1)[0]
                    if noise_indices.size > 0:
                        noise_ids = [face_ids[i] for i in noise_indices]
                        placeholders = ','.join('?' * len(noise_ids))
                        await db.execute(f"UPDATE face_embeddings SET cluster_id = 0 WHERE id IN ({placeholders})", noise_ids)

                    await db.commit()
                    logger.info(f"Clustering complete. Found {len(set(labels) - {-1})} new clusters.")
        except aiosqlite.OperationalError as e:
            logger.error(f"Database error during clustering: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during clustering: {e}", exc_info=True)
        await asyncio.sleep(CLUSTER_INTERVAL_SECONDS)


# ==============================================================================
# --- 6. FASTAPI APP & LIFECYCLE ---
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure all required directories exist
    STAGING_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    await init_databases()

    # --- Load AI Models ---
    logger.info("Loading Insightface model...")
    app.state.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.state.face_app.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("Insightface model loaded.")

    logger.info("Loading BLIP image captioning model...")
    blip_model_id = "Salesforce/blip-image-captioning-base"
    app.state.blip_processor = BlipProcessor.from_pretrained(blip_model_id)
    app.state.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_id)
    logger.info("BLIP model loaded.")

    # --- Initialize Executors and Background Tasks ---
    app.state.process_pool = ProcessPoolExecutor(max_workers=CPU_WORKERS)
    app.state.ai_executor = ThreadPoolExecutor(max_workers=AI_WORKERS)
    app.state.upload_task = asyncio.create_task(process_upload_queue(app))
    app.state.ai_task = asyncio.create_task(process_ai_queue(app))
    app.state.cluster_task = asyncio.create_task(run_face_clustering())
    logger.info("Background workers started.")
    yield
    logger.info("Shutting down...")
    app.state.upload_task.cancel()
    app.state.ai_task.cancel()
    app.state.cluster_task.cancel()
    app.state.process_pool.shutdown(wait=True)
    app.state.ai_executor.shutdown(wait=True)
    logger.info("Shutdown complete.")

app = FastAPI(title="AI Photo Gallery", lifespan=lifespan)
app.mount("/media_files", StaticFiles(directory=OUTPUT_DIR), name="media_files")

def get_full_url(request: Request, relative_path: str | None) -> str:
    return str(request.url_for('media_files', path=relative_path)) if relative_path else ""


# ==============================================================================
# --- 7. API ENDPOINTS ---
# ==============================================================================

@app.post("/upload/check-hashes", response_model=HashCheckResponse)
async def check_file_hashes(payload: HashCheckRequest):
    async with get_media_db() as db:
        placeholders = ','.join('?' for _ in payload.hashes)
        cursor = await db.execute(f"SELECT file_hash FROM media_files WHERE file_hash IN ({placeholders})", payload.hashes)
        existing_hashes = {row['file_hash'] for row in await cursor.fetchall()}
    needed_hashes = [h for h in payload.hashes if h not in existing_hashes]
    return {"needed_hashes": needed_hashes}

@app.post("/upload", status_code=202)
async def upload_files(files: List[UploadFile] = File(...)):
    tasks_created = []
    async with get_media_db() as db:
        for file in files:
            staging_path = STAGING_DIR / f"{uuid.uuid4()}-{file.filename}"
            try:
                with open(staging_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
                server_hash = calculate_hash(staging_path)
                ft = file.content_type.lower()
                file_type = 'image' if ft.startswith('image/') else 'video' if ft.startswith('video/') else 'unsupported'
                if file_type == 'unsupported':
                    staging_path.unlink()
                    continue

                task_id = str(uuid.uuid4())
                await db.execute("INSERT INTO media_files (id, original_filename, staging_path, status, file_type, file_hash) VALUES (?, ?, ?, ?, ?, ?)",
                                 (task_id, file.filename, str(staging_path), TaskStatusEnum.PENDING.value, file_type, server_hash))
                tasks_created.append({"filename": file.filename, "task_id": task_id})
            except Exception as e:
                staging_path.unlink(missing_ok=True)
                logger.error(f"Failed to stage file {file.filename}: {e}")
        await db.commit()
    return {"message": f"{len(tasks_created)} files accepted.", "tasks_created": tasks_created}

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    async with get_media_db() as db:
        cursor = await db.execute("SELECT * FROM media_files WHERE id = ?", (task_id,))
        task = await cursor.fetchone()
    if not task: raise HTTPException(status_code=404, detail="Task not found.")
    return TaskStatus.model_validate(task)

@app.get("/media", response_model=List[MediaItem])
async def get_all_media(request: Request, limit: int = 50, offset: int = 0):
    async with get_ai_db() as db:
        cursor = await db.execute("SELECT * FROM processed_files ORDER BY processed_at DESC LIMIT ? OFFSET ?", (limit, offset))
        files = await cursor.fetchall()
    return [MediaItem(**f, media_url=get_full_url(request, f['relative_path']),
                      thumbnail_url=get_full_url(request, f['relative_thumbnail_path'])) for f in files]

@app.get("/people", response_model=List[Person])
async def get_all_people(request: Request):
    async with get_ai_db() as db:
        query = """
            WITH RankedFaces AS (
                SELECT fe.cluster_id, pf.relative_thumbnail_path,
                ROW_NUMBER() OVER(PARTITION BY fe.cluster_id ORDER BY pf.processed_at DESC) as rn
                FROM face_embeddings fe JOIN processed_files pf ON fe.file_id = pf.id
            )
            SELECT fc.person_id, fc.name, COUNT(fe.id) as face_count, rf.relative_thumbnail_path
            FROM face_clusters fc
            JOIN face_embeddings fe ON fc.cluster_id = fe.cluster_id
            JOIN RankedFaces rf ON fc.cluster_id = rf.cluster_id AND rf.rn = 1
            WHERE fc.cluster_id != 0
            GROUP BY fc.person_id ORDER BY face_count DESC
        """
        people = await db.execute_fetchall(query)
    return [Person(**p, cover_thumbnail_url=get_full_url(request, p['relative_thumbnail_path'])) for p in people]

@app.get("/people/{person_id}", response_model=List[MediaItem])
async def get_person_media(person_id: str, request: Request, limit: int = 50, offset: int = 0):
    async with get_ai_db() as db:
        query = """
            SELECT DISTINCT pf.* FROM processed_files pf
            JOIN face_embeddings fe ON pf.id = fe.file_id
            JOIN face_clusters fc ON fe.cluster_id = fc.cluster_id
            WHERE fc.person_id = ? ORDER BY pf.processed_at DESC LIMIT ? OFFSET ?
        """
        files = await db.execute_fetchall(query, (person_id, limit, offset))
    if not files: raise HTTPException(status_code=404, detail="Person or their media not found.")
    return [MediaItem(**f, media_url=get_full_url(request, f['relative_path']),
                      thumbnail_url=get_full_url(request, f['relative_thumbnail_path'])) for f in files]

@app.get("/search", response_model=List[MediaItem])
async def search_media(
    request: Request,
    q: Optional[str] = Query(None, description="Text query for image captions."),
    person_id: Optional[str] = Query(None, description="Filter by person ID."),
    gender: Optional[str] = Query(None, description="Filter by gender ('Male' or 'Female')."),
    min_age: Optional[int] = Query(None, description="Minimum age of a person in the photo."),
    max_age: Optional[int] = Query(None, description="Maximum age of a person in the photo."),
    limit: int = 50, offset: int = 0
):
    base_query = "SELECT DISTINCT pf.* FROM processed_files pf"
    joins = []
    conditions = []
    params = []

    if q:
        joins.append("JOIN processed_files_fts fts ON pf.id = fts.id")
        conditions.append("fts.caption MATCH ?")
        params.append(q)

    if person_id or gender or min_age is not None or max_age is not None:
        joins.append("JOIN face_embeddings fe ON pf.id = fe.file_id")
        if person_id:
            joins.append("JOIN face_clusters fc ON fe.cluster_id = fc.cluster_id")
            conditions.append("fc.person_id = ?")
            params.append(person_id)
        if gender:
            conditions.append("fe.gender = ?")
            params.append(gender)
        if min_age is not None:
            conditions.append("fe.age >= ?")
            params.append(min_age)
        if max_age is not None:
            conditions.append("fe.age <= ?")
            params.append(max_age)

    query = f"{base_query} {' '.join(set(joins))}"
    if conditions:
        query += f" WHERE {' AND '.join(conditions)}"
    query += " ORDER BY pf.processed_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    async with get_ai_db() as db:
        files = await db.execute_fetchall(query, tuple(params))

    return [MediaItem(**f, media_url=get_full_url(request, f['relative_path']),
                      thumbnail_url=get_full_url(request, f['relative_thumbnail_path'])) for f in files]


# ==============================================================================
# --- 8. RUN APPLICATION ---
# ==============================================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)



