"""
api/utils.py
File validation and auto-delete logic for uploaded videos.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)

## Supported video MIME types
ALLOWED_MIME_TYPES = {
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
}

## 100 MB limit
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024

## Temp upload directory — created on first use
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

## Auto-delete delay in seconds (5 minutes)
AUTO_DELETE_SECONDS = 300


def validate_video(file: UploadFile) -> None:
    """Raise HTTPException if the file is not a supported video type."""
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Accepted types: {', '.join(sorted(ALLOWED_MIME_TYPES))}"
            ),
        )


async def save_upload(file: UploadFile) -> Path:
    """
    Stream the uploaded file to a temp path, enforce the size limit,
    and return the saved path.
    """
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    bytes_written = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):  ## 1 MB chunks
            bytes_written += len(chunk)
            if bytes_written > MAX_FILE_SIZE_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the 100 MB size limit.",
                )
            out.write(chunk)

    return dest


def schedule_delete(path: Path) -> None:
    """Schedule deletion of a file after AUTO_DELETE_SECONDS."""
    async def _delete():
        await asyncio.sleep(AUTO_DELETE_SECONDS)
        try:
            path.unlink(missing_ok=True)
            logger.info("Auto-deleted: %s", path)
        except Exception as exc:
            logger.warning("Could not delete %s: %s", path, exc)

    asyncio.create_task(_delete())
