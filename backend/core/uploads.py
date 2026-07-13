from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, UploadFile

from backend.core.config import settings


def _chunk_size() -> int:
    return max(64 * 1024, settings.UPLOAD_CHUNK_SIZE_KB * 1024)


async def read_upload_limited(
    upload: UploadFile,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Read an upload incrementally and stop as soon as it exceeds the limit."""
    data = bytearray()

    while chunk := await upload.read(_chunk_size()):
        data.extend(chunk)
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{label} exceeds the {max_bytes // (1024 * 1024)} MB limit",
            )

    return bytes(data)


async def save_upload_limited(
    upload: UploadFile,
    destination: Path,
    *,
    max_bytes: int,
    label: str,
) -> int:
    """Stream an upload to disk without retaining the full file in memory."""
    total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(_chunk_size()):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"{label} exceeds the "
                            f"{max_bytes // (1024 * 1024)} MB limit"
                        ),
                    )
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    return total
