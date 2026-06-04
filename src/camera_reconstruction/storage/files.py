from __future__ import annotations

from pathlib import Path
from uuid import uuid4


ALLOWED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def unique_image_path(directory: Path, original_filename: str, fallback_extension: str = ".png") -> Path:
    suffix = Path(original_filename or "").suffix.lower()
    if not suffix:
        suffix = fallback_extension
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        raise ValueError(f"Unsupported image extension '{suffix}'. Allowed: {allowed}")

    return ensure_directory(directory) / f"{uuid4().hex}{suffix}"


def save_upload_bytes(contents: bytes, directory: Path, original_filename: str) -> Path:
    if not contents:
        raise ValueError("Uploaded image is empty")

    output_path = unique_image_path(directory, original_filename)
    output_path.write_bytes(contents)
    return output_path

