from __future__ import annotations

import argparse
import math
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


HEIC_EXTENSIONS = {".heic", ".heif"}
MAX_PNG_SIZE_BYTES = 20 * 1024 * 1024


def save_png_with_size_limit(image: Image.Image, png_path: Path, max_size_bytes: int) -> int:
    save_kwargs = {"format": "PNG", "optimize": True, "compress_level": 9}
    current_image = image
    width, height = current_image.size

    while True:
        with BytesIO() as buffer:
            current_image.save(buffer, **save_kwargs)
            contents = buffer.getvalue()

        output_size = len(contents)
        if output_size <= max_size_bytes:
            png_path.write_bytes(contents)
            return output_size

        if width <= 64 or height <= 64:
            png_path.write_bytes(contents)
            return output_size

        scale = math.sqrt(max_size_bytes / output_size) * 0.95
        if scale >= 1.0:
            scale = 0.95

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        if new_width == width and new_height == height:
            new_width = max(1, width - 1)
            new_height = max(1, height - 1)

        current_image = current_image.resize((new_width, new_height), Image.LANCZOS)
        width, height = current_image.size


def convert_heic_to_png(source_dir: Path, output_dir: Path, overwrite: bool = False) -> tuple[int, int]:

    """Convert every HEIC/HEIF image in source_dir into PNG files in output_dir."""
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    register_heif_opener()

    converted = 0
    skipped = 0

    heic_files = sorted(
        path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in HEIC_EXTENSIONS
    )

    for heic_path in heic_files:
        relative_path = heic_path.relative_to(source_dir)
        png_path = output_dir / relative_path.with_suffix(".png")
        png_path.parent.mkdir(parents=True, exist_ok=True)

        if png_path.exists() and not overwrite:
            print(f"Skipped existing: {png_path}")
            skipped += 1
            continue

        with Image.open(heic_path) as image:
            image = ImageOps.exif_transpose(image)
            output_size = save_png_with_size_limit(image, png_path, MAX_PNG_SIZE_BYTES)

        print(f"Converted: {heic_path} -> {png_path} ({output_size / 1024 / 1024:.2f} MB)")
        converted += 1

    return converted, skipped


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Convert all HEIC/HEIF images to PNG.")
    parser.add_argument(
        "--source",
        type=Path,
        default=script_dir / "heic",
        help="Folder containing HEIC/HEIF images. Default: convertion/heic",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "png",
        help="Folder where PNG images will be saved. Default: convertion/png",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite PNG files that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converted, skipped = convert_heic_to_png(args.source, args.output, args.overwrite)
    print(f"\nDone. Converted: {converted}. Skipped: {skipped}.")


if __name__ == "__main__":
    main()
