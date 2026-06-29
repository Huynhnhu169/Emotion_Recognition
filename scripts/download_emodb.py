"""Download and extract the EmoDB speech emotion dataset.

The default project config expects EmoDB `.wav` files under `wav/`.
Kaggle may require authentication depending on your environment. If this script
fails with an authorization error, download the ZIP manually from Kaggle and
extract the `.wav` files into the output directory.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "piyushagni5/berlin-database-of-emotional-speech-emodb"
)


def download_file(url: str, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"Saving to {archive_path}")
    with urllib.request.urlopen(url) as response, archive_path.open("wb") as output:
        shutil.copyfileobj(response, output)


def extract_wavs(archive_path: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".wav"):
                continue
            target = output_dir / Path(member.filename).name
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EmoDB and extract .wav files.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Dataset ZIP URL.")
    parser.add_argument("--archive", default="downloads/emodb.zip", help="Where to save the ZIP archive.")
    parser.add_argument("--output-dir", default="wav", help="Directory for extracted .wav files.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not re-download an existing archive.")
    args = parser.parse_args()

    archive_path = Path(args.archive)
    output_dir = Path(args.output_dir)

    if not archive_path.exists() or not args.skip_existing:
        download_file(args.url, archive_path)

    count = extract_wavs(archive_path, output_dir)
    print(f"Extracted {count} wav files to {output_dir}")


if __name__ == "__main__":
    main()
