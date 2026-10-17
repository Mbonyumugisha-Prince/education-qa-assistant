import hashlib
import json
import urllib.request
from pathlib import Path


SOURCES = [
    {
        "filename": "openstax_world_history_v1.pdf",
        "title": "World History, Volume 1: to 1500",
        "url": (
            "https://assets.openstax.org/oscms-prodcms/media/"
            "documents/World_History_Volume_1-WEB.pdf"
        ),
    },
    {
        "filename": "openstax_world_history_v2.pdf",
        "title": "World History, Volume 2: from 1400",
        "url": (
            "https://assets.openstax.org/oscms-prodcms/media/"
            "documents/World_History_Volume_2-WEB.pdf"
        ),
    },
]


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "data/raw/rwanda_history_sources"
    folder.mkdir(parents=True, exist_ok=True)

    for source in SOURCES:
        destination = folder / source["filename"]
        temporary = destination.with_suffix(".pdf.part")
        metadata_file = destination.with_suffix(".source.json")

        if destination.exists():
            print("Already exists; preserved:", destination)
            continue

        if temporary.exists() or metadata_file.exists():
            raise FileExistsError(
                f"Partial download or metadata already exists "
                f"for {destination.name}. Existing files preserved."
            )

        print("\nDownloading:", source["title"], flush=True)
        request = urllib.request.Request(
            source["url"],
            headers={"User-Agent": "Mozilla/5.0"},
        )

        digest = hashlib.sha256()
        downloaded = 0
        next_progress = 10 * 1024 * 1024

        try:
            with urllib.request.urlopen(
                request, timeout=120
            ) as response:
                resolved_url = response.geturl()
                total = int(response.headers.get("Content-Length") or 0)

                with temporary.open("xb") as file:
                    first_chunk = True

                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break

                        if first_chunk:
                            if not chunk.startswith(b"%PDF-"):
                                raise ValueError(
                                    "Server did not return a PDF."
                                )
                            first_chunk = False

                        file.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)

                        if downloaded >= next_progress:
                            print(
                                f"Downloaded {downloaded / 1024**2:.1f} MB",
                                flush=True,
                            )
                            next_progress = downloaded + 10 * 1024 * 1024

                if downloaded == 0:
                    raise ValueError("Empty download.")
                if total and downloaded != total:
                    raise ValueError("Incomplete download.")

            temporary.rename(destination)

        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise

        metadata = {
            **source,
            "publisher": "OpenStax",
            "resolved_url": resolved_url,
            "sha256": digest.hexdigest(),
            "bytes": downloaded,
            "purpose": "World-history reference collection",
            "training_started": False,
        }

        with metadata_file.open("x", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        print("Saved:", destination)

    print("\nAvailable history PDFs:")
    for path in sorted(folder.glob("*.pdf")):
        print(path.name)

    print("\nNo training performed; existing adapters preserved.")


if __name__ == "__main__":
    main()