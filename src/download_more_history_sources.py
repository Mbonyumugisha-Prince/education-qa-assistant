import hashlib
import json
import urllib.request
from pathlib import Path


SOURCES = [
    {
        "filename": "reb_history_s4.pdf",
        "title": "REB History Senior 4",
        "url": (
            "https://uat-coremis.reb.rw/"
            "mod/resource/view.php?id=12569"
        ),
    },
    {
        "filename": "reb_history_s6.pdf",
        "title": "REB History Senior 6",
        "url": (
            "https://elearning.reb.rw/pluginfile.php/86289/"
            "mod_resource/content/1/"
            "GENERAL%20EDU-History-S6-SB%20%281%29.pdf"
        ),
    },
]


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "data/raw/rwanda_history_sources"
    folder.mkdir(parents=True, exist_ok=True)

    for source in SOURCES:
        destination = folder / source["filename"]
        metadata_file = destination.with_suffix(".source.json")

        if destination.exists():
            print("Already exists; preserved:", destination)
            continue

        if metadata_file.exists():
            raise FileExistsError(
                f"Metadata already exists: {metadata_file}"
            )

        print("Downloading:", source["title"], flush=True)

        try:
            request = urllib.request.Request(
                source["url"],
                headers={"User-Agent": "EducationQAResearch/1.0"},
            )
            with urllib.request.urlopen(
                request, timeout=90
            ) as response:
                data = response.read()
                resolved_url = response.geturl()

            if not data.startswith(b"%PDF-"):
                raise ValueError(
                    "Server returned a page instead of a PDF."
                )

        except Exception as error:
            print("Download failed:", error)
            print("No file saved for this source.")
            continue

        with destination.open("xb") as file:
            file.write(data)

        metadata = {
            **source,
            "publisher": "Rwanda Basic Education Board",
            "resolved_url": resolved_url,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "purpose": "Local history reference collection",
            "reuse_terms_reviewed": False,
            "training_started": False,
        }
        with metadata_file.open("x", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        print("Saved:", destination)

    print("\nAvailable PDFs:")
    for path in sorted(folder.glob("*.pdf")):
        print(path.name)

    print("\nNo training performed.")
    print("Existing books and adapters preserved.")


if __name__ == "__main__":
    main()