import hashlib
import  json
import urllib.request
from pathlib import Path
URL = (
    "https://elearning.reb.rw/pluginfile.php/11928/"
    "mod_resource/content/1/HISTORY%20S5%20SB.pdf"
)

def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "data/raw/rwanda_history_sources"
    destination = folder / "reb_history_s5.pdf"
    metadata_file = folder / "reb_history_s5_source.json"

    if destination.exists() or metadata_file.exists():
        raise FileExistsError(
            "Source files already exist. Nothing overwritten."
        )

    print("Downloading REB Senior 5 history textbook...", flush=True)

    request = urllib.request.Request(
        URL,
        headers={"User-Agent": "EducationQAResearch/1.0"},
    )

    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()
        final_url = response.geturl()

    if not data.startswith(b"%PDF-"):
        raise ValueError(
            "The server did not return a PDF. No files saved."
        )

    folder.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as file:
        file.write(data)

    metadata = {
        "publisher": "Rwanda Basic Education Board",
        "source_url": URL,
        "resolved_url": final_url,
        "filename": destination.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "purpose": "Reference material for preparing history QA",
        "reuse_terms_reviewed": False,
        "question_answer_pairs_prepared": False,
        "training_started": False,
    }

    with metadata_file.open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Saved:", destination)
    print("Size:", len(data), "bytes")
    print("Source metadata:", metadata_file)
    print("No training started.")


if __name__ == "__main__":
    main()