"""Audit the local traffic-law source manifest."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus import iter_local_traffic_records, load_source_manifest


def main() -> None:
    manifest = load_source_manifest()
    print(f"Source policy: {manifest['source_policy']}")
    print(f"Manifest documents: {len(manifest['documents'])}")
    print()

    total_chars = 0
    enabled = 0
    for record in iter_local_traffic_records():
        enabled += 1
        chars = len(record["text"])
        total_chars += chars
        print(f"- {record['doc_id']}: {chars:,} chars")
        print(f"  title: {record['title']}")
        print(f"  file:  {record['source_path']}")

    print()
    print(f"Enabled local text documents: {enabled}")
    print(f"Total trusted corpus size: {total_chars:,} chars")
    if enabled == 0:
        raise SystemExit("No enabled local text documents found.")


if __name__ == "__main__":
    main()
