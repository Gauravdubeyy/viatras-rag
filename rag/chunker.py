import json
from typing import List

def load_and_chunk_manual(file_path: str) -> List[dict]:
    """
    Loads manual.json and converts each entry into a chunk.
    Combines section + subsection + content for richer embeddings.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        print(f"✅ JSON manual loaded: {len(entries)} entries")

        chunks = []
        for entry in entries:
            section = entry.get("section", "")
            subsection = entry.get("subsection", "")
            content = entry.get("content", "")

            # Build rich text for embedding
            if subsection:
                rich_text = f"{section} > {subsection}: {content}"
            else:
                rich_text = f"{section}: {content}"

            chunks.append({
                "chunk_id": entry["id"],
                "text": rich_text,
                "section": section,
                "subsection": subsection
            })

        print(f"✅ Prepared {len(chunks)} chunks for indexing")
        return chunks

    except FileNotFoundError:
        raise RuntimeError(f"❌ Manual file not found at: {file_path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"❌ Invalid JSON in manual file: {e}")
