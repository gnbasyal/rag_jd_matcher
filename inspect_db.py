"""Quick ChromaDB inspection utility."""
import chromadb
from app.config import settings

client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
collection = client.get_collection(settings.jd_collection_name)

total = collection.count()
print(f"Total chunks: {total}\n")

# Sample first 5 chunks
sample = collection.get(limit=5, include=["metadatas", "documents", "embeddings"])

for i, (doc, meta, emb) in enumerate(zip(sample["documents"], sample["metadatas"], sample["embeddings"])):
    print(f"── Chunk {i+1} ──────────────────────────")
    print(f"  jd_id    : {meta.get('jd_id')}")
    print(f"  title    : {meta.get('title')}")
    print(f"  company  : {meta.get('company')}")
    print(f"  seniority: {meta.get('seniority')}")
    print(f"  skills   : {meta.get('required_skills', '')[:80]}")
    print(f"  tech     : {meta.get('tech_stack', '')[:80]}")
    print(f"  text     : {doc[:120].strip()!r}")
    print(f"  vector   : dim={len(emb)}  [{emb[0]:.4f}, {emb[1]:.4f}, ..., {emb[-1]:.4f}]")
    print()

# Unique JDs indexed
all_meta = collection.get(include=["metadatas"])["metadatas"]
jd_ids = sorted({m["jd_id"] for m in all_meta})
print(f"Indexed JDs ({len(jd_ids)}): {', '.join(jd_ids)}")
