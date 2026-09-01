print("TEST STARTED", flush=True)

from rag.chunker import LegalTextChunker

print("CHUNKER IMPORTED", flush=True)

TXT_PATH = "../section_output/WP-14650-2021-B.txt"

chunker = LegalTextChunker()

print("CHUNKER CREATED", flush=True)

chunks = chunker.chunk_file(TXT_PATH)

print("CHUNKING FINISHED", flush=True)

print(f"Total chunks: {len(chunks)}", flush=True)

for chunk in chunks:
    print("=" * 80, flush=True)
    print(f"Document : {chunk['document']}", flush=True)
    print(f"Section  : {chunk['section']}", flush=True)
    print(f"Chunk ID  : {chunk['chunk_id']}", flush=True)
    print(chunk["text"], flush=True)