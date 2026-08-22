import time


print("=" * 60)
print("CHUNKER SPEED TEST")
print("=" * 60)


start = time.perf_counter()

from RAG.chunker import LegalTextChunker

import_time = time.perf_counter() - start

print(f"1. CHUNKER IMPORT TIME       : {import_time:.2f} seconds")


start = time.perf_counter()

chunker = LegalTextChunker(
    chunk_size=450,
    overlap=50,
)

init_time = time.perf_counter() - start

print(f"2. CHUNKER INITIALIZATION    : {init_time:.2f} seconds")


print("=" * 60)
print(f"TOTAL                         : {import_time + init_time:.2f} seconds")
print("=" * 60)