from retriever import LegalRetriever


retriever = LegalRetriever()


query = "Which Acts are mentioned in the case?"


print("=" * 80)
print("QUERY")
print("=" * 80)

print(query)


print("\n" + "=" * 80)
print("RETRIEVED CHUNKS")
print("=" * 80)


results = retriever.search(
    query=query,
    top_k=5,
)


print(
    f"\nResults: {len(results)}\n"
)


for index, result in enumerate(
    results,
    start=1,
):

    print("=" * 80)

    print(
        f"RESULT {index}"
    )

    print(
        f"Score    : {result.score}"
    )

    print(
        f"Document : "
        f"{result.payload['document']}"
    )

    print(
        f"Section  : "
        f"{result.payload['section']}"
    )

    print(
        f"Chunk ID  : "
        f"{result.payload['chunk_id']}"
    )

    print("\nTEXT:")

    print(
        result.payload["text"]
    )