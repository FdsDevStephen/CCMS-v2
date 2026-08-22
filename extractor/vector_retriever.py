from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class VectorRetriever:
    def __init__(self, text: str, chunk_size: int = 800, chunk_overlap: int = 100):
        self.text = text
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = self._build_store(chunk_size, chunk_overlap)

    def _build_store(self, chunk_size: int, chunk_overlap: int):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )
        docs = splitter.create_documents([self.text])
        # Build in-memory FAISS index
        return FAISS.from_documents(docs, self.embeddings)

    def get_top_5_chunks(self) -> str:
        queries = [
            "Act, statutory legislation, Karnataka Land Reforms, Karnataka Land Revenue, enactments",
            "Survey Number, Sy No, Re-survey, village, hobli, taluk, district, land schedule"
        ]
        
        matches = []
        for q in queries:
            results = self.vectorstore.similarity_search(q, k=3)
            matches.extend(results)

        # Deduplicate to strictly top 5 chunks
        unique_chunks = list({doc.page_content: doc for doc in matches}.values())[:5]
        return "\n\n--- [CHUNK] ---\n\n".join(unique_chunks)