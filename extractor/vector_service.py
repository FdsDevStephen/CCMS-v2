from __future__ import annotations
import uuid
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # Or your local HuggingFaceEmbeddings / FastEmbed

class CaseVectorService:
    def __init__(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.text = text
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings (replace with HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") if local)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.session_id = str(uuid.uuid4())[:8]
        self.vectorstore = self._build_vector_store()

    def _build_vector_store(self) -> Chroma:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )
        docs = splitter.create_documents([self.text])
        
        return Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name=f"case_{self.session_id}"
        )

    def retrieve_act_chunks(self, sections: List[str], top_k: int = 4) -> List[str]:
        """
        Retrieves only chunks mentioning statutory acts and mapped sections.
        """
        section_str = " ".join([f"Section {s}" for s in sections[:10]])
        queries = [
            f"Statutory Act, Karnataka Land Reforms, Karnataka Land Revenue, enactment legislation {section_str}",
            "Act 1961, Act 1964, Act 1882 read with section violation"
        ]

        retrieved_docs = []
        for q in queries:
            docs = self.vectorstore.similarity_search(q, k=top_k)
            retrieved_docs.extend(docs)

        # Deduplicate
        unique_chunks = list({doc.page_content: doc for doc in retrieved_docs}.values())
        return [doc.page_content for doc in unique_chunks[:top_k]]

    def retrieve_survey_chunks(self, survey_number: str, top_k: int = 2) -> str:
        """
        Retrieves specific context chunks surrounding a given survey number and its village/hobli/taluk/district.
        """
        query = (
            f"Survey No. {survey_number} Sy. No. {survey_number} Re-Survey No. {survey_number} "
            f"village hobli taluk district measuring acres guntas situated at"
        )
        docs = self.vectorstore.similarity_search(query, k=top_k)
        
        # Merge the top 1-2 chunks
        unique_chunks = list({d.page_content: d for d in docs}.values())
        return "\n...\n".join([d.page_content for d in unique_chunks])

    def cleanup(self):
        """Delete in-memory collection after processing."""
        try:
            self.vectorstore.delete_collection()
        except Exception:
            pass