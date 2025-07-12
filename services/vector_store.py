import numpy as np
import faiss
import os
import json
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import normalize

class VectorStore:
    def __init__(self, dimension: int = 384):  # 384 هو حجم vectors من MiniLM-L6-v2
        """
        تهيئة vector store باستخدام FAISS
        dimension: حجم الـ vector (يعتمد على نموذج BERT المستخدم)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # استخدام Inner Product للتشابه بعد Normalization
        self.doc_ids: List[str] = []
        self.docs_dict: Dict[str, str] = {}

    def add_documents(self, doc_ids: List[str], embeddings: np.ndarray, docs_dict: Dict[str, str]):
        """
        إضافة مستندات جديدة إلى vector store
        """
        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError("عدد المستندات لا يتطابق مع عدد الـ embeddings")

        self.doc_ids.extend(doc_ids)
        self.docs_dict.update(docs_dict)

        # تطبيق normalization على كل vector
        normalized_embeddings = normalize(embeddings, axis=1)
        self.index.add(normalized_embeddings.astype(np.float32))

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        البحث عن أقرب المستندات لـ query vector
        """
        print("searchinh from vector store")
       
        if query_vector.shape[1] != self.dimension:
            raise ValueError(f"يجب أن يكون حجم query vector {self.dimension}")

        # Normalize query vector
        normalized_query = normalize(query_vector, axis=1)
        scores, indices = self.index.search(normalized_query.astype(np.float32), top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.doc_ids):
                doc_id = self.doc_ids[idx]
                doc_text = self.docs_dict.get(str(doc_id), "⚠️ نص الوثيقة غير موجود.")
                results.append((doc_id, doc_text, float(score)))

        return results

    def save(self, directory: str):
        """
        حفظ vector store على القرص
        """
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "faiss.index"))
        metadata = {
            "dimension": self.dimension,
            "doc_ids": self.doc_ids,
            "docs_dict": self.docs_dict
        }
        with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """
        تحميل vector store من القرص
        """
        with open(os.path.join(directory, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)

        store = cls(dimension=metadata["dimension"])
        store.doc_ids = metadata["doc_ids"]
        store.docs_dict = metadata["docs_dict"]
        store.index = faiss.read_index(os.path.join(directory, "faiss.index"))

        return store
