import numpy as np
from typing import List, Dict, Set
import json
import os
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class QueryRefinement:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        تهيئة نظام تحسين الاستعلامات
        """
        self.bert_model = SentenceTransformer(model_name)
        self.tfidf_vectorizer = TfidfVectorizer()
        
        # قاموس للكلمات المترادفة والمرتبطة
        self.related_terms = {
            "document": ["text", "file", "paper", "article"],
            "search": ["find", "lookup", "query", "retrieve"],
            "database": ["db", "data store", "repository"],
            # يمكن إضافة المزيد حسب المجال
        }
        
        # تخزين الاستعلامات السابقة
        self.query_history: List[str] = []
        self.query_embeddings = None
        self.popular_queries: Dict[str, int] = defaultdict(int)
        
        # تحميل البيانات المحفوظة إن وجدت
        self.load_data()
        
    def add_query(self, query: str):
        """
        إضافة استعلام جديد إلى التاريخ
        """
        self.query_history.append(query)
        self.popular_queries[query] += 1
        
        # تحديث embeddings
        if len(self.query_history) % 10 == 0:  # تحديث كل 10 استعلامات لتحسين الأداء
            self.update_embeddings()
            self.save_data()
            
    def update_embeddings(self):
        """
        تحديث embeddings للاستعلامات
        """
        if self.query_history:
            self.query_embeddings = self.bert_model.encode(self.query_history)
            
    def save_data(self):
        """
        حفظ البيانات على القرص
        """
        data = {
            "query_history": self.query_history,
            "popular_queries": dict(self.popular_queries)
        }
        os.makedirs("data/query_refinement", exist_ok=True)
        with open("data/query_refinement/data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def load_data(self):
        """
        تحميل البيانات من القرص
        """
        try:
            with open("data/query_refinement/data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.query_history = data["query_history"]
                self.popular_queries = defaultdict(int, data["popular_queries"])
                self.update_embeddings()
        except (FileNotFoundError, json.JSONDecodeError):
            pass
            
    def get_related_terms(self, query: str) -> Set[str]:
        """
        الحصول على المصطلحات المرتبطة
        """
        related = set()
        words = query.lower().split()
        
        for word in words:
            if word in self.related_terms:
                related.update(self.related_terms[word])
                
        return related
        
    def get_similar_queries(self, query: str, top_k: int = 5) -> List[str]:
        """
        البحث عن استعلامات مشابهة من التاريخ
        """
        if not self.query_history or self.query_embeddings is None:
            return []
            
        # حساب embedding للاستعلام الحالي
        query_embedding = self.bert_model.encode([query])
        
        # حساب التشابه مع الاستعلامات السابقة
        similarities = cosine_similarity(query_embedding, self.query_embeddings)[0]
        
        # اختيار أفضل النتائج
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [self.query_history[i] for i in top_indices if similarities[i] > 0.5]
        
    def get_popular_queries(self, top_k: int = 5) -> List[str]:
        """
        الحصول على أكثر الاستعلامات شيوعاً
        """
        return sorted(
            self.popular_queries.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
    def refine_query(self, query: str) -> Dict:
        """
        تحسين الاستعلام وتقديم اقتراحات
        """
        # إضافة الاستعلام الحالي
        self.add_query(query)
        
        # جمع الاقتراحات
        related_terms = self.get_related_terms(query)
        similar_queries = self.get_similar_queries(query)
        popular_queries = self.get_popular_queries()
        
        # تجميع النتائج
        suggestions = {
            "related_terms": list(related_terms),
            "similar_queries": similar_queries,
            "popular_queries": [q for q, _ in popular_queries],
            "refined_query": query  # يمكن إضافة منطق لتحسين الاستعلام نفسه
        }
        
        return suggestions 