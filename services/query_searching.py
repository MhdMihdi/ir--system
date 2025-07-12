
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from services.documents_service import query_advanced_preprocess ,semantic_spell_check

bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def preprocess_query(query):
    """
    معالجة الاستعلام مع التصحيح الدلالي واكتشاف المواضيع
    """
    # تصحيح الأخطاء الإملائية دلالياً
    corrected_query = semantic_spell_check(query)
    
    # تنظيف النص
    cleaned_query = query_advanced_preprocess(corrected_query)
    
    
    return {
        'original': query,
        'corrected': corrected_query,
        'cleaned': cleaned_query,
    }


def search_tfidf_with_inverted_index(query, inverted_index_data, tfidf_vectorizer, tfidf_matrix, doc_ids, docs_dict, top_k=10, candidate_size=100):
    # معالجة الاستعلام
    query_info = preprocess_query(query)
    cleaned_query = query_info['cleaned']
    query_terms = cleaned_query.split()
    
    if not query_terms:
        return []

    # 2. ترشيح الوثائق باستخدام الفهرس المعكوس (مجموع درجات tfidf)
    doc_scores = {}
    for term in query_terms:
        if term in inverted_index_data["inverted_index"]:
            postings = inverted_index_data["inverted_index"][term]
            for doc_id, score in postings:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += score

    # 3. اختيار أفضل candidate_size وثيقة للمعالجة الدقيقة
    candidate_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_size]
    candidate_doc_ids = [doc_id for doc_id, _ in candidate_docs]

    # 4. إيجاد إندكسات هذه الوثائق في مصفوفة tfidf_matrix
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    candidate_indices = [doc_id_to_index[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_id_to_index]

    # 5. بناء مصفوفة tfidf للوثائق المرشحة فقط
    candidate_tfidf_matrix = tfidf_matrix[candidate_indices]

    # 6. تحويل الاستعلام إلى تمثيل tfidf
    query_vector = tfidf_vectorizer.transform([cleaned_query])

    # 7. حساب cosine similarity بين الاستعلام والوثائق المرشحة فقط
    cosine_scores = cosine_similarity(query_vector, candidate_tfidf_matrix).flatten()

    # 8. اختيار أفضل top_k وثائق حسب cosine similarity
    top_indices = cosine_scores.argsort()[::-1][:top_k]

    # 9. تجهيز النتائج للعرض
    results = []
    for idx in top_indices:
        doc_idx = candidate_indices[idx]
        doc_id = doc_ids[doc_idx]
        doc_text = docs_dict.get(str(doc_id), "⚠️ نص الوثيقة غير موجود.")
        score = cosine_scores[idx]
        results.append({
            'doc_id': doc_id,
            'text': doc_text,
            'score': score,
            'query_info': query_info
        })

    return results


def search_bert(query, bert_embeddings, bert_doc_ids, docs_dict, vector_store=None, top_k=10):
    query_info = preprocess_query(query)
    query_embedding = bert_model.encode([query_info['cleaned']], normalize_embeddings=True).astype(np.float32)

    results = []
    if vector_store is not None:
        vector_store_results = vector_store.search(query_embedding, top_k=top_k)
        results = [
            {
                'doc_id': doc_id,
                'text': text,
                'score': score,
                'query_info': query_info
            }
            for doc_id, text, score in vector_store_results
        ]
    else:
        bert_scores = cosine_similarity(query_embedding, bert_embeddings).flatten()
        top_indices = np.argsort(bert_scores)[::-1][:top_k]

        for i in top_indices:
            doc_id = bert_doc_ids[i]
            doc_text = docs_dict.get(doc_id, "⚠️ نص الوثيقة غير موجود.")
            score = bert_scores[i]
            results.append({
                'doc_id': doc_id,
                'text': doc_text,
                'score': score,
                'query_info': query_info
            })

    return results


# def search_bert(query, bert_embeddings, bert_doc_ids, docs_dict, top_k=10):
#     # معالجة الاستعلام
#     query_info = preprocess_query(query)
    
#     # تحويل الاستعلام إلى vector
#     query_embedding = bert_model.encode([query_info['corrected']])
    
#     # استخدام vector store للبحث السريع
#     results = []
#     if 'vector_store' in docs_dict:  # إذا كان vector store متاحاً
#         vector_store_results = docs_dict['vector_store'].search(query_embedding, top_k)
#         results = [
#             {
#                 'doc_id': doc_id,
#                 'text': text,
#                 'score': score,
#                 'query_info': query_info
#             }
#             for doc_id, text, score in vector_store_results
#         ]
#     else:  # استخدام الطريقة التقليدية كاحتياط
#         bert_scores = cosine_similarity(query_embedding, bert_embeddings).flatten()
#         top_indices = np.argsort(bert_scores)[::-1][:top_k]
        
#         for i in top_indices:
#             doc_id = bert_doc_ids[i]
#             doc_text = docs_dict.get(doc_id, "⚠️ نص الوثيقة غير موجود.")
#             score = bert_scores[i]
#             results.append({
#                 'doc_id': doc_id,
#                 'text': doc_text,
#                 'score': score,
#                 'query_info': query_info
#             })
    
#     return results

def search_hybrid(query, tfidf_vectorizer, tfidf_matrix, bert_embeddings, tfidf_doc_ids, bert_doc_ids, docs_dict, vector_store=None, tfidf_weight=0.4, bert_weight=0.6, top_k=10):
    assert tfidf_weight + bert_weight == 1.0, "يجب أن يكون مجموع الأوزان 1.0"

    query_info = preprocess_query(query)

    # TF-IDF Scores
    tfidf_scores = cosine_similarity(
        tfidf_vectorizer.transform([query_info['cleaned']]),
        tfidf_matrix
    ).flatten()

    # BERT embedding
    query_embedding = bert_model.encode([query_info['cleaned']], normalize_embeddings=True).astype(np.float32)

    # Initialize empty BERT scores
    bert_scores = np.zeros_like(tfidf_scores)

    if vector_store is not None:
        # ✅ استرجع فقط النتائج العلوية من FAISS
        top_bert_results = vector_store.search(query_embedding, top_k=top_k * 20)

        # أنشئ mapping أسرع
        doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(bert_doc_ids)}

        for doc_id, _, score in top_bert_results:
            idx = doc_id_to_index.get(doc_id)
            if idx is not None:
                bert_scores[idx] = score
    else:
        # fallback
        bert_scores = cosine_similarity(query_embedding, bert_embeddings).flatten()

    if tfidf_doc_ids != bert_doc_ids:
        raise ValueError("قوائم doc_ids غير متطابقة بين النموذجين!")

    combined_scores = tfidf_weight * tfidf_scores + bert_weight * bert_scores
    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    results = []
    for i in top_indices:
        doc_id = tfidf_doc_ids[i]
        doc_text = docs_dict.get(doc_id, "⚠️ نص الوثيقة غير موجود.")
        score = combined_scores[i]
        results.append({
            'doc_id': doc_id,
            'text': doc_text,
            'score': score,
            'query_info': query_info
        })

    return results
