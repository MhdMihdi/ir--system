
import os
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import html
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from services.documents_service import advanced_preprocess , preprocess

def index_with_tfidf(input_path,preprocess = advanced_preprocess, output_dir=r"data/msmarco_train/index/TFIDF"):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    doc_ids = [doc['id'] for doc in data]
    texts = [doc['clean_text'] for doc in data]

    vectorizer = TfidfVectorizer(
        preprocessor=None,
        tokenizer=preprocess,
        max_df=0.85,
        min_df=2,
        sublinear_tf=True,
        norm='l2'
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer_antique_train.joblib"))
    joblib.dump(tfidf_matrix, os.path.join(output_dir, "tfidf_matrix_antique_train.joblib"))
    joblib.dump(doc_ids, os.path.join(output_dir, "doc_ids_antique_train.joblib"))
    print("✅ تم حفظ ملفات TF-IDF بنجاح.")

def index_with_bert(input_path, output_dir="data/msmarco_train/index/BERT", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    doc_ids = [doc['id'] for doc in data]
    texts = [doc['clean_text'] for doc in data]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "bert_embeddings.npy"), embeddings)
    joblib.dump(doc_ids, os.path.join(output_dir, "doc_ids.joblib"))
    print("✅ تم حفظ تمثيلات BERT بنجاح.")



def build_tfidf_inverted_index():
    print("\n--- 1. تحميل الملفات ---")

    doc_ids_path = "data/antique_train/index/TFIDF/doc_ids_antique_train.joblib"
    tfidf_matrix_path = "data/antique_train/index/TFIDF/tfidf_matrix_antique_train.joblib"
    tfidf_vectorizer_path = "data/antique_train/index/TFIDF/tfidf_vectorizer_antique_train.joblib"
    output_inverted_index_path = "data/antique_train/index/TFIDF/tfidf_inverted_index.joblib"

    try:
        doc_ids = joblib.load(doc_ids_path)
        tfidf_matrix = joblib.load(tfidf_matrix_path)
        tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        print("✅ تم تحميل الملفات بنجاح.")
        print(f"عدد المستندات: {len(doc_ids)}")
        print(f"أبعاد مصفوفة TF-IDF: {tfidf_matrix.shape}")
    except Exception as e:
        print(f"❌ خطأ أثناء تحميل الملفات: {e}")
        return

    print("\n--- 2. بناء الفهرس المعكوس ---")
    feature_names = tfidf_vectorizer.get_feature_names_out()
    inverted_index = {}

    tfidf_matrix_coo = tfidf_matrix.tocoo()
    for doc_idx, term_idx, tfidf_score in tqdm(
        zip(tfidf_matrix_coo.row, tfidf_matrix_coo.col, tfidf_matrix_coo.data),
        total=len(tfidf_matrix_coo.data),
        desc="بناء الفهرس"
    ):
        term = feature_names[term_idx]
        doc_id = doc_ids[doc_idx]
        if term not in inverted_index:
            inverted_index[term] = []
        inverted_index[term].append((doc_id, float(tfidf_score)))

    for term in inverted_index:
        inverted_index[term].sort(key=lambda x: x[1], reverse=True)

    inverted_index_data = {
        "inverted_index": inverted_index,
        "num_documents": len(doc_ids),
        "num_terms": len(inverted_index),
        "vocabulary_size": len(feature_names),
        "vectorizer_vocabulary": dict(tfidf_vectorizer.vocabulary_)
    }

    try:
        os.makedirs(os.path.dirname(output_inverted_index_path), exist_ok=True)
        joblib.dump(inverted_index_data, output_inverted_index_path)
        print(f"✅ تم حفظ الفهرس المعكوس في: {output_inverted_index_path}")
    except Exception as e:
        print(f"❌ خطأ أثناء حفظ الفهرس: {e}")

