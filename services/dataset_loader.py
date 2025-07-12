# services/dataset_loader.py
import joblib
import numpy as np
import json
import os
from services.documents_service import advanced_preprocess , preprocess
from services.vector_store import VectorStore

_loaded_datasets_cache = {}

# ✅ تحميل جميع الداتا سيتات المعرفة مسبقًا عند بداية البرنامج
PREDEFINED_DATASETS = ["msmarco_train", "beir_quora_test"]


def preload_all_datasets():
    print("loading dataset files")
    for dataset_name in PREDEFINED_DATASETS:
        if dataset_name not in _loaded_datasets_cache:
            print(f"📦 Loading dataset files {dataset_name}")
            _loaded_datasets_cache[dataset_name] = load_dataset_files_internal(dataset_name)


def load_dataset_files(dataset_name):
    # ✅ من الكاش فقط
    if dataset_name in _loaded_datasets_cache:
        return _loaded_datasets_cache[dataset_name]
    else:
        raise ValueError(f"❌ الداتا سيت غير محملة مسبقًا: {dataset_name}")


def load_dataset_files_internal(dataset_name):
    base_path = f"data/{dataset_name.replace('/', '_')}/index"
    vector_store_path = f"{base_path}/vector_store"

    vector_store = None
    if os.path.exists(vector_store_path):
        try:
            vector_store = VectorStore.load(vector_store_path)
        except Exception as e:
            print(f"⚠️ vector store not loaded: {e}")

    # TF-IDF
    tfidf_doc_ids = joblib.load(f"{base_path}/TFIDF/doc_ids_{dataset_name.replace('/', '_')}.joblib")
    tfidf_matrix = joblib.load(f"{base_path}/TFIDF/tfidf_matrix_{dataset_name.replace('/', '_')}.joblib")
    tfidf_vectorizer = joblib.load(f"{base_path}/TFIDF/tfidf_vectorizer_{dataset_name.replace('/', '_')}.joblib")
    inverted_index_data = joblib.load(f"{base_path}/TFIDF/tfidf_inverted_index.joblib")

    # BERT
    bert_embeddings = np.load(f"{base_path}/bert/bert_embeddings.npy")
    bert_doc_ids = joblib.load(f"{base_path}/bert/doc_ids.joblib")

    # Docs
    docs_dict = {}
    with open(f"data/{dataset_name.replace('/', '_')}/raw/raw_{dataset_name.replace('/', '_')}.json", "r", encoding="utf-8") as f:
        for line in f:
            try:
                doc = json.loads(line)
                doc_id = doc.get("id")
                doc_text = doc.get("text")
                if doc_id is not None:
                    docs_dict[str(doc_id)] = doc_text
            except json.JSONDecodeError:
                continue

    if vector_store is None:
        vector_store = VectorStore(dimension=bert_embeddings.shape[1])
        vector_store.add_documents(bert_doc_ids, bert_embeddings, docs_dict)
        try:
            vector_store.save(vector_store_path)
            print("✅ vector store saved ")
        except Exception as e:
            print(f"⚠️ vector store not saved: {e}")

    return {
        "tfidf_doc_ids": tfidf_doc_ids,
        "tfidf_matrix": tfidf_matrix,
        "tfidf_vectorizer": tfidf_vectorizer,
        "inverted_index_data": inverted_index_data,
        "bert_embeddings": bert_embeddings,
        "bert_doc_ids": bert_doc_ids,
        "docs_dict": docs_dict,
        "vector_store": vector_store,
    }
