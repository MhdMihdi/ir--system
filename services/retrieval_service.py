import numpy as np
from services.dataset_loader import load_dataset_files
from services.query_searching import (
    search_tfidf_with_inverted_index,
    search_bert,
    search_hybrid,
)




def retrieve_results(query, dataset, model,use_vectorstore = False):
    data = load_dataset_files(dataset)

    if model == "tfidf":
        return search_tfidf_with_inverted_index(
            query,
            data["inverted_index_data"],
            data["tfidf_vectorizer"],
            data["tfidf_matrix"],
            data["tfidf_doc_ids"],
            data["docs_dict"],
        )
    elif model == "bert":
        return search_bert(
            query,
            data["bert_embeddings"],
            data["bert_doc_ids"],
            data["docs_dict"],
            data["vector_store"] if use_vectorstore  else None
        )
    elif model == "hybrid":
        return search_hybrid(
            query,
            data["tfidf_vectorizer"],
            data["tfidf_matrix"],
            data["bert_embeddings"],
            data["tfidf_doc_ids"],
            data["bert_doc_ids"],
            data["docs_dict"],
            data["vector_store"] if use_vectorstore  else None,
            tfidf_weight=0.4,
            bert_weight=0.6,
        )
    else:
        raise ValueError("Model not supported.")
