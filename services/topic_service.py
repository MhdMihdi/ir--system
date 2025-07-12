import os
import json
from gensim import corpora, models

lda_model=None
dictionary =None
# --- تدريب نموذج LDA ---
def train_lda_model(texts, num_topics=10):
    global lda_model, dictionary

    texts_words = [text.split() for text in texts]
    dictionary = corpora.Dictionary(texts_words)
    corpus = [dictionary.doc2bow(text) for text in texts_words]

    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

# --- حفظ توزيع التوبيكات لكل وثيقة في ملف JSON ---
def save_document_topics(texts, output_path ="models/"):
    if lda_model is None or dictionary is None:
        raise ValueError("يجب تدريب النموذج أولاً")
   
    texts_words = [text.split() for text in texts]
    corpus = [dictionary.doc2bow(text) for text in texts_words]

    doc_topics = {}
    for i, bow in enumerate(corpus):
        topics = lda_model.get_document_topics(bow)
        sorted_topics = sorted(topics, key=lambda x: x[1], reverse=True)
        top_topic = sorted_topics[0][0] if sorted_topics else -1
        doc_topics[str(i)] = top_topic

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc_topics, f, ensure_ascii=False, indent=2)
    print(f"✅ تم حفظ توبيكات الوثائق في: {output_path}")
# --- استخراج المواضيع من نص واحد ---
def get_text_topics(text, top_n=3):
    if lda_model is None or dictionary is None:
        return []
    bow = dictionary.doc2bow(text.split())
    topics = lda_model.get_document_topics(bow)
    topics_sorted = sorted(topics, key=lambda x: x[1], reverse=True)[:top_n]

    topic_words = []
    for topic_id, score in topics_sorted:
        terms = lda_model.show_topic(topic_id, topn=3)
        words = [w for w, _ in terms]
        topic_words.append({
            "topic_id": topic_id,
            "score": round(score, 3),
            "words": words
        })
    return topic_words

# --- تحميل موديلات محفوظة إذا وجدت ---
def load_lda_models(model_path="models/lda_model", dict_path="models/lda_dict"):
    global lda_model, dictionary
    if os.path.exists(model_path) and os.path.exists(dict_path):
        lda_model = models.LdaModel.load(model_path)
        dictionary = corpora.Dictionary.load(dict_path)
        print("📦 LDA model loaded from disk")
        return True
    return False

# --- حفظ الموديلات للتخزين المؤقت ---
def save_lda_models(model_path="models/lda_model", dict_path="models/lda_dict"):
    if lda_model is not None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        lda_model.save(model_path)
        dictionary.save(dict_path)
