# Information Retrieval System 2024–2025

نظام استرجاع معلومات متعدد التمثيلات يعتمد على بنية معمارية موجهة بالخدمات (SOA). يدعم البحث باستخدام نماذج تقليدية وحديثة مثل TF-IDF وBERT وHybrid، مع إمكانية تحسين الاستعلامات وتخزين التمثيلات الدلالية باستخدام FAISS.

---

## sourec code in branch called code



## 📂 محتوى المشروع

- `services/` – جميع الخدمات الرئيسية مثل معالجة الوثائق، الفهرسة، التحسين، التخزين الشعاعي، وتحسين الاستعلامات.
- `data/` – تمثيلات جاهزة وفهارس لكل مجموعة بيانات.
- `web_ui/` – واجهة المستخدم إن وجدت.
- `evaluation/` – سكربتات التقييم باستخدام pytrec_eval أو trectools.
- `main.py` – نقطة التشغيل الرئيسية للنظام.

---

## 🚀 طريقة التشغيل

### 1. تحميل البيانات

```python
from services.dataset_loader import preload_all_datasets
preload_all_datasets()
```

### 2. استعلام من خلال retrieval service

```python
from services.retrieval_service import retrieve_results

results = retrieve_results(
    query="example query",
    dataset="msmarco_train",       # أو beir_quora_test
    model="hybrid",                # tfidf | bert | hybrid
    use_vectorstore=True
)
```

---

## 📊 التقييم

تدعم المنصة تقييم النماذج باستخدام مقاييس:
- MAP
- MRR
- Precision@10
- Recall

```bash
python evaluation/evaluate_msmarco.py
python evaluation/evaluate_quora.py
```

---

## ✅ الميزات الإضافية

- Query Refinement
  - اقتراح استعلامات مشابهة
  - تتبع الاستعلامات الأكثر شيوعًا
  - استخراج مصطلحات مرتبطة
- Vector Store باستخدام FAISS لتسريع BERT
- Hybrid Scoring
- دعم أكثر من مجموعة بيانات (MSMARCO / Quora)

---

## 🧱 بنية النظام (SOA)

كل مكون في النظام عبارة عن خدمة منفصلة:
- `document_service` – معالجة وتحضير الوثائق.
- `indexing_service` – بناء تمثيلات TF-IDF وBERT والفهارس.
- `dataset_loader` – تحميل التمثيلات الجاهزة إلى الذاكرة.
- `query_searching` – تنفيذ البحث الفعلي.
- `retrieval_service` – واجهة موحدة للاستعلام.
- `vector_store` – تخزين BERT باستخدام FAISS.
- `query_refinement` – تحسين وتوسيع الاستعلام.

---
