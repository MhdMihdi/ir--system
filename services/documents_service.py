# ir_system/services/document_service.py

import re
import html
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import os
import pandas as pd
import ir_datasets
import string
from symspellpy import SymSpell, Verbosity
import pkg_resources

# إعداد أدوات المعالجة
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# إعداد SymSpell للتصحيح الدلالي
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


def semantic_spell_check(text):
    """
    تصحيح الأخطاء الإملائية مع مراعاة المعنى
    """
    words = text.split()
    corrected_words = []
    
    for word in words:
        # البحث عن التصحيحات المحتملة
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        
        if suggestions:
            # اختيار التصحيح الأقرب
            corrected_words.append(suggestions[0].term)
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)

def safe_text(text):
    """
    يحاول إصلاح الترميز الخاطئ في النصوص.
    """
    try:
        return text.encode('latin1').decode('utf-8')
    except:
        return text

def query_advanced_preprocess(text):
    """
    تنظيف متقدم للنصوص يشمل:
    - إزالة HTML والرموز
    - تحويل إلى أحرف صغيرة
    - تصحيح الأخطاء الإملائية دلالياً
    - إزالة الكلمات الشائعة والتكرارات
    - Stemming
    """
    text = html.unescape(text)
    text = ''.join(c for c in text if c.isprintable())
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # # تصحيح الأخطاء الإملائية
    # text = semantic_spell_check(text)
    
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

punctuation_to_remove = string.punctuation.replace("?", "")
punctuation_regex = re.compile(f"[{re.escape(punctuation_to_remove)}]")

def preprocess(text):
    # 1. تحويل إلى حروف صغيرة
    text = text.lower()
    
    # 2. إزالة علامات الترقيم (باستثناء "?")
    text = punctuation_regex.sub("", text)
    
    # 3. إزالة المسافات الزائدة
    text = re.sub(r"\s+", " ", text).strip()
    
    # 4. Tokenization
    tokens = text.split()
    
    # 5. إزالة الكلمات القصيرة جداً
    tokens = [token for token in tokens if len(token) >= 2]
    
    return tokens 

def advanced_preprocess(text):
    # Lowercase
    text = text.lower()
    
    # إزالة الروابط
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # إزالة الإيميلات
    text = re.sub(r'\S+@\S+', '', text)

    # إزالة علامات HTML
    text = re.sub(r'<.*?>', '', text)

    # إزالة الأرقام
    text = re.sub(r'\d+', '', text)

    # إزالة علامات الترقيم والرموز الخاصة
    text = re.sub(r'[^a-z\s]', ' ', text)

    # إزالة التكرار الزائد في الحروف
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # إزالة الفراغات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization + Stopword Removal + Stemming
    tokens = text.split()
    processed = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 2]

    return processed


def process_documents(limit=200000, dataset_name="msmarco-passage/train", output_file=None):
    dataset = ir_datasets.load(dataset_name)
    doc_iterator = dataset.docs_iter()
    docs_data = []
    count = 0

    while count < limit:
        try:
            doc = next(doc_iterator)
            text_clean = safe_text(doc.text)
            docs_data.append({
                "id": doc.doc_id,
                "text": text_clean
            })
            count += 1
            if count % 10000 == 0:
                print(f"✅ تم تحميل {count} وثيقة...")
        except Exception:
            continue

    df = pd.DataFrame(docs_data)
    print("\n🧹 بدء تنظيف النصوص...")
    df['clean_text'] = df['text'].apply(advanced_preprocess)

    # المسار التلقائي إذا ما انمرر
    if output_file is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "msmarco_train", "processed")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "processed_msmarco_train.json")
        