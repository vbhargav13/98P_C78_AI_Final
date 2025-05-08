"""
Emotion Analyzer AI Project

💡 This script builds an AI Emotion Classifier using an ensemble of:
 • Multinomial Naive Bayes (Lesson 19)
 • Logistic Regression (Lesson 19)

Pipeline:
 1️⃣ Load & clean CSV (Lesson 14)
 2️⃣ Explore & visualize (Lesson 15-16)
 3️⃣ Split & vectorize (Lesson 17-18)
 4️⃣ Train NB & LR (Lesson 19)
 5️⃣ Evaluate NB, LR & Ensemble (Lesson 20)
 6️⃣ GUI: predict with emojis + confidence (Lesson 21-22)
 7️⃣ Save & reload model for reuse (Lesson 23)

──────────────────────────────────────────────
📦 PRE-REQUISITE LIBRARIES:
- pandas
- scikit-learn
- matplotlib
- joblib
- tkinter

▶️ INSTALL via terminal:
Windows:
    py -m pip install pandas scikit-learn matplotlib joblib

macOS:
    python3 -m pip install pandas scikit-learn matplotlib joblib

──────────────────────────────────────────────
🖥️ VS CODE RUNNING INSTRUCTIONS (Windows + macOS):

1️⃣ Open VS Code.
2️⃣ Create a folder (e.g., Emotion_Analyzer).
3️⃣ Place your CSV dataset (must have 'Text' and 'Emotion' columns) in the folder.
4️⃣ Save this script in the folder as emotion_analyzer.py.
5️⃣ Open the terminal (View > Terminal).
6️⃣ Install dependencies (see above).
7️⃣ Run script:
    Windows: py emotion_analyzer.py
    macOS: python3 emotion_analyzer.py

──────────────────────────────────────────────
⚠️ TROUBLESHOOTING:

Windows:
- UnicodeDecodeError → CSV might be non-UTF-8. The script auto-tries cp1252.
- ImportError → Check you installed ALL libraries (pandas, scikit-learn, matplotlib, joblib).

macOS:
- TclError (tkinter) → Run: brew install tcl-tk, then ensure Python is linked properly.
- ImportError → Use python3 -m pip install ...

──────────────────────────────────────────────
"""

# ─────────────── LESSON 13: PROJECT SETUP ───────────────────────────────
import pandas as pd                      # (Lesson 14) For reading and processing CSV files
import matplotlib.pyplot as plt          # (Lesson 16) For plotting emotion distribution
from sklearn.model_selection import train_test_split  # (Lesson 17) To split dataset into train/test
from sklearn.feature_extraction.text import TfidfVectorizer  # (Lesson 18) To convert text into vectors
from sklearn.naive_bayes import MultinomialNB            # (Lesson 19) Classifier 1: Naive Bayes
from sklearn.linear_model import LogisticRegression      # (Lesson 19) Classifier 2: Logistic Regression
from sklearn.metrics import accuracy_score, confusion_matrix  # (Lesson 20) For model evaluation
import joblib                            # (Lesson 23) For saving/reloading models

import time                              # (General) Measure training time

import tkinter as tk                     # (Lesson 21-22) For GUI app
from tkinter import messagebox           # (Lesson 21-22) For GUI alerts

# ─────────────── LESSON 14: DATASET LOADING & CLEANING ───────────────
def load_dataset():
    path = input("👉 CSV path: ").strip()  # Prompt user to enter path to dataset CSV file
    path = ''.join(ch for ch in path if ch.isprintable())  # Clean path input
    try:
        try:
            df = pd.read_csv(path)  # Try reading with UTF-8
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='cp1252')  # Fallback to cp1252
        df['Text'] = df['Text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)  # Lowercase & clean text
        df.columns = df.columns.str.strip()
        if not {'Text','Emotion'}.issubset(df.columns):
            print("❌ Need columns 'Text' & 'Emotion'.")
            return None
        print("✅ Loaded:", df.shape, "sample:\n", df.head(2))
        return df
    except Exception as e:
        print("❌ Load error:", e)
        return None

# ─────────────── LESSON 15-16: EXPLORATION & VISUALIZATION ───────────────
def explore_dataset(df):
    print("\n📊 Counts:\n", df['Emotion'].value_counts())
    df['Len'] = df['Text'].str.len()
    print("\n✏️ Avg length:\n", df.groupby('Emotion')['Len'].mean())
    df['Emotion'].value_counts().plot(kind='bar')
    plt.title("Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.show()

# ─────────────── LESSON 17-18: SPLIT & VECTORIZE ───────────────
def split_vectorize(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['Text'], df['Emotion'], test_size=0.2, random_state=42
    )
    vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_df=0.9, min_df=2)
    X_train_vec = vect.fit_transform(X_train)
    X_test_vec  = vect.transform(X_test)
    print(f"\n✅ Vectorized: {X_train_vec.shape}")
    return X_train_vec, X_test_vec, y_train, y_test, vect

# ─────────────── LESSON 19: TRAIN MULTIPLE MODELS ───────────────
def train_models(X_train_vec, y_train):
    nb = MultinomialNB(alpha=0.7)
    t0 = time.time()
    nb.fit(X_train_vec, y_train)
    print(f"✅ NB trained in {time.time()-t0:.2f}s")
    lr = LogisticRegression(max_iter=200)
    t1 = time.time()
    lr.fit(X_train_vec, y_train)
    print(f"✅ LR trained in {time.time()-t1:.2f}s")
    return nb, lr

# ─────────────── LESSON 20: EVALUATE & COMPARE ───────────────
def evaluate(nb, lr, X_test_vec, y_test):
    def report(name, model):
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        print(f"\n🔎 {name} Acc: {acc:.2f}")
        print(confusion_matrix(y_test, preds))
    report("Naive Bayes", nb)
    report("Logistic Reg", lr)
    proba_nb = nb.predict_proba(X_test_vec)
    proba_lr = lr.predict_proba(X_test_vec)
    proba_avg= (proba_nb + proba_lr)/2
    y_ens = nb.classes_[proba_avg.argmax(axis=1)]
    acc_e = accuracy_score(y_test, y_ens)
    print(f"\n🔎 Ensemble Acc: {acc_e:.2f}")
    print(confusion_matrix(y_test, y_ens))

# ─────────────── LESSON 21-22: GUI PREDICTOR ───────────────
def create_gui(nb, lr, vect):
    emoji = {'happy':'😊','sad':'😢','angry':'😠','surprise':'😲',
             'fear':'😨','love':'❤️','neutral':'😐'}
    def predict():
        txt = entry.get().strip().lower()
        if not txt:
            messagebox.showwarning("Input Error","Enter some text.")
            return
        vec = vect.transform([txt])
        p_nb = nb.predict_proba(vec)[0]
        p_lr = lr.predict_proba(vec)[0]
        p_avg = (p_nb + p_lr)/2
        idx = p_avg.argmax()
        label = nb.classes_[idx]
        conf = p_avg[idx]
        res.set(f"Emotion: {label} {emoji.get(label,'🙂')}  (Conf: {conf:.2f})")
    root = tk.Tk()
    root.title("Ensemble Emotion Predictor")
    tk.Label(root,text="Enter sentence:").pack(pady=5)
    entry = tk.Entry(root, width=50)
    entry.pack(pady=5)
    tk.Button(root,text="Predict",command=predict).pack(pady=5)
    res = tk.StringVar()
    tk.Label(root,textvariable=res,font=('Arial',14)).pack(pady=10)
    root.mainloop()

# ─────────────── LESSON 23: SAVE & RELOAD ───────────────
def save_model(nb, lr, vect):
    # Save models and vectorizer for future use
    joblib.dump({'nb': nb, 'lr': lr, 'vect': vect}, 'emotion_model.pkl')
    print("💾 Model & vectorizer saved as 'emotion_model.pkl' ✅")

def load_model():
    # Reload saved model & vectorizer
    print("🔄 Loading saved model...")
    saved = joblib.load('emotion_model.pkl')
    print("✅ Model loaded.")
    return saved['nb'], saved['lr'], saved['vect']

# ─────────────── MAIN WORKFLOW ───────────────
if __name__=="__main__":
    df = load_dataset()  # (Lesson 14)
    if df is not None:
        explore_dataset(df)  # (Lesson 15-16)
        Xtr, Xte, ytr, yte, vect = split_vectorize(df)  # (Lesson 17-18)
        nb_model, lr_model = train_models(Xtr, ytr)    # (Lesson 19)
        evaluate(nb_model, lr_model, Xte, yte)         # (Lesson 20)
        save_model(nb_model, lr_model, vect)           # (Lesson 23: Save after training)
        # Uncomment below to test reload functionality
        # nb_model, lr_model, vect = load_model()      # (Lesson 23: Load)
        create_gui(nb_model, lr_model, vect)           # (Lesson 21-22 GUI with saved model)
