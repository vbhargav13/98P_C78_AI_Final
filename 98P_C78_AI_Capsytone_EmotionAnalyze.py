"""
Emotion Analyzer AI Project 

This script builds an AI Emotion Classifier:
- Loads dataset (CSV path input).
- Explores and visualizes data.
- Splits & vectorizes data.
- Trains and evaluates model.
- Lets user type sentences to predict emotion (console-based).

==========================================================

✅ PRE-REQUISITE LIBRARIES:

- pandas            → For data handling (CSV files, DataFrames)
- scikit-learn      → For AI model: LogisticRegression + text vectorizer
- matplotlib        → For visualization (bar charts)

👉 Install them using the command below (same on Mac + Windows):

py -m pip install pandas scikit-learn matplotlib

==========================================================

🖥️ INSTRUCTIONS FOR RUNNING ON VS CODE:

1️⃣ Open VS Code.
2️⃣ Create a new Python file (e.g., emotion_analyzer.py) and paste this code.
3️⃣ Make sure your Python interpreter is selected in VS Code.
4️⃣ Save your CSV file with two columns:
    - Column 1: 'Text'
    - Column 2: 'Emotion'
   Example:  
   | Text                  | Emotion |
   |-----------------------|---------|
   | I am so happy today!  | happy   |
5️⃣ Run the code by opening the terminal (Ctrl + `) and typing:

python emotion_analyzer.py

6️⃣ When prompted:
👉 "Please enter the path to your CSV file:"
➡️ Type the full path to your CSV (e.g., C:/Users/YourName/Downloads/emotions.csv or /Users/YourName/Downloads/emotions.csv)

==========================================================

🍏 INSTRUCTIONS FOR RUNNING ON MACOS:

1️⃣ Install Python 3 from https://www.python.org if not installed.
2️⃣ Open VS Code or Terminal.
3️⃣ Follow the same steps as above to paste the code & save the file.
4️⃣ To install libraries, use:

python3 -m pip install pandas scikit-learn matplotlib

5️⃣ Run the script:

python3 emotion_analyzer.py

6️⃣ Type the path to your CSV when prompted (e.g., /Users/YourName/Downloads/emotions.csv).

==========================================================

🛠️ TROUBLESHOOTING (MAC & WINDOWS):

✅ ERROR: "ModuleNotFoundError: No module named 'pandas'"
➡️ Fix: Run → py -m pip install pandas

✅ ERROR: "ModuleNotFoundError: No module named 'sklearn'"
➡️ Fix: Run → py -m pip install scikit-learn

✅ ERROR: "'utf-8' codec can't decode byte..."
➡️ Fix: The code automatically retries with cp1252 encoding.
➡️ If error persists, ensure your CSV is saved as 'CSV UTF-8' from Excel.

✅ VS CODE: Python not found
➡️ Ensure Python is installed & VS Code interpreter is selected.

✅ MACOS: Permission issues with matplotlib
➡️ If charts don’t show, try running from Terminal instead of VS Code.

==========================================================
"""

# ===== L13: PROJECT SETUP - Import Required Libraries =====

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import time
import tkinter as tk          # (L21) For GUI
from tkinter import messagebox  # (L21) For showing prediction results


# ===== L14: Dataset Loader (CSV Handling + Validation) =====
def load_dataset():
    """
    (L14) Load the dataset by asking the user to type the path.
    - Handles encoding errors.
    - Validates the presence of 'Text' and 'Emotion' columns.
    """
    file_path = input("👉 Please enter the path to your CSV file: ")
    try:
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            print("⚠️ UTF-8 failed. Trying cp1252 encoding...")
            df = pd.read_csv(file_path, encoding='cp1252')

        df.columns = df.columns.str.strip()
        print("\n✅ Dataset loaded! First 5 rows:")
        print(df.head())

        if 'Text' not in df.columns or 'Emotion' not in df.columns:
            print(f"\n❌ ERROR: Required columns 'Text' and 'Emotion' not found. Found: {df.columns.tolist()}")
            return None

        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None


# ===== L15: Dataset Exploration =====
def explore_dataset(df):
    """
    (L15) Show dataset overview: columns, shape, emotion counts, avg text length.
    """
    print("\n📊 Dataset Info:")
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    print("Emotions:", df['Emotion'].unique())
    print("\nEmotion counts:")
    print(df['Emotion'].value_counts())

    df['TextLength'] = df['Text'].apply(len)
    avg_length = df.groupby('Emotion')['TextLength'].mean()
    print("\n✏️ Average text length per emotion:")
    print(avg_length)

    plot_emotion_distribution(df)


# ===== L16: Visualization =====
def plot_emotion_distribution(df):
    """
    (L16) Plot a bar chart showing the distribution of emotions.
    """
    df['Emotion'].value_counts().plot(kind='bar')
    plt.title("Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.show()


# ===== L17: Split Data =====
def split_data(df):
    """
    (L17) Split the dataset into training and testing sets.
    """
    X = df['Text']
    y = df['Emotion']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n✅ Data split: {len(X_train)} train, {len(X_test)} test samples.")
    return X_train, X_test, y_train, y_test


# ===== L18: Vectorization =====
def vectorize_text(X_train, X_test):
    """
    (L18) Convert text into numeric form using TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"✅ Text vectorized. Shape: {X_train_vec.shape}")
    return X_train_vec, X_test_vec, vectorizer


# ===== L19: Train Model =====
def train_model(X_train_vec, y_train):
    """
    (L19) Train LogisticRegression model.
    """
    model = LogisticRegression()
    start = time.time()
    model.fit(X_train_vec, y_train)
    print(f"✅ Model trained in {time.time() - start:.2f} seconds.")
    return model


# ===== L20: Evaluate Model =====
def evaluate_model(model, X_test_vec, y_test):
    """
    (L20) Evaluate the model using accuracy and confusion matrix.
    """
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔎 Accuracy: {acc:.2f}")
    print("Confusion Matrix:\n", cm)


# ===== L21 & L22: Tkinter GUI with Emojis =====
def create_predictor_gui_with_emojis(model, vectorizer):
    """
    (L21 & L22) Create a Tkinter GUI:
    - Users type a sentence and click 'Predict'.
    - The predicted emotion + emoji is displayed.
    """
    # (L22) Define emojis for each emotion
    emoji_mapping = {
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'surprise': '😲',
        'fear': '😨',
        'love': '❤️',
        'neutral': '😐',
    }

    def predict_emotion():
        user_text = entry.get()
        if user_text.strip() == '':
            messagebox.showwarning("Input Error", "⚠️ Please enter some text.")
            return
        vec = vectorizer.transform([user_text])
        prediction = model.predict(vec)[0]
        emoji = emoji_mapping.get(prediction.lower(), '🙂')  # Default to smile if not found
        result_var.set(f"Emotion: {prediction} {emoji}")

    # Set up Tkinter window
    gui = tk.Tk()
    gui.title("Emotion Predictor (with Emojis)")

    tk.Label(gui, text="Enter your sentence:", font=('Arial', 12)).pack(pady=10)
    entry = tk.Entry(gui, width=50, font=('Arial', 12))
    entry.pack(pady=5)

    predict_button = tk.Button(gui, text="Predict Emotion", command=predict_emotion, font=('Arial', 12))
    predict_button.pack(pady=10)

    result_var = tk.StringVar()
    result_label = tk.Label(gui, textvariable=result_var, font=('Arial', 14, 'bold'))
    result_label.pack(pady=20)

    gui.mainloop()


# ===== L23: Save Model =====
def save_model(model, vectorizer):
    """
    (L23) Save model + vectorizer for future use.
    """
    joblib.dump((model, vectorizer), 'emotion_model.pkl')
    print("✅ Model & vectorizer saved as 'emotion_model.pkl'.")


# ===== MAIN PROGRAM =====
if __name__ == "__main__":
    # (L14) Load dataset
    df = load_dataset()
    if df is not None:
        # (L15 & L16) Explore + visualize
        explore_dataset(df)

        # (L17) Split data
        X_train, X_test, y_train, y_test = split_data(df)

        # (L18) Vectorize
        X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

        # (L19) Train model
        model = train_model(X_train_vec, y_train)

        # (L20) Evaluate
        evaluate_model(model, X_test_vec, y_test)

        # (L23) Save
        save_model(model, vectorizer)

        # (L21 & L22) Launch GUI with emojis
        create_predictor_gui_with_emojis(model, vectorizer)
