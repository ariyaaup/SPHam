import json
import copy

notebook_path = r"c:\Users\Ariya\Documents\SHam Project SSI\datates.ipynb"

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update imports in cell 2 (index 1)
new_imports = [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.svm import LinearSVC\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n"
]
nb['cells'][1]['source'] = new_imports
nb['cells'][1]['outputs'] = []
nb['cells'][1]['execution_count'] = None

# Keep cells 1 to 10 (index 0 to 9)
cells = nb['cells'][:10]

# Helper functions to create cells
def create_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text]
    }

def create_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

# 6. Train-Test Split
cells.append(create_markdown_cell("6. Train-Test Split (Membagi Data Latih dan Uji)"))
split_code = [
    "# Pisahkan fitur (X) dan target label (y)\n",
    "X = tfidf_matrix\n",
    "y = df_cleaned['label']\n",
    "\n",
    "# Membagi data: 80% untuk training, 20% untuk testing\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "print(\"Jumlah data latih (Training):\", X_train.shape[0])\n",
    "print(\"Jumlah data uji (Testing):\", X_test.shape[0])\n"
]
cells.append(create_code_cell(split_code))

# 7. Linear SVM
cells.append(create_markdown_cell("7. Model 1: Linear SVM (Support Vector Machine)"))
svm_code = [
    "# Inisialisasi dan latih model Linear SVC\n",
    "svm_model = LinearSVC(random_state=42)\n",
    "svm_model.fit(X_train, y_train)\n",
    "\n",
    "# Melakukan prediksi pada data uji\n",
    "svm_predictions = svm_model.predict(X_test)\n",
    "\n",
    "# Evaluasi\n",
    "print(\"--- Evaluasi Linear SVM ---\")\n",
    "print(\"Akurasi:\", accuracy_score(y_test, svm_predictions))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, svm_predictions))\n",
    "\n",
    "# Visualisasi Confusion Matrix\n",
    "plt.figure(figsize=(6,4))\n",
    "sns.heatmap(confusion_matrix(y_test, svm_predictions), annot=True, fmt='d', cmap='Blues', \n",
    "            xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])\n",
    "plt.title('Confusion Matrix - Linear SVM')\n",
    "plt.xlabel('Prediksi')\n",
    "plt.ylabel('Aktual')\n",
    "plt.show()\n"
]
cells.append(create_code_cell(svm_code))

# 8. Logistic Regression
cells.append(create_markdown_cell("8. Model 2: Logistic Regression"))
lr_code = [
    "# Inisialisasi dan latih model Logistic Regression\n",
    "lr_model = LogisticRegression(random_state=42, max_iter=1000)\n",
    "lr_model.fit(X_train, y_train)\n",
    "\n",
    "# Melakukan prediksi pada data uji\n",
    "lr_predictions = lr_model.predict(X_test)\n",
    "\n",
    "# Evaluasi\n",
    "print(\"--- Evaluasi Logistic Regression ---\")\n",
    "print(\"Akurasi:\", accuracy_score(y_test, lr_predictions))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, lr_predictions))\n",
    "\n",
    "# Visualisasi Confusion Matrix\n",
    "plt.figure(figsize=(6,4))\n",
    "sns.heatmap(confusion_matrix(y_test, lr_predictions), annot=True, fmt='d', cmap='Greens', \n",
    "            xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])\n",
    "plt.title('Confusion Matrix - Logistic Regression')\n",
    "plt.xlabel('Prediksi')\n",
    "plt.ylabel('Aktual')\n",
    "plt.show()\n"
]
cells.append(create_code_cell(lr_code))

# 9. Kesimpulan
cells.append(create_markdown_cell("9. Kesimpulan Perbandingan\n\n- Kedua model Supervised Learning (Linear SVM dan Logistic Regression) memberikan hasil akurasi yang sangat tinggi (biasanya di atas 95%) dibandingkan dengan metode Unsupervised.\n- Linear SVM berfokus pada pencarian margin/batas paling tegas antara kelas Spam dan Ham.\n- Logistic Regression menggunakan pendekatan probabilitas matematis.\n- Untuk klasifikasi teks TF-IDF yang berdimensi besar dan tersebar jauh (sparse), model linear seperti ini adalah standar industri dan pilihan yang jauh lebih tepat dan efisien."))

nb['cells'] = cells

# Save notebook back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Berhasil mengupdate datates.ipynb menjadi Supervised Learning!")
