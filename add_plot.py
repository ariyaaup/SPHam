import json

notebook_path = "datates.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_source = [
    "# 10. Perbandingan Performa Model dalam Satu Grafik\n",
    "import numpy as np\n",
    "from sklearn.metrics import precision_recall_fscore_support\n",
    "\n",
    "# Menghitung metrik untuk SVM\n",
    "svm_acc = accuracy_score(y_test, svm_predictions)\n",
    "svm_prec, svm_rec, svm_f1, _ = precision_recall_fscore_support(y_test, svm_predictions, pos_label='spam', average='binary')\n",
    "\n",
    "# Menghitung metrik untuk Logistic Regression\n",
    "lr_acc = accuracy_score(y_test, lr_predictions)\n",
    "lr_prec, lr_rec, lr_f1, _ = precision_recall_fscore_support(y_test, lr_predictions, pos_label='spam', average='binary')\n",
    "\n",
    "# Data untuk plot\n",
    "metrics = ['Accuracy', 'Precision (Spam)', 'Recall (Spam)', 'F1-Score (Spam)']\n",
    "svm_scores = [svm_acc, svm_prec, svm_rec, svm_f1]\n",
    "lr_scores = [lr_acc, lr_prec, lr_rec, lr_f1]\n",
    "\n",
    "x = np.arange(len(metrics))\n",
    "width = 0.35\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "rects1 = ax.bar(x - width/2, svm_scores, width, label='Linear SVM', color='skyblue')\n",
    "rects2 = ax.bar(x + width/2, lr_scores, width, label='Logistic Regression', color='lightgreen')\n",
    "\n",
    "# Menambahkan teks dan label\n",
    "ax.set_ylabel('Scores')\n",
    "ax.set_title('Perbandingan Performa: Linear SVM vs Logistic Regression')\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(metrics)\n",
    "ax.set_ylim(0, 1.15)\n",
    "ax.legend(loc='lower right')\n",
    "\n",
    "# Menambahkan nilai di atas bar\n",
    "ax.bar_label(rects1, fmt='%.3f', padding=3)\n",
    "ax.bar_label(rects2, fmt='%.3f', padding=3)\n",
    "\n",
    "fig.tight_layout()\n",
    "plt.show()\n"
]

# Insert before the last cell (which is the conclusion markdown)
new_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': cell_source
}

nb['cells'].insert(-1, new_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Plot cell added!")
