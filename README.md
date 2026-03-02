# Fake-News-Detection-System
# 📰 Fake News Detection System

## 📌 Overview

This project implements a Machine Learning-based Fake News Detection System that classifies news headlines as **True** or **False** and predicts the probability of the predicted class.

The system follows a structured data mining pipeline:

- Data Collection  
- Data Preprocessing  
- Feature Extraction  
- Model Training & Hyperparameter Tuning  
- Ensemble Learning (Voting Classifier)  
- Model Evaluation  

The primary evaluation metric used for model selection is **F1-score**, ensuring a balanced trade-off between precision and recall.

---

# 📂 Dataset

Dataset Used:  
**Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection**  
Author: William Yang Wang  

The dataset contains:

- 12,836 short political statements  
- Labels for truthfulness (6 fine-grained classes)  
- Metadata such as speaker, party, state, subject, context  

Original 6 labels:

- pants-fire  
- false  
- barely-true  
- half-true  
- mostly-true  
- true  

---

# 🔄 Data Preprocessing

## 1️⃣ Label Mapping (Multi-class → Binary)

| Original Label | Mapped Label |
|---------------|-------------|
| pants-fire   | False |
| false        | False |
| barely-true  | False |
| half-true    | True |
| mostly-true  | True |
| true         | True |

Only the **news headline text** was used as model input.

## 2️⃣ Output Files

After preprocessing:

- train.csv  
- valid.csv  
- test.csv  

Each file contains:
- Headline  
- Binary Label (True/False)  

---

# 🧠 Feature Extraction

Two NLP feature extraction techniques were implemented:

## 🔹 CountVectorizer

- Stopword removal  
- Tokenization (space + punctuation)  
- Sparse Bag-of-Words matrix  
- N-gram support  
- Frequency-based representation  

## 🔹 TfidfVectorizer

- Stopword removal  
- Tokenization  
- TF-IDF weighting  
- N-gram support  
- Term importance normalization  

---

# 🤖 Models Implemented

The following supervised learning models were trained:

- Logistic Regression  
- Naïve Bayes  
- Support Vector Machine (SVM)  
- Random Forest Classifier  

All models were:

- Hyperparameter tuned using GridSearchCV  
- 5-fold cross validation  
- Optimized based on F1-score  

---

# 🗳 Ensemble Model – Voting Classifier

A Soft Voting Classifier was implemented using:

- Logistic Regression  
- Naïve Bayes  
- SVM  
- Random Forest  

### How It Works:

1. Each base model predicts class probabilities.  
2. Probabilities are averaged.  
3. Final label is selected based on highest averaged probability.  

---

# 📊 Results

## 🔹 Using CountVectorizer

| Model | Accuracy | F1 Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| Logistic Regression | 56.35% | 72.08% | 56.35% | 100% |
| Naïve Bayes | 62.03% | 73.26% | 60.73% | 92.29% |
| SVM | 56.66% | 72.11% | 56.57% | 99.43% |
| Random Forest | 56.51% | 72.15% | 56.44% | 100% |
| Voting Classifier | 60.45% | 72.93% | 59.33% | 94.53% |

Observation:
- Best F1-score: Naïve Bayes (73.26%)  
- High recall across models (>92%)  
- Precision limited due to small feature space  

---

## 🔹 Using TfidfVectorizer

| Model | Accuracy | F1 Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| Logistic Regression | 63.14% | 71.43% | 63.40% | 81.79% |
| Naïve Bayes | 60.53% | 73.29% | 59.24% | 96.07% |
| SVM | 60.06% | 72.01% | 59.50% | 91.17% |
| Random Forest | 57.22% | 72.48% | 56.84% | 100% |
| Voting Classifier | 62.27% | 72.36% | 61.61% | 87.67% |

Observation:
- TF-IDF improves accuracy and precision  
- Slight reduction in recall compared to CountVectorizer  
- Best F1-score again achieved by Naïve Bayes (73.29%)  

---

# 📌 Key Findings

- TF-IDF outperforms CountVectorizer overall.  
- Naïve Bayes performs best for sparse text classification.  
- Ensemble model improves stability but does not significantly outperform Naïve Bayes.  
- High recall suggests strong ability to detect fake news.  

---

# 📁 Repository Structure

```
.
├── codes
│   ├── training_scripts.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── notebook.ipynb
│
├── datasets
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
│
├── figures
│   └── model_performance_plots
│
├── images
│   └── readme_assets
│
└── models
    ├── naive_bayes.pkl
    └── voting_classifier.pkl
```

---

# ⚙️ Installation & Usage

```bash
# Clone repository
git clone https://github.com/yourusername/fake-news-detection.git

# Install dependencies
pip install -r requirements.txt

# Run training
python training_scripts.py
```

---

# 🛠 Tech Stack

- Python  
- scikit-learn  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Jupyter Notebook  

---

# 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score (Primary Metric)  

F1-score was chosen because:
- Dataset is imbalanced.  
- It balances precision and recall.  

---

# 🚀 Future Improvements

- Implement Transformer models (BERT, RoBERTa)  
- Use full article text instead of headlines  
- Add Explainable AI (SHAP / LIME)  
- Deploy as Flask / Streamlit web app  
- Perform class imbalance handling (SMOTE)  

---

# 📜 License

This project is for academic and research purposes.
