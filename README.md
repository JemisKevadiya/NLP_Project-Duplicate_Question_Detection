# 🧠 Duplicate Question Detection using NLP

## 📌 Overview

Online platforms like Quora often contain multiple questions that are semantically similar but phrased differently. This leads to duplicate content and inefficient information retrieval.

This project builds an **end-to-end NLP system** to automatically detect whether two questions are duplicates based on their semantic meaning.

---

## 🎯 Problem Statement

Given a pair of questions, predict whether they have the same meaning.

* **Input:** Question 1 & Question 2
* **Output:** Duplicate (1) / Not Duplicate (0)

---

## 📊 Dataset

* **Quora Question Pairs Dataset**
* Contains question pairs with labels indicating duplicate (1) or not (0)
* Dataset Link - https://www.kaggle.com/c/quora-question-pairs

---

## ⚙️ Tech Stack

* Python 
* Pandas, NumPy
* Scikit-learn
* NLTK
* FuzzyWuzzy
* Matplotlib, Seaborn
* Sentence Transformers (BERT)

---

## 🔄 Project Pipeline

### 1. Data Preprocessing

* Lowercasing
* Removing HTML tags, URLs, punctuation
* Expanding contractions
* Stopword removal
* Lemmatization

### 2. Basic Feature Engineering

* Question length
* Word count
* Common words
* Word share ratio

### 3. Advance Feature Engineering

* Token Features
* Leangth Based Features
* Fuzzy Feature
* TF-IDF vectors
* Cosine similarity
* BERT embeddings (semantic similarity)

### 4. Model Building

* Random Forest
* XGBoost

### 5. Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 📈 Key Insights

* Dataset is imbalanced (more non-duplicate questions)
* Duplicate questions share more semantic similarity than lexical similarity
* Traditional methods (TF-IDF) struggle with synonyms (e.g., "AI" vs "Artificial Intelligence")
* Transformer models (BERT) significantly improve performance

---

## 🏗️ Project Structure

```
project/
│
├── notebooks/
│   ├── data/
│   │   ├── train.csv
│   ├── Advance_Feature_Engineering+Model_Training.ipynb
│   ├── Basic_Feature_Engineering.ipynb
│   ├── EDA+Feature_Extraction.ipynb
│
├── app.py
├── model.pkl
├── w2v_model.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/your-username/project-name.git
cd project-name
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model

```bash
python src/train.py
```

### 4. Run App (Optional)

```bash
streamlit run app.py
```

---

## 🧪 Example

**Input:**

* Question 1: What is the best way to learn python?
* Question 2: How can i start Learning python?

**Output:**

* Duplicate ✅

<img width="944" height="557" alt="Screenshot 2026-04-08 155637" src="https://github.com/user-attachments/assets/5920ae60-cbbe-44db-a764-b93d471c2a72" />



## 👨‍💻 Author

* Jemis Kevadiya

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
