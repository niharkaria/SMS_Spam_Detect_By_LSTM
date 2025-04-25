# 📩 SMS Spam Detection using Word2Vec and LSTM

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![NLP](https://img.shields.io/badge/NLP-Text_Classification-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A deep learning-based project that classifies SMS messages as **spam** or **ham (not spam)** using custom-trained **Word2Vec embeddings** and a **Bidirectional LSTM** neural network.

---

## 📌 Project Overview

- **Dataset:** SMS Spam Collection (from UCI Machine Learning Repository)
- **Task:** Binary classification of text messages
- **Tech Stack:** Python, TensorFlow, Gensim, NLTK
- **Model:** Word2Vec + Bidirectional LSTM
- **Goal:** Build a robust spam classifier using deep learning and NLP techniques

---

## 🗂️ Dataset

📥 **Download Dataset:**
You can download the SMS Spam Collection dataset directly from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

Or use the file included here: `SMSSpamCollection`

The file contains over 5,000 SMS messages labeled as `ham` (non-spam) or `spam`.

---

## 🔧 Workflow

1. **Data Loading & Cleaning**
   - Lowercasing, number & punctuation removal
   - Whitespace normalization

2. **Tokenization & Embedding**
   - Tokenize using NLTK
   - Train custom Word2Vec embeddings (Gensim)

3. **Text to Sequence Conversion**
   - Pad sequences to fixed length
   - Create embedding matrix

4. **LSTM Model Building**
   - Embedding Layer (from Word2Vec)
   - Bidirectional LSTM
   - Dropout and Dense output

5. **Evaluation**
   - Classification report
   - Confusion matrix
   - Accuracy

---

## 📊 Results

### ✅ Accuracy: `99.37%`

### 🧾 Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Ham   | 0.99      | 1.00   | 1.00     | 966     |
| Spam  | 1.00      | 0.95   | 0.98     | 149     |

### 🔁 Confusion Matrix:

|              | Pred_Ham | Pred_Spam |
|--------------|----------|-----------|
| **Ham**      | 966      | 0         |
| **Spam**     | 7        | 142       |

---

## 🧠 Skills Applied

- Text Preprocessing
- Word Embeddings (Word2Vec)
- LSTM & Sequence Modeling
- Model Evaluation
- Binary Text Classification

---

## 🚀 Future Improvements

- Use pre-trained embeddings like GloVe or FastText
- Try Transformer-based models or Attention layers
- Deploy using Streamlit or Flask
- Integrate real-time SMS input classification

---

## 📁 Files Included

| File | Description |
|------|-------------|
| `SpamDetection_LSTM.ipynb` | Full project notebook |
| `SMSSpamCollection` | Raw dataset (tab-separated) |
| `README.md` | Project overview and documentation |

---

## 📜 License

This project is intended for educational purposes.

---

## 🤝 Connect

Feel free to connect on [LinkedIn](https://www.linkedin.com/) or explore other projects on [GitHub](https://github.com/)!