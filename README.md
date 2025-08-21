# Quora Duplicate Question Pair Detection

![Quora Logo](https://upload.wikimedia.org/wikipedia/commons/9/91/Quora_logo_2015.svg)

## 🚀 Problem Statement
Given a pair of questions from Quora, predict whether they are duplicates (i.e., essentially the same question, possibly phrased differently).

## 💼 Business Motivation
- **User Experience:** Prevent duplicate questions, improve answer quality, and reduce clutter.
- **Revenue:** Enable more relevant ad targeting and content recommendations.
- **Knowledge Sharing:** Connect similar questions to a broader set of answers and insights.

## 📊 Dataset & Preprocessing
- **Source:** 400,000+ Quora question pairs, each labeled as duplicate (1) or not (0).
- **Imbalance:** ~63% non-duplicates, ~37% duplicates. No oversampling/undersampling to avoid overfitting and information loss.
- **Preprocessing:**
  - Lowercasing, punctuation removal, contraction expansion
  - Stopword removal, stemming (NLTK), HTML tag removal (BeautifulSoup)
  - Feature engineering (see below)

## 🛠️ Feature Engineering
- **Basic Features:** Question lengths, word counts, common word counts, word share, etc.
- **Advanced Features:**
  - FuzzyWuzzy ratios (fuzz_ratio, fuzz_partial_ratio, token_set_ratio, token_sort_ratio)
  - Longest substring ratio
  - Frequency features (how often each question appears as qid1 or qid2)
  - Word embeddings (spaCy en_core_web_lg + TF-IDF weighting)
- **Why?** These features capture both surface-level and semantic similarities, crucial for detecting paraphrased duplicates.

## 🤖 Modeling Approach
- **Baseline:** Random model for lower bound on log-loss and accuracy.
- **Logistic Regression (SGD):** Fast, interpretable, good for large/sparse data. Calibrated for better probability estimates. Good for non-duplicates, weaker recall for duplicates.
- **Linear SVM:** Tried as an alternative, but underfit the data and failed to distinguish classes due to feature limitations and class imbalance.
- **XGBoost:**
  - **Why XGBoost?** Handles non-linear relationships, high-dimensional/sparse data, robust to feature interactions and missing values. Outperformed other models in log-loss, accuracy, and balanced precision/recall.
  - **Training:** Used both 100k and 300k samples, stratified train/test splits, SQLite for efficient chunked processing.

## 📈 Model Evaluation
- **Metrics:** Log-loss (main), accuracy, confusion matrix, precision, recall.
- **Results (XGBoost, 300k data):**
  - Test Log Loss: ~0.35 (very good)
  - Test Accuracy: ~82%
  - Precision/Recall: Balanced, with higher recall for non-duplicates but good performance for duplicates as well.
  - No overfitting: Small gap between train and test log-loss.

## 💡 Key Insights & Learnings
- Fuzzy features and word embeddings are critical for semantic similarity.
- Frequency and length-based features help capture question popularity and structure.
- No oversampling/undersampling; relied on strong features and stratified splits.
- XGBoost was chosen for its ability to model complex, non-linear relationships and its superior performance.
- The model is robust and production-ready, but further improvements (e.g., BERT embeddings, timestamp features) could enhance duplicate detection.

## 🛠️ Project Workflow
1. **Data Loading & EDA:** Explored and visualized data, engineered basic features.
2. **Preprocessing & Advanced Features:** Cleaned text, added fuzzy and embedding features.
3. **Model Training & Evaluation:** Compared random, logistic regression, SVM, and XGBoost. Used SQLite for efficient data handling.
4. **Deployment:** Built and deployed a Streamlit app for end users.

## 🌐 Deployment
- **Environment:** Python 3.11 (due to spaCy compatibility issues with 3.12+). Virtual environment for dependency management.
- **Web App:** Streamlit app for interactive duplicate question detection.
- **Hosting:** Deployed on Streamlit Community Cloud for public access.

## ⚠️ Challenges & Future Work
- **Large File Handling:** Extracted and used only essential metadata (e.g., feature columns) for deployment.
- **Semantic Understanding:** Some duplicate pairs still missed; future work could use transformer-based embeddings (BERT, etc.).
- **Timestamp Feature:** Would enable better temporal splits and trend analysis if available.

## 📦 How to Run
1. Clone the repo and upload all required files (model, feature columns, etc.).
2. Create and activate a Python 3.11 virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app_streamlit.py
   ```

## 📝 Notebooks Overview
- **Quora1.ipynb:** Data loading, basic feature engineering, EDA.
- **Quora2.ipynb:** Data preprocessing, advanced feature engineering.
- **Quora3.ipynb:** Feature analysis, word cloud, feature relationships.
- **Quora4.ipynb:** t-SNE for data visualization.
- **Quora5.ipynb:** Model training on engineered features.
- **Quora6.ipynb:** Baseline, logistic regression, SVM, XGBoost (1 lakh data).
- **Quora7.ipynb:** XGBoost on 350k data, final model selection.

## 📚 References
- [Quora Question Pairs Kaggle Competition](https://www.kaggle.com/c/quora-question-pairs)
- [spaCy Documentation](https://spacy.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Author:** Aditya Singh

**Contact:** [LinkedIn](https://linkedin.com/in/aditya-singh-2b319b299/) | [GitHub](https://github.com/AdiSinghCodes)

---

> This project demonstrates a full pipeline for duplicate question detection on Quora, from data exploration and feature engineering to model selection, evaluation, and deployment. XGBoost was chosen for its superior performance, and the final product is a robust, user-friendly web app ready for real-world use.
