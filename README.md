# 🛍️ Sentiment-Based Product Recommendation System for Ebuss

This project builds a smart, sentiment-aware product recommendation system for **Ebuss**, a fast-growing e-commerce platform. By leveraging user reviews and ratings, the system identifies customer sentiments and recommends top products tailored to individual users. The solution uses NLP, Machine Learning, and Collaborative Filtering, and is deployed as a web application using Flask and Heroku.

---

## 📌 Problem Statement

Ebuss sells products across diverse categories such as household essentials, personal care, medicines, and more. To gain a competitive edge over e-commerce giants like Amazon and Flipkart, Ebuss needs to provide personalized, sentiment-based product recommendations to its users.

You, as a Senior ML Engineer, are tasked with building an end-to-end sentiment-based recommendation engine, integrating:
- Sentiment analysis of product reviews
- Recommendation engine (user-based/item-based)
- Integration of both models to improve top product suggestions
- A deployable Flask-based UI hosted on Heroku

---

## 🎯 Business Objective

- Understand customer preferences using their reviews and ratings.
- Build a product recommendation system filtered by review sentiment.
- Recommend top 5 personalized products to a user.
- Deploy the system with a web interface for real-time usage.

---

## 📂 Dataset

- **Total Reviews:** 30,000+
- **Products:** 200+
- **Users:** 20,000+
- **Source:** [Subset of Kaggle’s Amazon product review dataset]

**Key Features:**
- `reviews_username`
- `reviews_rating`
- `reviews_text`
- `product_name`

---

## 🧠 Project Workflow

### 1. Sentiment Analysis Pipeline
- Exploratory Data Analysis
- Data Cleaning & Preprocessing (tokenization, stopword removal, etc.)
- Feature Engineering (TF-IDF, Bag of Words)
- Models:
  - Logistic Regression ✅
  - Random Forest ✅
  - XGBoost ✅
  - Naive Bayes ✅
- Best model selected based on Accuracy, F1-Score, Confusion Matrix

### 2. Recommendation Engine
- Implemented both:
  - User-Based Collaborative Filtering
  - Item-Based Collaborative Filtering
- Selected the best model for Ebuss based on RMSE and relevancy

### 3. Combined Recommendation Logic
- Top 20 products are recommended to a user
- Filtered top 5 products using sentiment scores from the ML model

---

## 🛠️ Tech Stack

| Component       | Tools/Frameworks Used                       |
|----------------|----------------------------------------------|
| Language        | Python                                      |
| ML Libraries    | scikit-learn, XGBoost, NLTK, Pandas, NumPy  |
| NLP Tools       | TF-IDF, CountVectorizer                     |
| Web Framework   | Flask                                       |
| Deployment      | Heroku                                      |
| Frontend        | HTML, CSS (with `index.html`)               |
| Miscellaneous   | Pickle (model serialization)                |

---

## 📈 Evaluation Metrics

- Accuracy / Precision / Recall / F1-Score (Sentiment Classification)
- RMSE (Recommendation System)
- User satisfaction based on sentiment-filtered recommendations

---

## 🧪 How to Run

### 📦 Local Setup

```bash
# Clone the repository
git clone https://github.com/ravikirankrishnaprasad/SentimentBasedProductRecommendationSystem.git
cd ebuss-recommendation-system

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
