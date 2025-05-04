# 🏡 Airbnb Data Analysis & Recommender System

## 📊 Project Overview

This project presents a complete data science pipeline for analyzing Airbnb rental listings and building a content-based recommendation engine. The dataset was sourced from Kaggle and includes ~70,000 listings.

The goal is to predict listing prices and provide personalized recommendations based on listing descriptions, using both machine learning and natural language processing (NLP) techniques.

## 🔧 Key Features

- Data preprocessing and cleaning:
  - Missing value imputation using KNN and statistical methods
  - Normalization and transformation of price using `log1p`
  - Encoding of categorical features

- Exploratory Data Analysis:
  - Distribution of prices across locations, host types, amenities
  - Correlations between price and features such as room type, availability, review count

- Price prediction models:
  - Linear Regression, Ridge, Lasso
  - Decision Tree, Random Forest
  - **CatBoost (best): R² = 0.69, MSE = 0.15**

- Content-based Recommendation System:
  - Description text vectorized with **TF-IDF**
  - Cosine similarity to find similar listings
  - Results returned as top 5 most similar listings

## 🚀 Technologies Used

- Python (Jupyter Notebook)
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, CatBoost, XGBoost
- NLTK, TfidfVectorizer (NLP)
- HDF5 (for storing similarity matrix efficiently)

## 📂 Files

| File | Description |
|------|-------------|
| `Airbnb_practicum-2.ipynb` | Jupyter Notebook with full pipeline from cleaning to modeling and recommendation |
| `instructions.pdf` | Project instructions & goals (Hebrew) |
| `practicum.pdf` | Final project report (Hebrew) |

## ▶️ How to Run

1. Install required libraries from `requirements.txt`
2. Launch Jupyter Notebook:
```bash
jupyter notebook Airbnb_practicum-2.ipynb
```
3. Run cells in order to view analysis, model training, and recommender results.

## 💡 Sample Output

> Top 5 similar listings to a selected Airbnb post based on description similarity:
```
1. Cozy room near central station, free Wi-Fi
2. Quiet modern studio in heart of the city
3. Spacious 2BR with balcony and city view
4. Clean private room, near metro station
5. Beautiful apartment with historic charm
```

## ⚠️ Notes
- Content-based recommender does not require user history or collaborative filtering
- Price prediction is done on log scale and transformed back for interpretability

---

This project demonstrates data science techniques from EDA and feature engineering to supervised learning and recommendation systems using NLP. Suitable for portfolios in machine learning, data analysis, or product personalization.
