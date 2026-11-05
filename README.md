# Credit Card Customer Feedback Sentiment Analysis

## Project Overview
Credit Card Customer Feedback Sentiment Analysis is a machine learning and Natural Language Processing (NLP) project designed to analyze customer reviews related to credit card services. The system processes review text and classifies customer feedback into three sentiment categories: Positive, Negative, and Neutral.

The project uses text preprocessing and machine learning algorithms to understand customer opinions and identify feedback patterns.

## Objectives
- Analyze customer feedback related to credit card services.
- Process and convert review text into numerical features.
- Classify reviews into positive, negative, and neutral sentiments.
- Compare different machine learning algorithms.
- Evaluate model performance using classification metrics.

## Dataset
The project uses a credit card customer reviews dataset.

**Dataset File:** `credit_card_reviews_360 (1).csv`

**Input Features:**
- Review_Text
- Card_Category
- Primary_Topic

**Target Variable:** `Sentiment`

## Methodology
1. Load and explore the dataset.
2. Handle missing values.
3. Preprocess customer review text.
4. Convert text into numerical features using TF-IDF.
5. Encode categorical features.
6. Split the dataset into training and testing sets.
7. Train and compare machine learning models.
8. Evaluate model performance.

## Algorithms Used
- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Spyder IDE

## Evaluation Metrics
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

## Expected Outcome
The system classifies credit card customer reviews as Positive, Negative, or Neutral. Comparing different algorithms helps identify their performance in customer sentiment classification.

## Applications
- Understanding customer satisfaction.
- Identifying negative customer feedback.
- Analyzing customer opinions about banking services.
- Supporting improvements in credit card services.

## Disclaimer
This project is developed for educational purposes and demonstrates sentiment analysis using machine learning techniques.

## Author
B.Tech Artificial Intelligence and Data Science Student
