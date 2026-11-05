
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)

# Load dataset
df = pd.read_csv(r"C:\Users\likit\Downloads\credit_card_reviews_360 (1).csv")

df = df.dropna(subset=[
    "Review_Text", "Card_Category", "Primary_Topic", "Sentiment"
])

print("Dataset Shape:", df.shape)
print("\nSentiment Distribution:")
print(df["Sentiment"].value_counts())

# Input features and target
X = df[["Review_Text", "Card_Category", "Primary_Topic"]]
y = df["Sentiment"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Text and categorical feature processing
preprocessor = ColumnTransformer([
    ("text", TfidfVectorizer(
        stop_words="english", max_features=3000
    ), "Review_Text"),
    ("category", OneHotEncoder(handle_unknown="ignore"),
     ["Card_Category", "Primary_Topic"])
])

# Three ML algorithms
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

results = []
predictions = {}

# Train and evaluate models
for name, algorithm in models.items():
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", algorithm)
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    results.append({
        "Algorithm": name,
        "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision (%)": round(
            precision_score(y_test, y_pred, average="macro",
                            zero_division=0) * 100, 2
        ),
        "Recall (%)": round(
            recall_score(y_test, y_pred, average="macro",
                         zero_division=0) * 100, 2
        ),
        "F1 Score (%)": round(
            f1_score(y_test, y_pred, average="macro",
                     zero_division=0) * 100, 2
        )
    })

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

# Compare algorithms
results_df = pd.DataFrame(results)

print("\nALGORITHM COMPARISON")
print(results_df.to_string(index=False))

# Plot 1: Sentiment distribution
plt.figure(figsize=(7, 5))
df["Sentiment"].value_counts().reindex(
    ["Negative", "Neutral", "Positive"]
).plot(kind="bar")

plt.title("Customer Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Plot 2: Algorithm performance comparison
results_df.set_index("Algorithm").plot(
    kind="bar", figsize=(10, 6)
)

plt.title("Comparison of Three ML Algorithms")
plt.ylabel("Score (%)")
plt.xlabel("Algorithm")
plt.ylim(0, 105)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Plot 3: Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (name, y_pred) in zip(axes, predictions.items()):
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=["Negative", "Neutral", "Positive"],
        ax=ax,
        colorbar=False
    )
    ax.set_title(name)

plt.tight_layout()
plt.show()

# Select the best model using macro F1 score
best_name = results_df.loc[
    results_df["F1 Score (%)"].idxmax(), "Algorithm"
]

print("\nBest Algorithm based on Macro F1 Score:", best_name)

# Train the selected model
best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", models[best_name])
])

best_model.fit(X_train, y_train)

# Create readable sample predictions
test_predictions = X_test.copy()
test_predictions["Actual_Sentiment"] = y_test
test_predictions["Predicted_Sentiment"] = best_model.predict(X_test)

print("\nSAMPLE PREDICTIONS")
print("-" * 75)

for i, (_, row) in enumerate(
    test_predictions.head(10).iterrows(), 1
):
    review = row["Review_Text"]

    print(f"\nReview {i}: {review[:100]}...")
    print(f"Actual Sentiment:    {row['Actual_Sentiment']}")
    print(f"Predicted Sentiment: {row['Predicted_Sentiment']}")

    if row["Actual_Sentiment"] == row["Predicted_Sentiment"]:
        print("Result: Correct Prediction")
    else:
        print("Result: Incorrect Prediction")

# Save test predictions
test_predictions.to_csv(
    "credit_card_sentiment_predictions.csv", index=False
)

print("\nPredictions saved to credit_card_sentiment_predictions.csv")

# Predict sentiment for a new customer review
new_review = pd.DataFrame([{
    "Review_Text": "The rewards are excellent and customer service was helpful.",
    "Card_Category": "Cash Back",
    "Primary_Topic": "Rewards"
}])

predicted_sentiment = best_model.predict(new_review)[0]

print("\nNEW CUSTOMER REVIEW")
print(new_review["Review_Text"].iloc[0])
print("Predicted Sentiment:", predicted_sentiment)

