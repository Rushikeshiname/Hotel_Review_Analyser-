import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK data
#nltk.download('stopwords')

# Sample dataset (you can replace this with your own CSV file)
data = {
    'review': [
        "This product is amazing, I love it!",
        "Worst experience ever. Totally disappointed.",
        "Pretty decent quality for the price.",
        "Excellent performance, very satisfied.",
        "Not worth the money, terrible quality."
    ],
    'sentiment': ['positive', 'negative', 'positive', 'positive', 'negative']
}

df = pd.DataFrame(data)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Clean reviews
df['clean_review'] = df['review'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['sentiment'], test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Function to analyze new review
def analyse_review(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return prediction

# Example usage
sample_review = "The battery life is awful and camera is bad."
print(f"Review: {sample_review}")
print(f"Predicted Sentiment: {analyse_review(sample_review)}")
