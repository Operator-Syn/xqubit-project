import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pandas as pd

# Ensure NLTK resources are available locally
nltk.download('stopwords', quiet=True)

def process_features(train_df, test_df, n_components=2):
    """
    Transforms raw text into a reduced numerical space for Quantum mapping.
    Uses TF-IDF for vectorization and PCA for dimensionality reduction.
    """
    print(f"🛠️ Vectorizing text and reducing to {n_components} components...")

    # 1. Initialize TF-IDF with English stopword removal
    # We limit to 500 features initially to maintain a manageable dense matrix
    stop_words = list(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=500)

    # 2. Fit and transform training data; transform test data using the training vocabulary
    # This prevents 'data leakage' from the test set into our feature space
    X_train_tfidf = vectorizer.fit_transform(train_df['text']).toarray()
    X_test_tfidf = vectorizer.transform(test_df['text']).toarray()

    # 3. Principal Component Analysis (PCA)
    # Reducing the semantic space to match our target Qubit count
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_tfidf)
    X_test_pca = pca.transform(X_test_tfidf)

    print(f"✅ Feature Engineering Complete.")
    print(f"PCA Variance Explained: {sum(pca.explained_variance_ratio_)*100:.2f}%")

    return X_train_pca, X_test_pca, vectorizer, pca

    print("=" * 40)
    print("  Code for the above program is made by        : John-Ronan S. Beira")
    print("=" * 40)
