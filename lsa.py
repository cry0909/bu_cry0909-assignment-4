import matplotlib
matplotlib.use('Agg')  # Use the Agg backend to avoid the threading issue
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Create the term-document matrix using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Perform Singular Value Decomposition (SVD) to reduce dimensionality
svd = TruncatedSVD(n_components=100)
X_reduced = svd.fit_transform(X)

# Process the query and compute cosine similarity
def process_query(query):
    query_vec = vectorizer.transform([query])
    query_vec_reduced = svd.transform(query_vec)
    similarities = cosine_similarity(query_vec_reduced, X_reduced)

    top_indices = similarities.argsort()[0][-5:][::-1]
    return similarities, top_indices

# Generate a bar chart for the top 5 documents
def create_bar_chart(similarities, top_indices):
    top_similarities = [similarities[0][i] for i in top_indices]
    top_docs = [f"Doc {i+1}" for i in top_indices]  # Just placeholder document names

    # Create the bar chart
    plt.figure(figsize=(8, 4))
    plt.bar(top_docs, top_similarities, color='skyblue')
    plt.xlabel('Documents')
    plt.ylabel('Cosine Similarity')
    plt.title('Top 5 Documents by Cosine Similarity')
    
    # Save the plot as an image file in the 'static' directory
    plt.savefig('static/similarity_chart.png')
    plt.close()  # Close the plot to avoid keeping it open
