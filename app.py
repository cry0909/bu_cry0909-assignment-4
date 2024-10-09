from flask import Flask, render_template, request
from lsa import process_query, create_bar_chart, documents  # Import the documents dataset

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    
    # Step 1: Process the query and get similarities and top document indices
    similarities, top_indices = process_query(query)
    
    # Step 2: Retrieve the top 5 documents, excerpts, and their similarity scores
    top_docs = []
    for i in top_indices:
        doc_name = f"Document {i+1}"  # Create a placeholder for the document name
        doc_excerpt = documents[i][:200] + '...'  # Get the first 200 characters of the document
        similarity_score = f"Similarity: {similarities[0][i]:.4f}"  # Format the similarity score
        
        # Combine the document name, excerpt, and similarity score in the desired format
        top_docs.append(f"{doc_name}\n{doc_excerpt}\n{similarity_score}")
    
    # Step 3: Create a bar chart for the top 5 documents
    create_bar_chart(similarities, top_indices)
    
    # Step 4: Render the results in the HTML template along with the chart
    return render_template('index.html', results=top_docs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
