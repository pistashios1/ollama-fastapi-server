# Separate file to store static functions

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([y ** 2 for y in b]) ** 0.5
    return dot_product / (norm_a * norm_b)


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 300):
	# Create an instance of the RecursiveCharacterTextSplitter class
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
	)

	# Split the input text into chunks
	text_chunks = text_splitter.split_text(text)

	return text_chunks

def retrieve_relevant_chunks(query: str, text_chunks: list[str]):
    # Create an instance of the OllamaEmbeddings class
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://127.0.0.1:11434/",
        
    )
    query_embedding = embeddings.embed_query(query)

    # Calculate similarity scores between the query embedding and text chunk embeddings
    similarity_scores = []
    for chunk in text_chunks:
        chunk_embedding = embeddings.embed_documents([chunk])[0]
        similarity_score = cosine_similarity(query_embedding, chunk_embedding)
        similarity_scores.append((chunk, similarity_score))

    # Sort the chunks based on similarity scores in descending order
    sorted_chunks = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    return sorted_chunks

if __name__ == "__main__":
    # Example usage
    sample_text = "Your sample text goes here."
    chunks = chunk_text(sample_text)
    query = "Your query goes here."
    relevant_chunks = retrieve_relevant_chunks(query, chunks)
    print(relevant_chunks)