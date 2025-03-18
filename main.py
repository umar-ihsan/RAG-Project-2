from src.mongodb_utils import connect_to_mongodb, get_articles_from_mongodb, convert_to_documents, split_documents
from src.vector_store import initialize_embeddings, load_or_create_faiss_vectorstore
from src.rag import run_rag_system

def setup_documents():
    # Set your MongoDB connection details here:
    connection_string = "mongodb+srv://jamshidjunaid763:JUNAID12345@insightwirecluster.qz5cz.mongodb.net/?retryWrites=true&w=majority&appName=InsightWireCluster"
    database_name = "Scraped-Articles-11"
    collection_name = "Articles"

    client = connect_to_mongodb(connection_string)
    if client:
        articles = get_articles_from_mongodb(client, database_name, collection_name)
        documents = convert_to_documents(articles)
        document_chunks = split_documents(documents)
        return document_chunks
    else:
        print("MongoDB connection failed. Exiting.")
        return []

def main():
    print("Welcome to the local RAG system. Type 'exit' to quit.\n")
    
    # Setup documents and persistent vector store once at startup
    document_chunks = setup_documents()
    if not document_chunks:
        print("No documents loaded. Exiting.")
        return

    embeddings = initialize_embeddings()
    vector_store = load_or_create_faiss_vectorstore(document_chunks, embeddings)

    # Interactive query loop
    while True:
        query = input("Enter your query: ").strip()
        if query.lower() in ['exit', 'quit']:
            print("Exiting the RAG system. Goodbye!")
            break

        final_response = run_rag_system(query, vector_store)
        print("\nFinal Response:")
        print(final_response, "\n")

if __name__ == '__main__':
    main()
