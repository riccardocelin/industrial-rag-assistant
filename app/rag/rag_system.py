from qdrant_client import QdrantClient
from openai import OpenAI
from app.core.settings import get_settings

# RAG pipeline
# the idea is to have a RAG class that takes in a query and returns and answer
# the RAG class will have a retriever and a generator methods

class RAG:
    def __init__(self):
        
        settings = get_settings()
        self.openai_client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self.vector_db_client = QdrantClient(host=settings.vector_db_host, port=settings.vector_db_port)
        self.embedding_model = settings.openai_embedding_model
        self.llm_model = settings.openai_llm_model
        self.collection_name = settings.vector_db_collection_name
        self.top_k = settings.retrieval_top_k

        
    def retrieve(self, query: str) -> list[dict]:
        # retrieve relevant documents from vector db
        query_embedding = self._get_query_embedding(query)

        if query_embedding is None:
            print("Failed to get query embedding. Cannot retrieve documents.")
            return []

        results = self.vector_db_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=self.top_k,
            with_payload=True
            )

        retrieved_docs = self._retrieve_docs_list_from_results(results)
        return retrieved_docs


    def generate(self, query: str, retrieved_docs: list[dict] = None, force_no_context: bool = False) -> str:
        response = self._generate_response(query, retrieved_docs, force_no_context)
        return response
    
    def get_internal_state(self):
        # for debugging purposes, to check the internal state of the RAG system
        settings = get_settings()
        state = {
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "vector_db_host": settings.vector_db_host,
            "vector_db_port": settings.vector_db_port,
            "collection_name": self.collection_name,
            "top_k": self.top_k,
        }
        return state

    def _get_query_embedding(self, query: str):

        try:
            query_embedding = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[query]
            ).data[0].embedding
            return query_embedding
        except Exception as e:
            print(f"Error getting query embedding: {e}")
            return None


    def _generate_response(self, query: str, retrieved_docs: list[dict] = None, force_no_context: bool = False) -> str:

        response = None
        context_from_docs = self._get_text_from_retrieved_docs(retrieved_docs)

        try:

            if force_no_context: # DEBUG: to test how the model responds without context
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant, expert in industrial field."},
                        {
                            "role": "user",
                            "content": query
                        }
                    ]
                )
                
            else: # regular rag behaviour
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {   
                            "role": "system",
                            "content":
                            ""
                                f"You are a helpful assistant, expert in industrial field."
                                f"Answer the user <USER_QUERY> based on the provided <CONTEXT>, providing a summary. If the query is not covered by the context, say that you don't know.\n\n"
                                f"If available from teh context, answer by structuring: possible troubleshootin, checks to be performed, actions or next steps.\n\n"
                                f"<CONTEXT>\n{context_from_docs}\n</CONTEXT>\n\n"
                            ""
                        },
                        {
                            "role": "user",
                            "content":
                            ""
                                f"<USER_QUERY>\n{query}\n</USER_QUERY>\n\n"
                            ""
                        }
                    ]
                )
            
            answer =response.choices[0].message.content.strip()
            return answer
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I encountered an error while generating the response."

    def _retrieve_docs_list_from_results(self, qdrant_results: tuple) -> list[dict]:
        docs = []
        for point in qdrant_results.points:
            
            doc_info = {
                "chunk_id": point.id,
                "source": point.payload.get("source"),
                "text": point.payload.get("text"),
                "score": point.score
                }
            docs.append(doc_info)

        return docs
    
    def _get_text_from_retrieved_docs(self, retrieved_docs: list[dict]) -> str:
        # concatenate the text of the retrieved docs to provide context to the generator
        empty_str = ""
        context = empty_str.join([f"doc[{i}]: {doc['text']}\n\n" for i, doc in enumerate(retrieved_docs)])
        return context