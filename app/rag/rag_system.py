from qdrant_client import QdrantClient
from openai import OpenAI

# RAG pipeline
# the idea is to have a RAG class that takes in a query and returns and answer
# the RAG class will have a retriever and a generator methods

class RAG:
    def __init__(
            self, openai_key: str, embedding_model: str = "text-embedding-3-small", llm_model: str = "gpt-5.4-mini",
              qdrant_config: dict = {"host": "localhost", "port": 6333, "collection_name": "my-collection"}, top_k: int = 5):
        
        self.openai_client = OpenAI(api_key=openai_key)
        self.vector_db_client = QdrantClient(host=qdrant_config["host"], port=qdrant_config["port"])
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.collection_name = qdrant_config["collection_name"]
        self.top_k = top_k

        
    def retrieve(self, query: str):
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

        retrieved_docs = self._retrieve_text_list_from_docs(results)
        return retrieved_docs


    def generate(self, query: str, retrieved_docs: list[str] = None, force_no_context: bool = False):
        response = self._generate_response(query, retrieved_docs, force_no_context)
        return response 

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


    def _generate_response(self, query: str, retrieved_docs: list[str] = None, force_no_context: bool = False) -> str:

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
                return response.choices[0].message.content.strip()
                
            else: # regular rag behaviour
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant, expert in industrial field."},
                        {
                            "role": "user",
                            "content":
                            ""
                            f"Answer the user <QUERY> based on the provided <CONTEXT>, providing a summary. If the query is not covered by the context, say that you don't know.\n\n"
                            f"<CONTEXT>\n{retrieved_docs}\n</CONTEXT>\n\n"
                            f"<QUERY>\n{query}\n</QUERY>\n\n"
                            ""
                        }
                    ]
                )
                return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I encountered an error while generating the response."

    def _retrieve_text_list_from_docs(self, qdrant_results: tuple) -> list[str]:
        docs = []
        for point in qdrant_results.points:
            docs.append(point.payload["text"])

        return docs