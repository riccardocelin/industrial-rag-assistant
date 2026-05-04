from qdrant_client import QdrantClient
from openai import OpenAI

class RAG:
    def __init__(
        self,
        openai_client: OpenAI,
        vector_db_client: QdrantClient,
        embedding_model: str,
        llm_model: str,
        collection_name: str,
        top_k: int,
    ):
        self.openai_client = openai_client
        self.vector_db_client = vector_db_client
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.collection_name = collection_name
        self.top_k = top_k

        
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
        state = {
            "openai_client":        self.openai_client,
            "vector_db_client":     self.vector_db_client,
            "embedding_model":      self.embedding_model,
            "llm_model":            self.llm_model,
            "collection_name":      self.collection_name,
            "top_k":                self.top_k
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
                        {"role": "system", "content": "You are a helpful assistant, expert in industrial field. Do not answer to questions not related to the industrial field."},
                        {
                            "role": "user",
                            "content": query
                        }
                    ],
                    temperature=0.0, # lower temperature for more deterministic responses
                    max_completion_tokens=1000,
                    verbosity="low",
                    seed=42
                )

            else: # regular rag behaviour
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {   
                            "role": "system",
                            "content":
                            ""
                                f"You are a helpful assistant, expert in industrial field. Do not answer to questions not related to the industrial field."
                                f"Answer the user <USER_QUERY> based only on the provided <CONTEXT>, providing a summary. If the query is not covered by the context, say that you don't know.\n\n"
                                f"If available from the context, answer by structuring: possible troubleshooting, checks to be performed, actions or next steps.\n\n"
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
                    ],
                    temperature=0.0, # lower temperature for more deterministic responses
                    max_completion_tokens=1000,
                    verbosity="low",
                    seed=42
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
                "score": point.score,
                "text": point.payload.get("text"),
                "source": point.payload.get("source"),
                "pages": point.payload.get("pages"),
                }
            docs.append(doc_info)

        return docs
    
    def _get_text_from_retrieved_docs(self, retrieved_docs: list[dict]) -> str:
        # concatenate the text of the retrieved docs to provide context to the generator
        empty_str = ""
        context = empty_str.join([f"doc[{i}]: {doc['text']}\n\n" for i, doc in enumerate(retrieved_docs)])
        return context