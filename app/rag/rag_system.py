from qdrant_client import QdrantClient
from openai import OpenAI

class RAG:
    def __init__(
        self,
        openai_client: OpenAI = None,
        vector_db_client: QdrantClient = None,
        embedding_model: str = None,
        llm_model: str = None,
        collection_name: str = None,
        top_k: int = 5,
        score_threshold: float = 0.0
    ):
        self.openai_client = openai_client
        self.vector_db_client = vector_db_client
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.collection_name = collection_name
        self.top_k = top_k
        self.score_threshold = score_threshold
        
    def retrieve(self, query: str) -> list[dict]:
        # retrieve relevant documents from vector db
        query_embedding = self._get_query_embedding(query)

        if query_embedding is None:
            print("Failed to get query embedding. Cannot retrieve documents.")
            return []

        retrieved_docs = self._retrieve_docs_list_from_query_embedding(query_embedding)

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
        base_system_prompt = (
            "You are a helpful assistant and an expert in industrial systems. "
            "If a question is outside the industrial domain, state that you cannot answer it."
        )

        try:

            if force_no_context: # DEBUG: to test how the model responds without context
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": base_system_prompt},
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
                            "content": (
                                f"{base_system_prompt}\n"
                                "Answer the user's <USER_QUERY> using only <CONTEXT>. "
                                "Do not add facts that are not present in the context. "
                                "If the context is missing, insufficient, or does not cover the request, reply exactly: "
                                "'I don't know based on the provided context.'\n\n"
                                "When context is sufficient, provide:\n"
                                "1) Short summary\n"
                                "2) Possible troubleshooting\n"
                                "3) Checks to perform\n"
                                "4) Actions / next steps\n\n"
                                f"<CONTEXT>\n{context_from_docs or 'No context provided.'}\n</CONTEXT>\n"
                            )
                        },
                        {
                            "role": "user",
                            "content": f"<USER_QUERY>\n{query}\n</USER_QUERY>"
                        }
                    ],
                    temperature=0.0, # lower temperature for more deterministic responses
                    max_completion_tokens=1000,
                    verbosity="low",
                    seed=42
                )
            
            answer = response.choices[0].message.content.strip()
            return answer
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I encountered an error while generating the response."

    def _retrieve_docs_list_from_query_embedding(self, query_embedding: list[float]) -> list[dict]:

        results = self.vector_db_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=self.top_k,
            score_threshold=self.score_threshold,
            with_payload=True
            )
        
        docs = []
        for point in results.points:
        
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
