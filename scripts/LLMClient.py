from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from google import genai
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional, Dict, Any
import os
import logging
import torch
from openai import OpenAI
import time
import random

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        api_key,
        model_name,
        temperature=0,
        embedding_model="all-MiniLM-L6-v2",
        base_url=None,
        max_retries=3,
        initial_delay=1.0,
        max_delay=60.0,
        backoff_factor=2.0,
    ):
        # if model name has gemini in it, use genai
        if "gemini" in model_name:
            self.client = genai.Client(api_key=api_key)
        elif base_url:
            self.base_url = base_url
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            # Use ChatGroq for other models
            self.client = ChatGroq(temperature=temperature, groq_api_key=api_key, model_name=model_name)

        self.model_name = model_name
        
        # Retry configuration
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

        # Initialize embeddings model with error handling
        try:
            # Set default device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": device},
                encode_kwargs={"device": device, "batch_size": 32},
            )
            # Test the embeddings
            test_text = "Test embedding initialization"
            _ = self.embeddings.embed_query(test_text)
            logger.info("Successfully initialized embeddings model")

        except Exception as e:
            logger.error(f"Failed to initialize embeddings model: {str(e)}")
            logger.warning("Attempting fallback to CPU-only mode...")

            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"device": "cpu", "batch_size": 32},
                )
                # Test the embeddings
                test_text = "Test embedding initialization"
                _ = self.embeddings.embed_query(test_text)
                logger.info("Successfully initialized embeddings model in CPU mode")

            except Exception as e2:
                logger.error(f"Fallback initialization also failed: {str(e2)}")
                logger.warning("RAG capabilities will be disabled")
                self.embeddings = None

        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def _is_retryable_error(self, error):
        """Check if an error is retryable (timeout, rate limit, temporary server error)"""
        error_str = str(error).lower()
        retryable_keywords = [
            'timeout',
            'rate limit',
            'rate_limit',
            'too many requests',
            'service unavailable',
            'internal server error',
            'bad gateway',
            'connection error',
            'connection timeout',
            'read timeout',
            'request timeout',
            'quota exceeded',
            'overloaded',
            'temporarily unavailable',
            'try again',
            'retry',
            'status code 429',
            'status code 500',
            'status code 502',
            'status code 503',
            'status code 504',
        ]
        
        return any(keyword in error_str for keyword in retryable_keywords)

    def _calculate_delay(self, attempt):
        """Calculate delay for exponential backoff with jitter"""
        delay = self.initial_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.3) * delay
        return delay + jitter

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    # Last attempt, don't retry
                    logger.error(f"Max retries ({self.max_retries}) exceeded for LLM request. Final error: {str(e)}")
                    raise e
                
                if not self._is_retryable_error(e):
                    # Not a retryable error, don't retry
                    logger.error(f"Non-retryable error in LLM request: {str(e)}")
                    raise e
                
                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                logger.warning(f"LLM request failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}")
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        
        # This should never be reached, but just in case
        if last_exception:
            raise last_exception

    def create_vector_store(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Create a vector store from a list of documents.

        Args:
            documents: List of text documents to index
            metadata: Optional list of metadata dictionaries for each document
        """
        if not self.embeddings:
            raise ValueError("Embeddings model not initialized. RAG capabilities are disabled. Please check the logs for initialization errors.")

        if not documents:
            raise ValueError("No documents provided for indexing")

        try:
            # Split documents into chunks
            splits = self.text_splitter.split_text("\n\n".join(documents))

            # Create vector store
            self.vector_store = Chroma.from_texts(
                texts=splits,
                embedding=self.embeddings,
                metadatas=[metadata[i] if metadata else None for i in range(len(splits))],
                persist_directory="chroma_db",
            )
            logger.info(f"Successfully created vector store with {len(splits)} chunks")

        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise ValueError(f"Failed to create vector store: {str(e)}")

    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        """
        Perform similarity search on the vector store.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of similar documents
        """
        if not self.embeddings:
            raise ValueError("Embeddings model not initialized. RAG capabilities are disabled.")
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")

        try:
            results = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise ValueError(f"Error during similarity search: {str(e)}")

    def generate_with_context(self, prompt: str, k: int = 3):
        """
        Generate a response using RAG - retrieve relevant context and use it to augment the prompt.

        Args:
            prompt: The user's prompt
            k: Number of relevant documents to retrieve

        Returns:
            Generated response
        """
        if not self.embeddings:
            logger.warning("RAG capabilities are disabled. Falling back to standard generation.")
            return self.generate(prompt)

        if not self.vector_store:
            logger.warning("Vector store not initialized. Falling back to standard generation.")
            return self.generate(prompt)

        try:
            # Retrieve relevant documents
            relevant_docs = self.similarity_search(prompt, k=k)

            # Construct augmented prompt
            context = "\n\n".join(relevant_docs)
            augmented_prompt = f"""Context information is below.
            ---------------------
            {context}
            ---------------------
            Given the context information and not prior knowledge, answer the following query:
            {prompt}"""

            print(augmented_prompt)

            return self.generate(augmented_prompt)

        except Exception as e:
            logger.error(f"Error in generate_with_context: {str(e)}")
            logger.warning("Falling back to standard generation due to error")
            return self.generate(prompt)

    def generate(self, prompt):
        """Generate response with retry logic for timeout and rate limit errors"""
        def _generate_internal():
            # Escape curly braces in the context TODO: the prompt should be escaped in the template
            escaped_context = prompt.replace("{", "{{").replace("}", "}}")

            if "gemini" in self.model_name:
                response = self.client.models.generate_content(model="gemini-2.0-flash", contents=escaped_context)
                return response.text

            elif self.base_url:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": escaped_context}],
                )
                return response.choices[0].message.content

            # Create a ChatPromptTemplate with only a system message
            model_prompt = ChatPromptTemplate.from_messages([("system", escaped_context)])

            # Invoke the prompt without any additional input
            prompt_value = model_prompt.invoke({})

            # Pass the formatted prompt to the model
            response = self.client.invoke(prompt_value)
            return response.content

        # Use retry logic for generation
        return self._retry_with_backoff(_generate_internal)
