"""
RAG Module for Workshop Chat Application

This module contains the core RAG (Retrieval-Augmented Generation) logic
that is shared between the Streamlit chat application and the DeepEval
evaluation framework.

By centralizing the RAG logic here, we ensure that:
1. The chat app and evaluation test the exact same system
2. Changes to prompts/retrieval affect both consistently
3. Code is DRY (Don't Repeat Yourself)
4. Easier to maintain and improve
"""

import os
from typing import Tuple, List
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_iris import IRISVector
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

class WorkshopRAG:
    """
    Workshop RAG System
    
    This class encapsulates the RAG pipeline used throughout the workshop.
    It can be used by both the Streamlit chat app and the evaluation framework.
    """
    
    def __init__(self, 
    ##             collection_name: str = "case_reports",
                 connection_string: str = None,
                 llm_model: str = "gpt-4-turbo",
                 temperature: float = 0.0):
        """
        Initialize the RAG system
        
        Args:
            collection_name: Name of the IRIS vector collection
            connection_string: IRIS connection string (defaults to workshop settings)
            llm_model: LLM model to use
            temperature: LLM temperature setting
        """
        # Set default connection if not provided
        if connection_string is None:
            connection_string = "iris://_SYSTEM:SYS@IRIS:1972/IRISAPP"
        
        # Initialize embeddings
        self.embeddings = FastEmbedEmbeddings()
        
        # Initialize IRIS vector store
        self.db = IRISVector(
            embedding_function=self.embeddings,
            dimension=384,  # FastEmbed dimension
            collection_name=collection_name,
            connection_string=connection_string,
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature
        )
        
        print(f"✅ RAG system initialized:")
        print(f"   - Collection: {collection_name}")
        print(f"   - LLM: {llm_model}")
        print(f"   - Temperature: {temperature}")
    
    def query(self, question: str, conversation_history: str = "", use_conversation_chain: bool = False, conversation_chain=None) -> Tuple[str, List[str]]:
        """
        Process a question through the RAG pipeline
        
        Args:
            question: User's question
            conversation_history: Previous conversation context (optional)
            
        Returns:
            Tuple of (answer, retrieved_contexts)
        """
        # Step 1: Vector similarity search
        docs_with_score = self.db.similarity_search_with_score(question)
        
        # Step 2: Build relevant documents
        relevant_docs = [
            "".join(str(doc.page_content)) + " " for doc, _ in docs_with_score
        ]
        
        # Step 3: Create the template prompt
        template = f"""
Prompt: {question}

### Add conversation history here

Relevant Documents: {x}

### Add guard rails here
                """
        
        # Step 4: Generate response using LLM (same as chat app)
        if use_conversation_chain and conversation_chain:
            # Use conversation chain for chat app (maintains memory)
            response = conversation_chain.predict(input=template)
            answer = response
        else:
            # Use direct LLM for evaluation (no memory needed)
            response = self.llm.invoke(template)
            answer = response.content
        
        # Return answer and contexts for evaluation
        contexts = [doc.page_content for doc, _ in docs_with_score]
        return answer, contexts
    
    def get_retriever(self):
        """Get the retriever for compatibility with existing code"""
        return self.db.as_retriever()
    
    def get_db(self):
        """Get the database for direct access"""
        return self.db
    
    def get_llm(self):
        """Get the LLM for direct access"""
        return self.llm

# Convenience function for quick RAG queries
def quick_rag_query(question: str, conversation_history: str = "") -> Tuple[str, List[str]]:
    """
    Quick RAG query using default settings
    
    Args:
        question: User's question
        conversation_history: Previous conversation context (optional)
        
    Returns:
        Tuple of (answer, retrieved_contexts)
    """
    rag = WorkshopRAG()
    return rag.query(question, conversation_history)

if __name__ == "__main__":
    # Test the RAG module
    print("🧪 Testing RAG Module...")
    
    try:
        rag = WorkshopRAG()
        
        test_question = "What are common symptoms of knee problems in adult patients?"
        answer, contexts = rag.query(test_question)
        
        print(f"\n📝 Test Question: {test_question}")
        print(f"🤖 Answer: {answer[:200]}...")
        print(f"📄 Retrieved {len(contexts)} contexts")
        print("\n✅ RAG module test successful!")
        
    except Exception as e:
        print(f"❌ RAG module test failed: {e}")