# Import the Python libaries that will be used for this app.
# Libraries of note:
# Streamlit, a Python library that makes it easy to create and share beautiful, custom web apps for data science and machine learning.
# ChatOpenAI, a class that provides a simple interface to interact with OpenAI's models.
# ConversationChain and ConversationSummaryMemory, classes that represents a conversation between a user and an AI and retain the context of a conversation.
# OpenAIEmbeddings, a class that provides a way to perform vector embeddings using OpenAI's embeddings.
# IRISVector, a class that provides a way to interact with the IRIS vector store.
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_iris import IRISVector
import os

# Import dotenv, a module that provides a way to read environment variable files, and load the dotenv (.env) file that provides a few variables we need
from dotenv import load_dotenv

load_dotenv(override=True)

from utils import LLM_MODEL

# Load the urlextractor, a module that extracts URLs and will enable us to follow web-links
from urlextract import URLExtract

extractor = URLExtract()

# Import our shared RAG module
from rag_module import WorkshopRAG

# Initialize the RAG system
@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached for performance)"""
    return WorkshopRAG(
        collection_name="case_reports",
        llm_model=LLM_MODEL,
        temperature=0.0
    )

# Get the RAG system
rag_system = initialize_rag()

# Used to have a starting message in our application
# Check if the "messages" key exists in the Streamlit session state.
# If it doesn't exist, create a new list and assign it to the "messages" key.
if "messages" not in st.session_state:
    # Initialize the "messages" list with a welcome message from the assistant.
    st.session_state["messages"] = [
        # The role of this message is "assistant", and the content is a welcome message.
        {
            "role": "assistant",
            "content": "Hi, I'm a chatbot that can access your vector stores. What would you like to know?",
        }
    ]

# Initialize conversation chain in session state if not present
if "conversation_sum" not in st.session_state:
    llm = ChatOpenAI(
        temperature=0.0,
        model_name=LLM_MODEL,
    )
    st.session_state["conversation_sum"] = ConversationChain(
        llm=llm,
        memory=ConversationSummaryMemory(llm=llm),
        verbose=True,
    )

# Add a title for the application
# This line creates a header in the Streamlit application with the title "GS 2024 Vector Search"
st.header("↗️ READY 2025 Vector Search ↗️")

# Customize the UI
# In streamlit we can add settings using the st.sidebar
with st.sidebar:
    st.header("Settings")
    temperature_slider = st.slider("Temperature", float(0), float(1), float(0.0), float(0.01))
    # link_retrieval = st.radio("Retrieve Links?:",("No","Yes"),index=0)
    # Allow user to toggle whether explanation is shown with responses
    explain = st.radio("Show explanation?:", ("Yes", "No"), index=0)

# In streamlet, we can add our messages to the user screen by listening to our session
for msg in st.session_state["messages"]:
    # If the "chat" is coming from AI, we write the content with the ISC logo
    if msg["role"] == "assistant":
        st.chat_message(msg["role"]).write(msg["content"])
    # If the "chat" is the user, we write the content as the user image, and replace some strings the UI doesn't like
    else:
        st.chat_message(msg["role"]).write(msg["content"].replace("$", "\$"))

# Check if the user has entered a prompt (input) in the chat window
if prompt := st.chat_input():

    # Add the user's input to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the user's input in the chat window, escaping any '$' characters
    st.chat_message("user").write(prompt.replace("$", "\$"))

    # Retrieve the conversation chain instance from session state.
    conversation_sum = st.session_state["conversation_sum"]

    # Here we respond to the user based on the messages they receive
    with st.chat_message("assistant"):
        # Get conversation history from memory
        conversation_history = conversation_sum.memory.load_memory_variables({})['history']
        
        # 🚀 NEW: Use the shared RAG module instead of duplicating logic
        try:
            # Query the RAG system with conversation chain for memory
            resp, retrieved_contexts = rag_system.query(
                question=prompt,
                conversation_history=conversation_history,
                use_conversation_chain=True,
                conversation_chain=conversation_sum
            )
            
            # Display debug information if requested
            if explain == "Yes":
                st.write("**Retrieved Contexts:**")
                for i, context in enumerate(retrieved_contexts[:2]):  # Show first 2 contexts
                    st.write(f"Context {i+1}: {context[:200]}...")
                st.write("---")
            
        except Exception as e:
            resp = f"Sorry, I encountered an error: {str(e)}"
            st.error(f"Error in RAG pipeline: {e}")

        # Finally, we make sure that if the user didn't put anything or cleared session, we reset the page
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": "Hi, I'm a chatbot that can access your vector stores. What would you like to know?",
                }
            ]

        # And we add to the session state the message history
        st.session_state.messages.append(
            {"role": "assistant", "content": resp}
        )
        
        # Display the response
        st.write(resp)
        print(resp)