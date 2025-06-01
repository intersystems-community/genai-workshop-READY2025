# Import the Python libraries that will be used for this app.
# Libraries of note:
# Streamlit, a Python library that makes it easy to create and share beautiful, custom web apps for data science and machine learning.
# ChatOpenAI, a class that provides a simple interface to interact with OpenAI's models.
# LangGraph for building stateful, multi-actor applications with LLMs
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_iris import IRISVector
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence, TypedDict
import operator
import requests
import json

# Import dotenv, a module that provides a way to read environment variable files
from dotenv import load_dotenv

load_dotenv(override=True)

# Define the IRIS connection
username = "_SYSTEM"
password = "SYS"
hostname = "IRIS"
port = 1972
namespace = "IRISAPP"

# Create the connection string for the IRIS connection
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

# Create an instance of embeddings
embeddings = FastEmbedEmbeddings()

# Define the healthcare collection in the IRIS vector store
HEALTHCARE_COLLECTION_NAME = "case_reports"

# Create an instance of IRISVector
db2 = IRISVector(
    embedding_function=embeddings,
    dimension=384,
    collection_name=HEALTHCARE_COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage | ToolMessage], operator.add]

# Vector Search Tool
class VectorSearchTool(BaseTool):
    name: str = "vector_search"
    description: str = "Search the healthcare vector database for relevant case reports and medical information. Input should be a search query string."
    
    def _run(self, query: str) -> str:
        """Search the vector database for relevant documents."""
        try:
            docs_with_score = db2.similarity_search_with_score(query)
            
            if not docs_with_score:
                return "No relevant documents found in the vector database."
            
            results = []
            for i, (doc, score) in enumerate(docs_with_score[:3]):
                content = doc.page_content
                results.append(f"Document {i+1} (relevance score: {score:.3f}):\n{content}\n")
            
            return "\n".join(results)
        
        except Exception as e:
            return f"Error searching vector database: {str(e)}"

# Email Tool
class EmailTool(BaseTool):
    name: str = "send_email"
    description: str = "Send an email notification. Input should be a JSON string with keys: to, subject, message, html (optional). User must provide an email address."
    
    def _run(self, input_data: str) -> str:
        """Send an email using the provided API endpoint."""
        try:
            # Parse JSON input
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                # If not JSON, treat as simple message
                data = {"message": input_data}
            
            if data.get("to","")=="":
                return f"❌ Error sending email: recipient is required"

            # Default values
            email_data = {
                "to": data.get("to", ""),
                "subject": data.get("subject", "Message from Vector Search App"),
                "message": data.get("message", ""),
                "html": data.get("html", "")
            }
            
            # Send POST request to the email API
            url = "https://g7jisuypzsugoopz4in4yyqfza0igivb.lambda-url.ap-southeast-2.on.aws/"
            headers = {'Content-Type': 'application/json'}
            
            response = requests.post(url, headers=headers, json=email_data, timeout=30)
            
            if response.status_code == 200:
                return f"✅ Email sent successfully to {email_data['to']} with subject: '{email_data['subject']}'"
            else:
                return f"❌ Failed to send email. Status code: {response.status_code}"
        
        except Exception as e:
            return f"❌ Error sending email: {str(e)}"

# Initialize tools
tools = [VectorSearchTool(), EmailTool()]
tool_node = ToolNode(tools)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful healthcare assistant that can search medical database and send email notifications.

Always search the database first when users ask for medical information.
When sending emails, be professional and include relevant medical findings. You must confirm user's email address with the user if not provided.
Remember the conversation context and refer to previous searches when appropriate."""),
    MessagesPlaceholder(variable_name="messages"),
])

# Initialize LLM
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Define the agent node
def call_model(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(prompt.format_messages(messages=messages))
    return {"messages": [response]}

# Define conditional logic for routing
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # If LLM makes a tool call, continue to tools
    if last_message.tool_calls:
        return "tools"
    # Otherwise, end the conversation
    return END

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    }
)

# Add edge from tools back to agent
workflow.add_edge("tools", "agent")

# Initialize memory
memory = MemorySaver()

# Compile the graph
app = workflow.compile(checkpointer=memory)

# Streamlit UI
st.header("🏥 READY 2025 Agentic Healthcare Assistant")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Dataset selection
    ## choose_dataset = st.radio(
    ##    "Choose an IRIS collection:", ("Case Reports", "Encounters"), index=0
    ##)
    
    # Show explanation toggle
    explain = st.radio("Show reasoning steps?:", ("Yes", "No"), index=0)
    
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        st.success("Thread cleared!")
    

    st.write("- *Search for trauma treatment cases*")
    st.write("- *Find patients with lung diseases and email me details in HTML format*")
    st.write("- *Email me a summary of this chat*")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# Handle user input
if user_input := st.chat_input("Ask about medical cases or request actions..."):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    # Prepare thread configuration
    config = {"configurable": {"thread_id": "thread_id"}}
    
    with st.chat_message("assistant"):
        # Show processing status
        with st.status("🔄 Processing with Agentic AI...", expanded=False) as status:
            try:
                # Convert session messages to LangChain format
                messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))
                
                # Create initial state
                initial_state = {"messages": messages}
                
                # Run the graph
                result = app.invoke(initial_state, config=config)
                
                # Get the final response
                final_message = result["messages"][-1]
                response_content = final_message.content
                
                status.update(label="✅ Processing complete!", state="complete")
                
            except Exception as e:
                response_content = f"❌ Error: {str(e)}"
                status.update(label="❌ Error occurred", state="error")
        
        # Display the response
        st.write(response_content)
        
        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        
        # Show reasoning steps if requested
        if explain == "Yes":
            with st.expander("🔍 LangGraph Execution Details"):
                st.write("**Graph Execution Flow:**")
                
                # Show the message flow
                for i, msg in enumerate(result["messages"]):
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        st.write(f"**Step {i+1}: Tool Calls**")
                        for tool_call in msg.tool_calls:
                            st.write(f"- Tool: `{tool_call['name']}`")
                            st.write(f"- Input: `{tool_call['args']}`")
                    elif isinstance(msg, ToolMessage):
                        st.write(f"**Step {i+1}: Tool Result**")
                        st.write(f"- Content: {msg.content[:200]}...")
                    elif hasattr(msg, 'content'):
                        if msg.content and len(msg.content) > 10:
                            st.write(f"**Step {i+1}: Agent Response**")
                            st.write(f"- Response: {msg.content[:200]}...")
