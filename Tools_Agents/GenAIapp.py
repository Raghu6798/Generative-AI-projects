# Initialize the Streamlit application
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import initialize_agent, AgentType

# Set up your tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_arxiv_wrapper)

# Streamlit app title
st.title("🔍 Langchain Search Agent")

# Sidebar for API keys
st.sidebar.title("Settings")
api_keys = st.sidebar.text_input("Enter your Groq API key:", type="password")

# DuckDuckGo search tool
search = DuckDuckGoSearchRun(name="Search")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Input and response
if prompt := st.chat_input(placeholder="What is machine learning?"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Set up language model
    llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", api_key=api_keys, streaming=True)
    tools = [search, arxiv, wiki]

    # Initialize agent
    search_agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
    )

    # Handle the assistant's response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])

        # Append assistant's response to chat history
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.write(response)
