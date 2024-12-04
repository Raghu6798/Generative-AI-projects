from langchain_groq import ChatGroq  
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from sympy import sympify, SympifyError, symbols, integrate
import streamlit as st 
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config with .ico file
st.set_page_config(
    page_title="Text to Math Problems Solver and Data Search Assistant",
    page_icon=""
)

# Initialize LLM API with Groq model
groq_api = os.getenv("GROQ_API")
llm = ChatGroq(model="Gemma2-9b-It", api_key=groq_api)

# Validate mathematical expression
def validate_expression(expression):
    try:
        sympify(expression)
        return True
    except SympifyError:
        return False

# Safe math chain execution
def safe_run_math_chain(expression):
    if validate_expression(expression):
        try:
            response = math_chain.run(expression)
            return response
        except ValueError as e:
            return f"Error: {str(e)}. Please ensure the expression is valid."
    else:
        return "Invalid input. Please provide a valid mathematical expression."

# Solve expression using sympy
def solve_with_sympy(expression):
    x = symbols('x')
    try:
        result = str(integrate(expression, x))
        return f"Result using sympy: {result}"
    except Exception as e:
        return f"Sympy error: {str(e)}"

# Initialize Wikipedia API and math tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet to find various information."
)

math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math-related questions. Only input mathematical expressions."
)

# Set up prompt template
prompt = """
You are a highly skilled mathematical assistant tasked with solving complex mathematical problems. Break down the problem step-by-step, and provide clear explanations for each step as you progress toward the solution. When applicable, include intermediate steps and final answers with justifications.
Question: {question}
Answer:
"""
prompt_template = PromptTemplate(template=prompt, input_variables=["question"])

chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for answering logical and reasoning questions."
)

# Initialize agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Session state to store chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I am a Math Problem-solving chatbot who can answer all your math-related queries."}
    ]

# Display chat messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Function to generate response
def generate_response(user_question):
    response = assistant_agent.invoke({"input": user_question})
    return response

# Input field for math problem
question = st.text_input("Enter your math problem here:")

# Button to process and display the answer
if st.button("Find my answer"):
    if question:
        with st.spinner("Generating a response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run({"input": question}, callbacks=[st_cb])

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("## Response")
            st.markdown(response)
    else:
        st.warning("Please enter the question")  
