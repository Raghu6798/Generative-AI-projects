import streamlit as st
import pandas as pd
import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq

def main():
    st.sidebar.title('Customization')

    # Prompt user to enter API key if not in the environment variables
    api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key  # Set it temporarily for this session

    # Ensure the API key is set before proceeding
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("API key is required to proceed. Please enter your Groq API key.")
        return

    # Choose model option
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )

    # Initialize the LLM
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=groq_api_key, 
        model=model,
        provider="groq"
    )

    # Streamlit UI
    st.title('CrewAI Machine Learning Assistant')
    st.markdown("""
    The CrewAI Machine Learning Assistant is designed to guide users through defining, assessing, 
    and solving machine learning problems. It leverages a team of AI agents, each with a specific role, 
    to clarify the problem, evaluate data, recommend suitable models, and generate starter Python code.
    """, unsafe_allow_html=True)

    # Display the Groq logo
    st.image(r'C:\Users\Raghu\Downloads\private KEY\ETLpipeline\llm-vectors-unstructured\CrewAI\groqcloud_darkmode.png')

    # Define agents
    Problem_Definition_Agent = Agent(
        role='Problem_Definition_Agent',
        goal="Clarify the machine learning problem, identifying type (e.g., classification, regression).",
        backstory="Expert in defining machine learning problems, providing clear project foundations.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Data_Assessment_Agent = Agent(
        role='Data_Assessment_Agent',
        goal="Evaluate data quality, suitability, and suggest preprocessing steps.",
        backstory="Expert in data evaluation, guiding dataset preparation for ML models.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Model_Recommendation_Agent = Agent(
        role='Model_Recommendation_Agent',
        goal="Recommend suitable machine learning models based on the problem definition and data.",
        backstory="Expert in ML algorithms, recommending models suitable for the problem and data.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Starter_Code_Generator_Agent = Agent(
        role='Starter_Code_Generator_Agent',
        goal="Generate starter Python code, including data loading, model definition, and training loop.",
        backstory="Code generator for starter templates, giving users a head start on their project.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # User inputs
    user_question = st.text_input("Describe your ML problem:")
    data_upload = False
    uploaded_file = st.file_uploader("Upload a sample .csv of your data (optional)")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file).head(5)
            data_upload = True
            st.write("Data successfully uploaded and read as DataFrame:")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading the file: {e}")

    if user_question:
        # Define tasks
        task_define_problem = Task(
            description="Clarify and define the machine learning problem, identifying problem type.",
            agent=Problem_Definition_Agent,
            expected_output="Clear problem definition."
        )

        task_assess_data = Task(
            description="Evaluate the data quality and suggest preprocessing steps.",
            agent=Data_Assessment_Agent,
            expected_output="Assessment of data quality and preprocessing suggestions."
        ) if data_upload else Task(
            description="Consider a hypothetical dataset for user's ML problem.",
            agent=Data_Assessment_Agent,
            expected_output="Hypothetical dataset for the user's problem."
        )

        task_recommend_model = Task(
            description="Recommend suitable models with rationale.",
            agent=Model_Recommendation_Agent,
            expected_output="List of suitable ML models and rationale."
        )

        task_generate_code = Task(
            description="Generate starter Python code including model recommendation.",
            agent=Starter_Code_Generator_Agent,
            expected_output="Python code snippets for data handling and model training."
        )

        # Crew
        crew = Crew(
            agents=[Problem_Definition_Agent, Data_Assessment_Agent, Model_Recommendation_Agent, Starter_Code_Generator_Agent],
            tasks=[task_define_problem, task_assess_data, task_recommend_model, task_generate_code],
            verbose=True
        )

        result = crew.kickoff()
        st.write(result)


if __name__ == "__main__":
    main()
