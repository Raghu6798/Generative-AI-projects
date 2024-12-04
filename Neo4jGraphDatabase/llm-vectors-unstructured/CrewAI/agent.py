from crewai import Agent, LLM  # Importing Agent and LLM from crewai
from tools import tool
from dotenv import load_dotenv
from langchain.llms import ollama

import os

load_dotenv()


# Initialize the ChatGroq model using the API key

print(llm.invoke("Hi!"))
# Define the Content Planner agent
planner = Agent(
    llm=llm,
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article about the topic: {topic}."
              " You collect information that helps the audience learn something "
              "and make informed decisions. Your work is the basis for the "
              "Content Writer to write an article on this topic.",
              tools=[tool],
    allow_delegation=False,
    verbose=True
)

# Define the Content Writer agent
writer = Agent(
    llm=llm,
    role="Content Writer",
    goal="Write an insightful and factually accurate opinion piece about the topic: {topic}",
    backstory="You're working on writing a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of the Content Planner, who provides an outline "
              "and relevant context about the topic. You follow the main objectives and "
              "direction of the outline, as provided by the Content Planner. "
              "You also provide objective and impartial insights and back them up with information "
              "provided by the Content Planner. You acknowledge in your opinion piece "
              "when your statements are opinions as opposed to objective statements.",
              tools=[tool],
    allow_delegation=False,
    max_iter=2,
    verbose=True
)

# Define the Editor agent
editor = Agent(
    llm=llm,    
    role="Editor",
    goal="Edit a given blog post to align with the writing style of the organization.",
    backstory="You are an editor who receives a blog post from the Content Writer. "
              "Your goal is to review the blog post to ensure that it follows journalistic best practices, "
              "provides balanced viewpoints when providing opinions or assertions, "
              "and avoids major controversial topics or opinions when possible.",
              tools=[tool],
    allow_delegation=False,
    max_iter=2,
    verbose=True
)
