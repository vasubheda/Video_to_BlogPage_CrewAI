from crewai import Agent
from tools import yt_tool
from langchain_groq import ChatGroq
from litellm import completion

from dotenv import load_dotenv
load_dotenv()

import os
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

def gen_completion(prompt):
    response = completion(
        model="groq/Gemma-7b-It",
        messages=[
            {"role":"user","content":prompt}
        ],
        temperature=0.7,
        api_key=groq_api_key
    )
    return response['choices'][0]['message']['content']

## Create a senior blog content researcher 

blog_researcher=Agent(
    role='Blog researcher from Youtube Videos',
    goal='Get the relevant video content for the topic{topic} from Yt channel',
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos of AI, Data Science, Machine Learning and GenAI"
    ),
    tools=[yt_tool],
    allow_delegation=True
)

## Create a senior blog writer agent with YT Tool

blog_writer=Agent(
    role='Blog Writer',
    goal='Narrate the compelling tech stories about the video {topic} from YT Channel',
    verbose=True,
    memory=True,
    backstory=(
        "With a flare of simplifying complex topics, you craft engaging narratives that captivates and educates, bringing new discoveries to light in an accessible manner"
    ),
    llm=gen_completion,
    tools=[yt_tool],
    allow_delegation=False
)