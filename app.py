import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import ollama
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith traking

os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot With OpenAI" 
groq_api_key=os.environ["GROQ_API_KEY"]


## define our prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a psycologist you know everything about child psycology and human psycology. act as a facilitator and help users to learn social emotional learning. Don't make a biased dicisions and help user to learn social emotional learning through experiences. If you want then give them some tasks which they can perform at there home or with there friends and can learn social emotional learning."),
        ("user", "Question:{question}")
    ]
)

## create a function to generate a response

# def generate_response(question, api_key, llm, temperature, max_tokens):
#     openai.api_key=api_key
#     llm = ChatOpenAI(model=llm)
#     output_parser = StrOutputParser()
#     chain = prompt | llm | output_parser
#     answer = chain.invoke({'question': question})
#     return answer

## creating a function to generate response by om open source models

def generate_response_new(question, temperature, max_tokens):
    llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant", temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer
    

## Title of the app

st.title("Sochu Beta-1.0.0 Using Groq + LLaMA3")
st.caption("🧠This model is still learning and gathering data from various sources..")
st.metric(label="GPU-Temp", value="~32 °C", delta="2 °C")
## sidebar for settings

st.sidebar.title("Settings")
# api_key = st.sidebar.text_input("Enter your OpenAI API Key here:", type="password")

## dropdown to select openAI models

# llm = st.sidebar.selectbox("Select an OpenAI model", ["gpt4-o", "gpt-4-turbo", "gpt-4"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## main interface for user input

st.write("Go ahead and ask any question")
user_input = st.text_input("you:")

if user_input:
    response = generate_response_new(user_input, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")




