from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from openai import OpenAI
import streamlit as st
import os
import PyPDF2
import prompts as pt
import base64

#keys
os.environ["OPENAI_API_KEY"] =st.secrets["OPENAI_API_KEY"]

#CONSTANTS
MODEL="gpt-4o-mini"

# set the openai model
llm = ChatOpenAI(model=MODEL, temperature=0)

# create client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

#Function to extract text from pdf
def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    # read the pdf
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        # extract the text
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            pdf_text += page.extract_text()

    final_text = pdf_text.strip().replace("\n", "")        
    return pdf_text


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

#Function to extract text from an image
def extract_text_from_image(image_path:str):
    
    # Encode the image to base64
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
    model=MODEL,

    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": pt.IMAGE_INSTRUCTIONS,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
    )

    return response.choices[0].message.content



#Function to analyse bias
def bias_analyser(text: str):
    prompt = ChatPromptTemplate.from_template(pt.BIAS_ANALYSER_PROMPT)
    model=llm
    output_parser = StrOutputParser()

    # create the chain
    chain = prompt | model | output_parser

    # get the answer
    answer = chain.invoke({"text":text})

    return answer
