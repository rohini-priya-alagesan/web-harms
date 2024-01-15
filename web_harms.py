import streamlit as st
import pandas as pd
import os
#import fitz
import openai
from datetime import datetime
import base64
import matplotlib.pyplot as plt
import numpy as np
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
#import PyMuPDFb



# Set page configuration
st.set_page_config(page_title="My Webpage", layout="wide")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style.css")

st.markdown("""
    <style>
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }

        body {
            margin: 0;
            padding: 0;
            height: 100vh;
            background-color: lightblue;
            animation: fadeIn 1s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)

os.environ["OPENAI_API_KEY"] = "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"
OPENAI_API_KEY = "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"


# Read data from a file into a DataFrame
data = pd.read_csv('ACC_benchmark_questions.csv')  # Replace with your actual file path

unique_values_main = data['harm classifier'].unique()

# Title at the top middle
st.markdown("<h1 style='text-align: center;'>Welcome to Ofcoms Harms Library!</h1>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Hello and welcome! At Ofcom's Harms Library, Our curated collection covers a range of harms, each with their unique characteristics, aims, and impacts!</h4>", unsafe_allow_html=True)












search_options = [' 🔍 Search for Harm'] + list(data['harm classifier'].unique())

st.write("<h3 style='text-align: left;'>Search or explore list of all harms below:</h3>", unsafe_allow_html=True)
search_keyword = st.selectbox("",search_options)

if search_keyword:
    if search_keyword != 'Please Choose an option':
        # Use a custom lambda function for case-insensitive matching
        filtered_data = data[data.apply(lambda row: any(str(cell).lower().find(search_keyword.lower()) != -1 for cell in row), axis=1)]

        if not filtered_data.empty:
            # Assuming you want to extract values from the first row of the filtered data
            row_data = filtered_data.iloc[0]

            # Extract specific column values into variables
            harm_value = row_data['harm classifier']
            harm_category = row_data['harm category']
            descriptor_value = row_data['classifier aim']
            architecture = row_data['AI/ML Architecture']
            potential = row_data["potential APIS"]
            type_value = row_data["input type"]
            reference = row_data["Name of reference"]
            accuracy = row_data["Accuracy achieved"]
            dataset = row_data["dataset"]
            url = row_data["link reference"]
            accuracy_text = row_data["accuracy text"]

            # Create a layout with columns
            col1, col2, col3 = st.columns(3)

            # Display variables in separate expanders with content expanded by default
            with col1.expander(r"$\Large{\underline{\textbf{\textsf{Harm Selected}}}}$", expanded=True):
                st.markdown(f"<p style='font-size:20px;'>{harm_value}</p>", unsafe_allow_html=True)

            with col1.expander(r"$\Large{\underline{\textbf{\textsf{Harm Category}}}}$", expanded=True):
                st.markdown(f"<p style='font-size:20px;'>{harm_category}</p>", unsafe_allow_html=True)

                if harm_category == "Harmful but not illegal":
                    st.image("harmfull.png", width=120)
                elif harm_category == "Illegal Content":

                    st.image("illegall.png", width=120)
                else:
                    st.image("NSFK.png", width=120)

            with col2.expander(r"$\Large{\underline{\textbf{\textsf{Classifier aim}}}}$", expanded=True):
                st.markdown(f"<p style='font-size:20px;'>{descriptor_value}</p>", unsafe_allow_html=True)


            with col3.expander(r"$\Large{\underline{\textbf{\textsf{Architecture}}}}$", expanded=True):
                st.markdown(f"<p style='font-size:20px;'>{architecture}</p>", unsafe_allow_html=True)


            with col1.expander(r"$\Large{\underline{\textbf{\textsf{Potential Off the Shelf API's}}}}$", expanded=True):
                st.markdown(f"<p style='font-size:20px;'>{potential}</p>", unsafe_allow_html=True)


            with col2.expander(r"$\Large{\underline{\textbf{\textsf{Input Type}}}}$", expanded=True):
                st.markdown(f"<p style='font-size:20px;'>{type_value}</p>", unsafe_allow_html=True)


            with col3.expander(r"$\Large{\underline{\textbf{\textsf{Reference}}}}$", expanded=True):
                st.markdown(f"<p style='font-size:20px;'>{reference}</p>", unsafe_allow_html=True)




            with col2.expander(r"$\Large{\underline{\textbf{\textsf{Accuracy Scored}}}}$", expanded=True):
                st.write(accuracy)   
                progress_bar_class = "progress-bar"
                progress_bar = f"<progress class='{progress_bar_class}' value='{accuracy}' max='100'></progress>"

                st.markdown(f"<div class='progress-container'>{progress_bar}</div>", unsafe_allow_html=True)

            with col1.expander(r"$\Large{\underline{\textbf{\textsf{Dataset}}}}$", expanded=True):
                st.markdown(f"<p style='font-size:20px;'>{dataset}</p>", unsafe_allow_html=True)

            with col3.expander(r"$\Large{\underline{\textbf{\textsf{Reference URL}}}}$", expanded=True):
                st.markdown(f"<p style='font-size:20px;'>{url}</p>", unsafe_allow_html=True)

            with col3.expander(r"$\Large{\underline{\textbf{\textsf{Accuracy}}}}$", expanded=True):
                st.markdown(f"<p style='font-size:20px;'>{accuracy_text}</p>", unsafe_allow_html=True)

            user_input_key = "user_input"
         #   search_button_key = "search_button"
          #  user_input, search_button = st.text_input("Ask Our AI about this Paper/Harm:", "What is this paper about?" , max_chars=200, key=user_input_key), st.button("Search", key=search_button_key)

    













        


# Main data manipulation
st.markdown("<hr>", unsafe_allow_html=True)  # Separator line
st.markdown("<h1 style='text-align: center;'>Explore all the harms.</h1>", unsafe_allow_html=True)

# Arrange expanders in columns of 3
cols = st.columns(3)

import streamlit as st

# Display raw data
cols = st.columns(2)



for i, value in enumerate(unique_values_main):
    # Set expanded=True for the first 3 expanders
    expander = cols[i % 2].expander(r"$\large{\textbf{\textsf{" + f"{value}" + "}}}$", expanded=(i < 2))

    
    with expander:
        row_data = data[data['harm classifier'] == value].iloc[0]
        harm_value, descriptor_value,harm_category, architecture, potential, type_value, reference, accuracy, dataset, url, atext = row_data['harm classifier'], row_data['classifier aim'],row_data['harm category'], row_data['AI/ML Architecture'], row_data["potential APIS"], row_data["input type"], row_data["Name of reference"], row_data["Accuracy achieved"], row_data["dataset"], row_data["link reference"], row_data["accuracy text"]




        # Display the data in two columns
        col3, col4 = st.columns(2)
        
        with col3:
          #  st.markdown(f"<p style='font-size: 50px; font-weight: bold;text-decoration: underline;'>Harm: \n\n{harm_value}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>Harm Category  \n\n{harm_category}</p>", unsafe_allow_html=True)
            if harm_category == "Harmful but not illegal":
               st.image("harmfull.png", width=160)
            elif harm_category == "Illegal Content":
                st.image("illegall.png", width=150)
            else:
                st.image("NSFK.png", width=150)
          #  elif harm_category == "Illegal Content":
          #      st.image("illegal.png", width=400)
            st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>Classifier Aim \n\n{descriptor_value}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>Architecture \n\n{architecture}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 15px; font-weight: bold;'>Potential API \n\n{potential}</p>", unsafe_allow_html=True)     
        with col4:
            st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>Benchmark  Reference \n\n{reference}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>Reference URL \n\n{url}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 18px; font-weight: bold;'>Input Type: \n\n{type_value}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 15px; font-weight: bold;'>Dataset \n\n{dataset}</p>", unsafe_allow_html=True) 
                        # Set the static accuracy value
          #  accuracy_value = 75  # You can change this value as needed

            # Define the style for the progress bar
            progress_bar_style = """
                <style>
                    .progress-container {
                        display: flex;
                        align-items: center;
                    }
                    .progress-bar {
                        width: 80%;
                        height: 60px; /* Adjust the height to make it thicker */
                        margin-right: 10px;
                    }
                    .progress-label {
                        font-size: 16px;
                        font-weight: bold;
                    }
                    .low-accuracy {'.
                        color: red;
                    }
                </style>
            """

            # Display the styled progress bar with tags and progress text
            st.markdown(progress_bar_style, unsafe_allow_html=True)

         #   st.markdown(f"<p style='font-size: 15px; font-weight: bold;'> \n\n{atext}</p>", unsafe_allow_html=True)

            st.markdown(f"<p style='font-size: 15px; font-weight: bold;'>Accuracy \n\n{atext}</p>", unsafe_allow_html=True)
        
            progress_bar_class = "progress-bar"
            progress_bar = f"<progress class='{progress_bar_class}' value='{accuracy}' max='100'></progress>"

            st.markdown(f"<div class='progress-container'>{progress_bar}</div>", unsafe_allow_html=True) 

            

            



       # Input box and search button in a horizontal layout
        user_input_key = f"user_input_{i}"
        search_button_key = f"search_button_{i}"
        st.markdown("<hr>", unsafe_allow_html=True)  # Separator line
        user_input, search_button = st.text_input("Ask Our AI about this Benchmark:","What is this paper about?",max_chars=300,key=user_input_key), st.button("Search", key=search_button_key)
        

        # Check if the search button is clicked or a common question is selected

        # Initialize session_state
        if 'last_search_time' not in st.session_state:
            st.session_state.last_search_time = datetime.min


        time_limit_seconds = 10
        current_time = datetime.now()
        time_difference = current_time - st.session_state.last_search_time

        if search_button and time_difference.total_seconds() >= time_limit_seconds:
            st.session_state.last_search_time = current_time 




            #IMPORTING SELECTED PDF
       

            # Retrieve PDF file
            pdf_filename = f"{value}.pdf"  # Assuming the PDF file has the same name as the harm classifier
            pdf_path = os.path.join(r"C:\Users\umaralinew\Downloads\web-harms-website\web-harms-website\web-harms\web-harms", pdf_filename)

            if os.path.exists(pdf_path):
                reader = PdfReader(pdf_path)

                # read data from the file and put them into a variable called raw_text
                raw_text = ''
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        raw_text += text

                    text_splitter = CharacterTextSplitter(        
                    separator = "\n",
                    chunk_size = 1000,
                    chunk_overlap  = 200,
                    length_function = len,
                )
                    texts = text_splitter.split_text(raw_text)


                    embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002", model="text-embedding-ada-002")


                    docsearch = FAISS.from_texts(texts, embeddings)

                    from langchain.chains.question_answering import load_qa_chain
                    from langchain.chat_models import ChatOpenAI

                   

                    chain = load_qa_chain(ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="map_rerank")
                    query = user_input
                    docs = docsearch.similarity_search(query)
                    answers = []
            
                    with st.spinner("Loading answers..."):
                        for _ in range(3):
                            answer = chain.run(input_documents=docs, question=query)
                            answers.append(answer.strip())



                    for i, answer in enumerate(answers, 1):
                    #   st.write(f"Answer {i}: {answer}")
                       st.markdown(f"<p style='font-size: 15px; font-weight: bold;'>Answer {i}:</p> {answer}", unsafe_allow_html=True) 
                       st.markdown("<hr>", unsafe_allow_html=True)  # Separator line


                    # Create a formatted string with the query, answer, date, and time
                    current_datetime = datetime.now()
                    formatted_content = f"Harm: {harm_value}\n\nClassifier Aim: {descriptor_value}\n\nReference: {reference}\n\nPotential API's': {potential}\n\nArchitecture: {architecture}\n\nInput Type: {type_value}\n\nDataset: {dataset}\n\nAccuracy: {accuracy}\n\nQuery: {query}\n\nAnswer: {answer}\n\nDate: {current_datetime.date()}\n\nTime: {current_datetime.time()}\n\nLink to selected Harm Paper: {url}"

                    # Add a download button for the formatted content
                    download_button_label = "Download Answer"

                    # Use st.download_button to create the download button
                    download_button = st.download_button(
                        label=download_button_label,
                        data=formatted_content,  # Encode the content as bytes
                        key=("download_button"),
                        help="Click to download the answer",
                    )


        elif search_button and time_difference.total_seconds() <= time_limit_seconds:
            st.write(f"Please wait {time_limit_seconds} before searching again!")
