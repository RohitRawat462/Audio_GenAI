import streamlit as st
import requests
import time
import json
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="Teacher Lecture Evaluator", layout="centered")

# FAF AI API credentials
API_URL = "https://queue.fal.run/fal-ai/whisper"

# OpenAI API key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FAF_KEY = os.getenv("FAF_KEY")

# Supported languages
languages = [
    {"name": "Tamil", "code": "ta"},
    {"name": "Telugu", "code": "te"},
    {"name": "Malayalam", "code": "ml"},
    {"name": "Hindi", "code": "hi"},
    {"name": "Bengali", "code": "bn"},
    {"name": "Urdu", "code": "ur"},
    {"name": "English", "code": "en"},
]

# Initialize session state
if "response_url" not in st.session_state:
    st.session_state.response_url = None
if "transcription_text" not in st.session_state:
    st.session_state.transcription_text = None

# Function to call FAF AI Whisper API to get response URL
def transcribe_audio(audio_url: str, language: str, task: str = "transcribe") -> Optional[str]:
    headers = {
        "Authorization": f"key {FAF_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "audio_url": audio_url,
       # "language": language,
        "task": task,
        "chunk_level": "segment",
        "version": "3",
        "batch_size": 64,
        "num_speakers": None
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        response_url = response_data.get("response_url")
        
        if not response_url:
            st.error("No response URL received from the API.")
            return None
        
        st.session_state.response_url = response_url
        return response_url
    
    except requests.RequestException as e:
        st.error(f"API Request Failed: {str(e)}")
        return None

# Function to poll response URL for transcription text
def get_transcription_result(response_url: str) -> Optional[str]:
    headers = {
        "Authorization": f"key {FAF_KEY}",
        "Content-Type": "application/json",
    }
    
    try:
        for _ in range(30):  # Max 30 seconds polling
            result_response = requests.get(response_url, headers=headers)
            result_response.raise_for_status()
            result_data = result_response.json()
            
            if result_data.get("text"):
                return result_data["text"]
            elif "detail" in result_data and "Failed to get remote file properties" in result_data["detail"]:
                st.error(f"API Error: {result_data['detail']}")
                return None
            time.sleep(1)
        
        st.error("Transcription timed out.")
        return None
    
    except requests.RequestException as e:
        st.error(f"API Request Failed: {str(e)}")
        return None

# Function to analyze transcription using LangChain and OpenAI
def analyze_lecture(transcription: str) -> Optional:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    
    prompt_template = PromptTemplate(
    input_variables=["transcription"],
    template="""
    You are an expert in educational pedagogy and assessment, tasked with evaluating a teacher's classroom lecture based on its transcription. Your goal is to produce a professional, evidence-based report that redefines teacher evaluation by offering unparalleled depth, fairness, and actionable insights. This report will serve as a transformative tool for educators and administrators to enhance teaching quality and address common evaluation challenges, such as subjective bias and lack of specific feedback.

    Analyze the transcription across the following five dimensions, considering diverse teaching styles and classroom contexts (e.g., subject matter, student age, cultural factors):

    1. **Vocal Delivery and Presence**: Evaluate tone, pacing, volume, and vocal expressiveness. Does the teacher project confidence and maintain student attention through vocal dynamics? Consider clarity for diverse learners.
    2. **Clarity and Accessibility**: Assess the use of clear, concise language, avoidance of filler words, and explanation of complex terms. Is the content accessible to all students, including those with varying linguistic or cognitive abilities?
    3. **Engagement and Interaction**: Analyze strategies for student engagement, such as questions, discussions, storytelling, or humor. Does the teacher foster active participation and sustain student interest?
    4. **Content Organization and Flow**: Evaluate the lecture’s structure, including clear objectives, logical progression, and effective transitions. Is the content cohesive and aligned with learning goals?
    5. **Inclusivity and Adaptability**: Assess efforts to create an inclusive environment, such as addressing diverse perspectives, using culturally relevant examples, or adapting to student needs. Does the teacher promote equity in participation?

    Produce a detailed report with the following structure:
    - **Overall Teaching Effectiveness Score**: A weighted average score (1–10, rounded to one decimal place), with weights: Vocal Delivery (20%), Clarity (25%), Engagement (25%), Content Organization (20%), Inclusivity (10%). Provide a brief justification for the overall score.
    - **Dimension Scores and Analysis**: For each dimension, provide:
      - A score (1–10).
      - A concise, evidence-based explanation citing specific examples from the transcription.
      - A brief note on how the dimension impacts student learning outcomes.
    - **Notable Strengths**: Highlight 3–4 specific strengths, emphasizing unique or exemplary practices that enhance teaching quality. Use examples from the transcription.
    - **Areas for Growth**: Identify 2–3 areas for improvement, prioritized by their potential to enhance student learning. Avoid vague critiques; use evidence from the transcription.
    - **Tailored Professional Development Recommendations**: Provide 3–4 specific, actionable strategies for improvement, tailored to the teacher’s performance and context. Include practical examples (e.g., specific techniques, phrasing, or activities) and explain how each addresses an area for growth.
    - **Contextual Considerations**: Briefly note any contextual factors (e.g., subject complexity, classroom dynamics) that may influence the evaluation, ensuring fairness and nuance.
    - When giving score add total score at the end of each section, e.g. "Vocal Delivery: 8.5/10".
    Ensure the report is professional, concise, and structured with clear headings and bullet points for readability. Use academic language suitable for educational administrators. Avoid generic feedback; every point must be grounded in the transcription. Strive for fairness by recognizing diverse teaching approaches and avoiding bias toward specific styles.

    Transcription:
    {transcription}
    """
)
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    result = chain.run(transcription=transcription)
    return result

# Streamlit UI
st.title("Teacher Lecture Evaluator")
st.markdown("Transcribe and analyze classroom lectures to evaluate teaching performance using AI.")

# Input section (Row 1)
with st.container():
    st.header("Input Audio")
    audio_url = st.text_input("Enter Audio URL (mp3, mp4, mpeg, mpga, m4a, wav, webm)", "")
    # language = st.selectbox("Select Language", [lang["name"] for lang in languages], index=6)  # Default to English
    # language_code = next(lang["code"] for lang in languages if lang["name"] == language)

# Button 1: Run (Row 2)
with st.container():
    if st.button("Run"):
        if not audio_url:
            st.error("Please enter a valid audio URL.")
        else:
            with st.spinner("Initiating transcription..."):
                response_url = transcribe_audio(audio_url, "en")  # Default to English for now
                if response_url:
                    st.write(f"Transcription request initiated. Response URL: {response_url}")
                    st.success("Transcription request initiated. Click 'Get Text' to retrieve the transcription.")

# Button 2: Get Text (Row 3)
with st.container():
    if st.button("Get Text"):
        if not st.session_state.response_url:
            st.error("No transcription request found. Please click 'Run' first.")
        else:
            with st.spinner("Retrieving transcription..."):
                transcription_text = get_transcription_result(st.session_state.response_url)
                if transcription_text:
                    st.session_state.transcription_text = transcription_text
                    st.header("Transcription Output")
                    st.write(transcription_text)

# Button 3: Run Analysis (Row 4)
with st.container():
    if st.button("Run Analysis"):
        if not st.session_state.transcription_text:
            st.error("No transcription available. Please click 'Get Text' first.")
        else:
            with st.spinner("Analyzing lecture..."):
                analysis = analyze_lecture(st.session_state.transcription_text)
                st.write(analysis)
        

# # Footer
# st.markdown("---")
