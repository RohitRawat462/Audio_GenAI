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
import openai

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="Teacher Lecture Evaluator", layout="centered")

# OpenAI API key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI Whisper API endpoint
WHISPER_API_URL = "https://api.openai.com/v1/audio/transcriptions"

# Supported languages (same as before, for consistency)
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
if "transcription_text" not in st.session_state:
    st.session_state.transcription_text = None

# Function to transcribe audio using OpenAI Whisper API
def transcribe_audio(audio_file: bytes, language: str, task: str = "transcribe") -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    
    files = {
        "file": ("audio.mp3", audio_file, "audio/mpeg"),
        "model": (None, "whisper-1"),
        "language": (None, language),
        "response_format": (None, "text")
    }
    
    if task == "translate":
        files["task"] = (None, "translate")
    
    try:
        response = requests.post(WHISPER_API_URL, headers=headers, files=files)
        # st.write(f"API Response Status: {response.status_code}")
        # st.write(f"API Response: {response.text}")
        response.raise_for_status()
        transcription_text = response.text
        st.write(response)
        
        if not transcription_text:
            st.error("No transcription received from the API.")
            return None
        
        return transcription_text
    
    except requests.RequestException as e:
        if "Maximum content size limit" in str(e):
            st.error("File size exceeds OpenAI Whisper's 25MB limit. Please upload a smaller file.")
        else:
            st.error(f"API Request Failed: {str(e)}")
        return None

# Function to analyze transcription using LangChain and OpenAI
def analyze_lecture(transcription: str) -> Optional[str]:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    
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
          - A score (1–10, e.g., "Vocal Delivery: 8.5/10").
          - A concise, evidence-based explanation citing specific examples from the transcription.
          - A brief note on how the dimension impacts student learning outcomes.
        - **Notable Strengths**: Highlight 3–4 specific strengths, emphasizing unique or exemplary practices that enhance teaching quality. Use examples from the transcription.
        - **Areas for Growth**: Identify 2–3 specific areas for improvement, prioritized by their potential to enhance student learning. Avoid vague critiques; use evidence from the transcription.
        - **Tailored Professional Development Recommendations**: Provide 3–4 specific, actionable strategies for improvement, tailored to the teacher’s performance and context. Include practical examples (e.g., specific techniques, phrasing, or activities) and explain how each addresses an area for growth.
        - **Contextual Considerations**: Briefly note any contextual factors (e.g., subject complexity, classroom dynamics) that may influence the evaluation, ensuring fairness and nuance.

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
st.markdown("Transcribe and analyze classroom lectures to evaluate teaching performance using OpenAI Whisper. Supports audio files up to 25MB (MP3, MP4, MPEG, MPGA, M4A, WAV, WEBM).")

# Input section (Row 1)
with st.container():
    st.header("Input Audio")
    uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"])
    language = st.selectbox("Select Language", [lang["name"] for lang in languages], index=6)  # Default to English
    language_code = next(lang["code"] for lang in languages if lang["name"] == language)
    task = st.radio("Task", ["Transcribe", "Translate to English"], index=0)

# Button 1: Transcribe (Row 2)
with st.container():
    if st.button("Transcribe"):
        if not uploaded_file:
            st.error("Please upload a valid audio file.")
        else:
            file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
            if file_size > 25:
                st.error("File size exceeds OpenAI Whisper's 25MB limit. Please upload a smaller file or split the audio.")
            else:
                with st.spinner("Transcribing audio..."):
                    audio_data = uploaded_file.getbuffer()
                    transcription_text = transcribe_audio(audio_data, language_code, task.lower())
                    if transcription_text:
                        st.session_state.transcription_text = transcription_text
                        st.header("Transcription Output")
                        st.write(transcription_text)

# Button 2: Run Analysis (Row 3)
with st.container():
    if st.button("Run Analysis"):
        if not st.session_state.transcription_text:
            st.error("No transcription available. Please click 'Transcribe' first.")
        else:
            with st.spinner("Analyzing lecture..."):
                analysis = analyze_lecture(st.session_state.transcription_text)
                st.header("Lecture Analysis")
                st.write(analysis)