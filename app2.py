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

# AssemblyAI API key from .env
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# OpenAI API key for analysis
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# AssemblyAI API endpoints
ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

# Supported languages (aligned with AssemblyAI capabilities)
languages = [
    {"name": "Tamil", "code": "ta"},
    {"name": "Telugu", "code": "te"},
    {"name": "Malayalam", "code": "ml"},
    {"name": "Hindi", "code": "hi"},
    {"name": "Bengali", "code": "bn"},
    {"name": "Urdu", "code": "ur"},
    {"name": "English", "code": "en_us"},
]

# Initialize session state
if "transcription_text" not in st.session_state:
    st.session_state.transcription_text = None
if "transcription_id" not in st.session_state:
    st.session_state.transcription_id = None

# Function to upload audio to AssemblyAI
def upload_audio(audio_file: bytes) -> Optional[str]:
    return 'https://cdn.assemblyai.com/upload/19cbfca5-ed9b-4cfd-a02e-6ae6b418ce49'
    headers = {
        "Authorization": ASSEMBLYAI_API_KEY,
        "Content-Type": "application/octet-stream"
    }
    try:
        response = requests.post(ASSEMBLYAI_UPLOAD_URL, headers=headers, data=audio_file)
        if response.status_code != 200:
            error_message = response.json().get("error", "Unknown error")
            st.error(f"Audio upload failed: {response.status_code} - {error_message}")
            return None
        return response.json().get("upload_url")
    except requests.RequestException as e:
        st.error(f"Audio upload failed: {str(e)}")
        return None

# Function to transcribe audio using AssemblyAI API
def transcribe_audio(audio_url: str, language: str) -> Optional[str]:
    headers = {
        "Authorization": ASSEMBLYAI_API_KEY,
        "Content-Type": "application/json"
    }

    # Transcription payload based on provided cURL
    payload = {
       
    "audio_url": audio_url,
    # Commented out to avoid duration errors; re-enable if needed after verifying audio duration
    # "audio_end_at": 280000,  # Convert to milliseconds (e.g., 280s clip)
    # "audio_start_from": 10000,  # Convert to milliseconds (e.g., start at 10s)
    # "boost_param": "high",  # Unchanged: Prioritizes word_boost terms
    # "custom_spelling": [
    #     {"from": ["गम्ला", "गमल"], "to": "गमला"},  # Corrects common misspellings
    #     {"from": ["गाड़ी", "गादी"], "to": "गाड़ी"},
    #     {"from": ["घड़ी", "घडी"], "to": "घड़ी"},
    #     {"from": ["कमल"], "to": "कोमल"}  # From sample: "कोमल बन गया"
    # ],  # NEW: Corrects Hindi misspellings for phonics terms
    # "disfluencies": False,  # Unchanged: Excludes filler words (e.g., "उह") for cleaner text
    # "entity_detection": True,  # Unchanged: Identifies entities (e.g., names) for PII redaction
    # "filter_profanity": True,  # Unchanged: Filters inappropriate language
    # "format_text": True,  # Unchanged: Adds capitalization and formatting
     "language_code": language,  # Unchanged: Dynamic (e.g., "hi" for Hindi)
    # "language_confidence_threshold": 0.7,  # Unchanged: Not relevant since language_detection is False
    # "language_detection": False,  # Unchanged: Avoids conflict with language_code
     "multichannel": True,  # Unchanged: Avoids conflict with speaker_labels
    # "punctuate": True,  # Unchanged: Adds punctuation for readability
    # "redact_pii": True,  # Unchanged: Protects sensitive data (e.g., student names)
    # "redact_pii_audio": True,  # Unchanged: Redacts PII in output audio
    # "redact_pii_audio_quality": "mp3",  # Unchanged: Maintains audio quality
    # "redact_pii_policies": [
    #     "person_name",  # CHANGED: Added to redact student/teacher names
    #     "us_social_security_number",
    #     "credit_card_number"
    # ],  # Enhanced to protect classroom-specific PII
    # "redact_pii_sub": "hash",  # Unchanged: Hashes redacted data
   # "speaker_labels": True,  # Unchanged: Enables diarization for teacher vs. students
    #"speakers_expected": 3,  # CHANGED: Increased to 3 to account for multiple students
    # "speech_threshold": 0.4,  # CHANGED: Lowered to capture quieter student responses
    # "dual_channel": False,  # Unchanged: Single-channel audio
    # "word_boost": [
    #     "math manipulatives",
    #     "phonics",
    #     "classroom",
    #     "teacher",
       
    # ]  # Enhanced with Hindi vocabulary for better accuracy

    }

    try:
        # Submit transcription request
        response = requests.post(ASSEMBLYAI_TRANSCRIPT_URL, headers=headers, json=payload)
        if response.status_code != 200:
            error_message = response.json().get("error", "Unknown error")
            st.error(f"Transcription request failed: {response.status_code} - {error_message}")
            return None
        transcription_id = response.json().get("id")
        st.session_state.transcription_id = transcription_id

        # Poll for transcription completion
        poll_url = f"{ASSEMBLYAI_TRANSCRIPT_URL}/{transcription_id}"
        while True:
            poll_response = requests.get(poll_url, headers=headers)
            if poll_response.status_code != 200:
                error_message = poll_response.json().get("error", "Unknown error")
                st.error(f"Polling failed: {poll_response.status_code} - {error_message}")
                return None
            status = poll_response.json().get("status")

            if status == "completed":
                transcription_text = poll_response.json().get("text")
                if not transcription_text:
                    st.error("No transcription text received from AssemblyAI.")
                    return None
                json_data = json.dumps(poll_response.json(), indent=2, ensure_ascii=False)
            # Display the JSON data in a text area
                st.text_area("Poll Response", value=json_data, height=300)
                st.write(audio_url)
                return transcription_text
            elif status == "error":
                st.error(f"Transcription failed: {poll_response.json().get('error')}")
                return None
            time.sleep(5)  # Wait before polling again
    except requests.RequestException as e:
        st.error(f"Transcription request failed: {str(e)}")
        return None

# Function to analyze transcription using LangChain and OpenAI
def analyze_lecture(transcription: str) -> Optional[str]:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, api_key=OPENAI_API_KEY)
    
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
st.markdown("Upload and transcribe classroom audio files using AssemblyAI, then analyze teaching performance. Ideal for noisy classroom environments with speaker diarization. Supports MP3, WAV, M4A, and other formats (up to 100MB).")

# Input section (Row 1)
with st.container():
    st.header("Upload Audio File")
    uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a", "mpeg", "mp4"])
    language = st.selectbox("Select Language", [lang["name"] for lang in languages], index=6)  # Default to English
    language_code = next(lang["code"] for lang in languages if lang["name"] == language)

# Button 1: Transcribe (Row 2)
with st.container():
    if st.button("Transcribe"):
        if not uploaded_file:
            st.error("Please upload a valid audio file.")
        else:
            file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
            if file_size > 100:
                st.error("File size exceeds 100MB. Please upload a smaller file or split the audio.")
            else:
                with st.spinner("Uploading and transcribing audio with AssemblyAI..."):
                    audio_data = uploaded_file.getbuffer()
                    audio_url = upload_audio(audio_data)
                    if audio_url:
                        transcription_text = transcribe_audio(audio_url, language_code)
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
                st.markdown(analysis)