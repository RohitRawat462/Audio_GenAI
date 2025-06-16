import streamlit as st
import os
import tempfile
import google.generativeai as genai
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="Teacher Lecture Evaluator", layout="centered")

# API keys from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Supported languages (aligned with Gemini capabilities)
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

# Function to transcribe audio using Gemini 2.5 Pro
def transcribe_with_gemini(audio_path: str, language: str) -> Optional[str]:
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
        # Upload audio file
        audio_file = genai.upload_file(audio_path)
        # Prompt for transcription
        # prompt = f"""
        # Transcribe this Hindi classroom audio (grades 1–5, phonics/vocabulary lesson in a noisy environment) into text. Format the output with:
        # - Timestamps (e.g., [00:01]) every 5 seconds.
        # - Speaker labels (e.g., Teacher, Student A, Student B) inferred from context (e.g., questions like "ये क्या है?" indicate teacher).
        # - Correct phonics terms (e.g., "गमला", "गाड़ी", "घड़ी", "कोमल", "चिड़िया", "फूल", "गजल", "घर", "मात्रा").
        # - Add punctuation for readability.
        # - Ignore background noise or irrelevant chatter.

        # Language: {language}
        # """

        prompt = f"""
          You are a highly accurate audio transcription expert.

          Your ONLY task is to transcribe the given audio into a **single clean paragraph** of text. The paragraph should:

        - Contain no headings, bullet points, or metadata.
        - Exclude any formatting instructions or explanations.
        - NOT include any introduction or closing statements.

        Just return the **pure transcription** as a plain paragraph, nothing else.

        Language of the audio: {language}
        Begin transcription now:
        
        """
        response = model.generate_content([prompt, audio_file])
        transcription = response.text
        if not transcription:
            st.error("No transcription text received from Gemini.")
            return None
        return transcription
    except Exception as e:
        st.error(f"Gemini transcription failed: {str(e)}")
        return None
    finally:
        # Clean up uploaded file
        try:
            genai.delete_file(audio_file.name)
        except:
            pass

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
st.markdown("Upload and transcribe classroom audio files using Gemini, then analyze teaching performance with OpenAI. Ideal for noisy classroom environments. Supports MP3, WAV, M4A (up to 100MB).")

# Input section
with st.container():
    st.header("Upload Audio File")
    uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a", "mpeg", "mp4"])
    language = st.selectbox("Select Language", [lang["name"] for lang in languages], index=3)  # Default to Hindi
    language_code = next(lang["code"] for lang in languages if lang["name"] == language)

# Button 1: Transcribe
with st.container():
    if st.button("Transcribe"):
        if not uploaded_file:
            st.error("Please upload a valid audio file.")
        else:
            file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
            if file_size > 100:
                st.error("File size exceeds 100MB. Please upload a smaller file or split the audio.")
            else:
                with st.spinner("Transcribing audio with Gemini..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_file_path = tmp_file.name

                    # Transcribe with Gemini
                    transcription_text = transcribe_with_gemini(tmp_file_path, language_code)
                    if transcription_text:
                        st.session_state.transcription_text = transcription_text
                        st.header("Transcription Output")
                        st.write(transcription_text)

                    # Clean up temporary file
                    os.unlink(tmp_file_path)

# Button 2: Run Analysis
with st.container():
    if st.button("Run Analysis"):
        if not st.session_state.transcription_text:
            st.error("No transcription available. Please click 'Transcribe' first.")
        else:
            with st.spinner("Analyzing lecture with OpenAI..."):
                analysis = analyze_lecture(st.session_state.transcription_text)
                st.header("Lecture Analysis")
                st.markdown(analysis)