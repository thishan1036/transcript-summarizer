import streamlit as st
import google.generativeai as genai
import pdfplumber
import json



try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-flash-latest')
except Exception as e:
    st.error(f"Error configuring Google Generative AI: {e}")
    model = None

CHUNK_AGENT_PROMPT = """
You are a text-processing specialist. Your only job is to parse an earnings call transcript.
You will be given the full text of an earnings call.

Your task is to identify and segment the text into its two main parts:

The "Prepared Remarks" (the opening speech by management).
The "Questions and Answers" (the Q&A session with analysts).
You must return the output as a single JSON object.
The keys must be prepared_remarks and questions_and_answers.
The values must be the complete, verbatim text of those sections.
Do not include any other sections (like the operator introduction or legal disclaimers).
Do not add any commentary. Your only output must be the valid JSON.

Example Output Format:
{
  "prepared_remarks": "Thank you, operator. Good afternoon, everyone... [full text]...",
  "questions_and_answers": "Thank you. We will now begin the question-and-answer session... [full text]..."
}
"""

ANALYZER_AGENT_PROMPT = """
Your only job and only output must be a single, valid JSON object. Do not add any text, commentary, or explanation before or after the JSON.

Your task is to analyze a chunk of text from a financial report based on the following rules:

You will act as a senior financial analyst.
The JSON you return must have these exact keys: key_numbers, strategic_updates, risk_factors, and red_flags.
The value for each key must be a list of strings (the bullet points you extract).

Extract:
key_numbers: Specific financial figures, percentages, guidance, or hard numbers.
strategic_updates: New products, M&A activity, market expansion, or changes in business focus.
risk_factors: New or newly emphasized risks.
red_flags: Any language that seems unusual, evasive, or overly promotional.
Be concise. Stick only to what the text explicitly states. Ignore boilerplate.
If you find no information for a key, you must return an empty list [].

Example Output (This is the only format you will use):
{
  "key_numbers": [
    "Revenue increased 15% to $10M"
  ],
  "strategic_updates": [],
  "risk_factors": [],
  "red_flags": []
}
"""

SYNTHESIZER_AGENT_PROMPT = """
You are an executive editor at a top-tier financial publication. 
Your only job is to synthesize a collection of analyst notes into a single, high-level executive summary for a busy CEO.

You will be given a list of JSON objects. Each object represents an analysis of a different section of a financial report.
Your task is to review all the notes and write a single, cohesive, 1-page summary.
Do not just list the sections. Synthesize the information. For example, if "Key Numbers" appear in multiple notes, combine them into one coherent section.

Your final output must follow this structure (using Markdown for formatting):

Executive Summary
A 2-3 sentence overview of the most important takeaways from the entire report.

Key Metrics & Guidance
A bulleted list of the most critical numbers (revenue, EPS, guidance, etc.).

Strategic Developments
A bulleted list of key updates (new products, M&A, market changes).

Risks & Red Flags
A bulleted list of the most significant risks and any red flags identified by the analysts.

Be concise and professional. Use clear, direct language. Do not add any commentary or introduction. Your output should be the summary itself.
"""

def clean_json_response(response_text):
    """
    Helper function to clean the json artifacts from the AI's response.
    """
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1

    if json_start != -1 and json_end != -1:
        clean_text = response_text[json_start:json_end] 
        return clean_text
    else:
        return response_text

def call_gemini(prompt, data_to_process, retry_count=2):
    """
    A single, reliable function to make the Gemini API call with retries.
    """
    if model is None:
        return ("Error: Google Generative AI model is not configured.")

    full_prompt = f"{prompt}\n\nHere is the text to process:\n\n{data_to_process}"

    for attempt in range(retry_count):
        try:
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            st.error(f"API Error (Attempt {attempt + 1}/{retry_count}): {e}")
            if "quota" in str(e).lower():
                st.error("Quota exceeded. Please check your Google AI billing.")
                return None
    st.error("Failed to get a response from the API after multiple attempts.")
    return None

st.set_page_config(layout="wide")
st.title("Earnings Call Transcript Summarizer")

uploaded_file = st.file_uploader("Upload an Earnings Call Transcript (PDF)", type=["pdf"])
if uploaded_file:
    if st.button("Generate Summary"):
        if model is None:
            st.error("Model not configured. Check API key.")
            st.stop()

# --- This block runs when the user clicks the button ---
    with st.spinner("Processing... This may take a minute."):
        
        # --- Step 1: Read the PDF ---
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                full_text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
            st.info("PDF Read Successfully.")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            st.stop()

        # --- Step 2: Call Agent 1 (Chunker) ---
        st.subheader("Step 1: Chunking Transcript")
        with st.expander("See Chunker Details"):
            chunker_response_text = call_gemini(CHUNK_AGENT_PROMPT, full_text)
            if not chunker_response_text:
                st.error("Chunker Agent failed.")
                st.stop()
            
            # Clean the response to get pure JSON
            chunker_json_text = clean_json_response(chunker_response_text)
            st.json(chunker_json_text) # Show the raw JSON for debugging

        # --- Step 3: Call Agent 2 (Analyzer) ---
        st.subheader("Step 2: Analyzing Text")
        with st.expander("See Analyzer Details"):
            # We will analyze the *entire* text in one go, as we found
            # in our manual test that this works well.
            analyzer_response_text = call_gemini(ANALYZER_AGENT_PROMPT, full_text)
            if not analyzer_response_text:
                st.error("Analyzer Agent failed.")
                st.stop()
            
            # Clean the response to get pure JSON
            analyzer_json_text = clean_json_response(analyzer_response_text)
            st.json(analyzer_json_text) # Show the raw JSON for debugging

        # --- Step 4: Call Agent 3 (Synthesizer) ---
        st.subheader("Step 3: Synthesizing Final Report")
        with st.expander("See Synthesizer Details"):
            # Pass the ANALYZER's JSON output to the Synthesizer
            final_report_text = call_gemini(SYNTHESIZER_AGENT_PROMPT, analyzer_json_text)
            if not final_report_text:
                st.error("Synthesizer Agent failed.")
                st.stop()
            
            st.text("Synthesizer received the JSON and produced this report:")

        # --- Step 5: Display the Final Product ---
        st.subheader("Your Executive Summary")
        st.markdown(final_report_text)

        st.balloons()