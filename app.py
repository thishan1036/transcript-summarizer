import streamlit as st
import google.generativeai as genai
import pdfplumber
import json
import re # We are no longer using re, but in case you add it back, we import it.
import time

# --- Configuration ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash-latest') # Using the model that worked
except Exception as e:
    st.error("Error configuring Google AI. Check your Streamlit Secrets.")
    model = None

# --- Agent Prompts (The New Assembly Line) ---

# Agent 1: The Cleaner (NEW)
# Its ONLY job is to fix the messy text, using your explicit rules.
cleaner_agent_prompt = """
Your only job is to be a text-cleaning specialist. You will be given messy text from a PDF.
You must rewrite it as perfectly clean, human-readable text.
**You MUST use these examples as a strict guide:**

* `51.2billion(up26` = `$51.2 billion (up 26%)`
* `18.6billion(7.25 per share)` = `$18.6 billion ($7.25 per share)`
* `56billionto59billion` = `$56 billion to $59 billion`
* `70-72 billion` = `$70 billion to $72 billion`
* `116-118 billion` = `$116 billion to $118 billion`
* `116billionto118billion` = `$116 billion to $118 billion`
* `$60 billion.Engagementremainshigh` = `$60 billion. Engagement remains high`
* `"personalsuperintelligence."Thisaggressivepivot` = `"personal superintelligence." This aggressive pivot`

Do not summarize. Do not analyze. Do not change any numbers.
Your only output is the clean text.
"""

# Agent 2: The Analyzer (Simplified)
# Its job is simple: extract facts from *clean text*.
analyzer_agent_prompt = """
Your only job and only output must be a single, valid JSON object. Do not add any text, commentary, or explanation before or after the JSON.

You will be given **clean, human-readable text.**
Your task is to analyze this text and extract the information into these keys:
`key_numbers`, `strategic_updates`, `risk_factors`, and `red_flags`.

The value for each key must be a list of strings (bullet points).
If you find no information for a key, return an empty list `[]`.
"""

# Agent 3: The Synthesizer (Plain Text)
# Its job is simple: take clean JSON and make a plain-text report.
synthesizer_agent_prompt = """
You are an executive editor. Your only job is to synthesize a JSON object of analyst notes into a single, high-level, plain-text executive summary.

1.  You will be given a JSON object of extracted facts.
2.  Your task is to write a cohesive summary.
3.  **STYLE RULE:** Your final output **must be plain text**. 
    * Do NOT use any Markdown (no asterisks for bolding or italics).
    * Use ALL-CAPS for headings, followed by a new line.
    * Use a simple dash (-) for bullet points.

**EXAMPLE OUTPUT FORMAT:**

EXECUTIVE SUMMARY
The company reported strong Q3 growth...

KEY METRICS & GUIDANCE
- Q3 Total Revenue: $51.2 billion, up 26% Y/Y.
- Q4 Revenue Guidance: $56 billion to $59 billion.
"""

# --- Helper Functions (The "Connectors") ---

def clean_json_response(response_text):
    """
    Helper function to clean the json artifacts from the AI's response.
    """
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1
    if json_start != -1 and json_end != -1:
        return response_text[json_start:json_end]
    else:
        return response_text # Fallback

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
            if response and response.text:
                return response.text
            else:
                st.error(f"API Error (Attempt {attempt + 1}): Received empty response.")
                time.sleep(1)
        except Exception as e:
            st.error(f"API Error (Attempt {attempt + 1}/{retry_count}): {e}")
            if "quota" in str(e).lower():
                st.error("Quota exceeded. Please check your Google AI billing.")
                return None
            time.sleep(1)
    
    st.error("Failed to get a response from the API after multiple attempts.")
    return None

# --- Main Application (The New "Factory Floor") ---

def main():
    """
    The main function to run the Streamlit app.
    """
    st.set_page_config(layout="wide")
    st.title("🤖 AI Earnings Call Summarizer")

    uploaded_file = st.file_uploader("Upload an Earnings Call Transcript (PDF)", type=["pdf"])

    if uploaded_file:
        if st.button("Generate Summary"):
            if model is None:
                st.error("Model not configured. Check API key in Streamlit Secrets.")
                st.stop()

            # --- This block runs when the user clicks the button ---
            with st.spinner("Processing... This may take 1-2 minutes."):
                
                # --- Step 1: Read the PDF ---
                full_text = ""
                try:
                    with pdfplumber.open(uploaded_file) as pdf:
                        full_text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
                    st.info("PDF Read Successfully.")
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")
                    st.stop()
                
                # --- Step 2: Call Agent 1 (Cleaner) ---
                st.subheader("Step 1: Cleaning Raw Text")
                clean_text = None
                with st.expander("See Cleaner Details"):
                    clean_text = call_gemini(cleaner_agent_prompt, full_text)
                    if not clean_text:
                        st.error("Cleaner Agent failed.")
                        st.stop()
                    st.text("Raw text has been cleaned by the AI.")
                    # st.text(clean_text) # Uncomment to debug

                # --- Step 3: Call Agent 2 (Analyzer) ---
                st.subheader("Step 2: Analyzing Clean Text")
                analyzer_json_text = None
                if clean_text:
                    with st.expander("See Analyzer Details"):
                        analyzer_response_text = call_gemini(analyzer_agent_prompt, clean_text)
                        if not analyzer_response_text:
                            st.error("Analyzer Agent failed.")
                            st.stop()
                        
                        analyzer_json_text = clean_json_response(analyzer_response_text)
                        st.json(analyzer_json_text)

                # --- Step 4: Call Agent 3 (Synthesizer) ---
                st.subheader("Step 3: Synthesizing Final Report")
                final_report_text = None
                if analyzer_json_text:
                    with st.expander("See Synthesizer Details"):
                        final_report_text = call_gemini(synthesizer_agent_prompt, analyzer_json_text)
                        if not final_report_text:
                            st.error("Synthesizer Agent failed.")
                            st.stop()
                        
                        st.text("Synthesizer received the JSON and produced this report:")

                # --- Step 5: Display the Final Product ---
                if final_report_text:
                    st.subheader("🎉 Your Executive Summary")
                    # Use st.text to display plain text perfectly
                    st.text(final_report_text)
                    st.balloons()
                else:
                    st.error("Could not generate the final report.")

# This makes the script runnable
if __name__ == "__main__":
    main()