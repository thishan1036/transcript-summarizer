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

cleaner_agent_prompt = """
You are a text-cleaning specialist. Your only job is to take a raw, messy chunk of text from a PDF and rewrite it with perfect, human-readable formatting.

1.  The text you receive will have formatting errors (e.g., "50billion", "26%YoY", "51.2billion(up26").
2.  Your task is to fix all of them. Add correct spacing between words, numbers, and punctuation.
3.  You must not change any words or numbers. You*must not summarize or analyze.
4.  Your only output is the perfectly clean, rewritten text. Do not add any commentary.
"""

analyzer_agent_prompt = """
Your only job and only output must be a single, valid JSON object. Do not add any text, commentary, or explanation before or after the JSON.

Your task is to be a senior financial analyst. The text you receive is messy and comes from a PDF. It will have formatting errors.

**YOUR MOST IMPORTANT JOB IS TO FIX THESE ERRORS.**

1.  Analyze the text and extract the information into the keys: `key_numbers`, `strategic_updates`, `risk_factors`, and `red_flags`.
2.  When you extract data, you **must** rewrite it as clean, human-readable text.
3.  Use these examples as a strict guide for how to fix the text:
    * `51.2billion(up26` = `$51.2 billion (up 26%)`
    * `18.6billion(7.25 per share)` = `$18.6 billion ($7.25 per share)`
    * `56billionto59billion` = `$56 billion to $59 billion`
    * `$70-72 billion` = `$70 billion to $72 billion`
    * `116-118 billion` = `$116 billion to $118 billion`

4.  You are an **analyst and a cleaner**. Do not just copy the broken text.
5.  If you find no information for a key, return an empty list `[]`.
"""

synthesizer_agent_prompt = """
You are an executive editor at a top-tier financial publication. 
Your only job is to synthesize a JSON object of analyst notes into a single, high-level, plain-text executive summary.

1.  You will be given a JSON object of extracted facts.
2.  Your task is to write a cohesive summary.
3.  **STYLE RULE:** Your final output **must be plain text**. 
    * Do NOT use any Markdown (no asterisks for bolding or italics).
    * Use ALL-CAPS for headings, followed by a new line.
    * Use a simple dash (-) for bullet points.

**EXAMPLE OUTPUT FORMAT:**

EXECUTIVE SUMMARY
The company reported strong Q3 growth with revenue up 26%, driven by...

KEY METRICS & GUIDANCE
- Q3 Total Revenue: $51.2 billion, up 26% Y/Y.
- Q4 Revenue Guidance: $56 billion to $59 billion.

STRATEGIC DEVELOPMENTS
- Strategic priority is establishing the company as the leading frontier AI lab.

RISKS & RED FLAGS
- Regulatory Headwinds (Europe): Cannot rule out...
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

# --- Main Application (The "Factory Floor") ---

def main():
    """
    The main function to run the Streamlit app.
    """
    st.set_page_config(layout="wide")
    st.title("Earnings Call Summarizer")

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
                        # Ensure text is extracted and joined with a space
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
                    st.text(clean_text) # Show the clean text for debugging

                # --- Step 3: Call Agent 2 (Analyzer) ---
                st.subheader("Step 2: Analyzing Clean Text")
                analyzer_json_text = None # Initialize
                if clean_text: # Only run if cleaning was successful
                    with st.expander("See Analyzer Details"):
                        analyzer_response_text = call_gemini(analyzer_agent_prompt, clean_text)
                        if not analyzer_response_text:
                            st.error("Analyzer Agent failed.")
                            st.stop()
                        
                        # Clean the response to get pure JSON
                        analyzer_json_text = clean_json_response(analyzer_response_text)
                        st.json(analyzer_json_text) # Show the raw JSON for debugging

                # --- Step 4: Call Agent 3 (Synthesizer) ---
                st.subheader("Step 3: Synthesizing Final Report")
                final_report_text = None # Initialize
                if analyzer_json_text: # Only run if analyzer was successful
                    with st.expander("See Synthesizer Details"):
                        # Pass the ANALYZER's JSON output to the Synthesizer
                        final_report_text = call_gemini(synthesizer_agent_prompt, analyzer_json_text)
                        if not final_report_text:
                            st.error("Synthesizer Agent failed.")
                            st.stop()
                        
                        st.text("Synthesizer received the JSON and produced this report:")

                # --- Step 5: Display the Final Product ---
                if final_report_text:
                    st.subheader("Your Executive Summary")
                    st.markdown(final_report_text)
                    st.balloons()
                else:
                    st.error("Could not generate the final report.")

# This makes the script runnable
if __name__ == "__main__":
    main()