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
You are an executive editor. Your *only* job is to synthesize a JSON object of analyst notes into a *new, final JSON object* that will be used to build a report.

1.  You will be given a JSON object of extracted facts.
2.  Your task is to review all the notes and synthesize the information.
3.  You **must** return a single, valid JSON object with the following keys:
    * `executive_summary`: A 2-3 sentence overview.
    * `key_metrics`: A list of strings (bullets) for the most critical numbers.
    * `strategic_developments`: A list of strings (bullets) for key updates.
    * `risks_and_red_flags`: A list of strings (bullets) for significant risks.
4.  **CRITICAL:** When you write the text for the lists, ensure all spacing is correct (e.g., "$56 billion to $59 billion", not "$56billionto").
5.  Do not add any commentary. Your *only* output is the final JSON.

**Example Output Format:**
```json
{
  "executive_summary": "The company reported strong Q3 growth with revenue up 26%, driven by...",
  "key_metrics": [
    "Q3 Total Revenue: $51.2 billion, up 26% Y/Y.",
    "Q4 Revenue Guidance: $56 billion to $59 billion."
  ],
  "strategic_developments": [
    "Strategic priority is establishing the company as the leading frontier AI lab."
  ],
  "risks_and_red_flags": [
    "Regulatory Headwinds (Europe): Cannot rule out...",
    "Legal Exposure: Youth-related trials..."
  ]
}
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

def build_report_from_json(report_json_text):
    """
    Takes the final JSON from the synthesizer and builds the clean,
    human-readable markdown report.
    """
    try:
        data = json.loads(report_json_text)
    except json.JSONDecodeError:
        st.error("Error: Unable to parse final report JSON.")
        return report_json_text
    
    report_markdown = []

    if "executive_summary" in data:
        report_markdown.append("### Executive Summary")
        report_markdown.append(data["executive_summary"])
        report_markdown.append("\n")

    if "key_metrics" in data:
        report_markdown.append("**Key Metrics & Guidance**")
        for item in data.get("key_metrics", []):
            report_markdown.append(f"*{item}*")
        report_markdown.append("\n")

    if "strategic_developments" in data:
        report_markdown.append("**Strategic Developments**")
        for item in data.get("strategic_developments", []):
            report_markdown.append(f"*{item}*")
        report_markdown.append("\n")

    if "risks_and_red_flags" in data:
        report_markdown.append("**Risks & Red Flags**")
        for item in data.get("risks_and_red_flags", []):
            report_markdown.append(f"*{item}*")
        report_markdown.append("\n")
    
    return "\n".join(report_markdown)




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
                try:
                    with pdfplumber.open(uploaded_file) as pdf:
                        # Ensure text is extracted and joined with a space
                        full_text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
                    st.info("PDF Read Successfully.")
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")
                    st.stop()

                # --- Step 2: Call Agent 2 (Analyzer) ---
                st.subheader("Step 1: Analyzing Full Transcript")
                analyzer_json_text = None # Initialize
                with st.expander("See Analyzer Details"):
                    analyzer_response_text = call_gemini(ANALYZER_AGENT_PROMPT, full_text)
                    if not analyzer_response_text:
                        st.error("Analyzer Agent failed.")
                        st.stop()
                    
                    # Clean the response to get pure JSON
                    analyzer_json_text = clean_json_response(analyzer_response_text)
                    st.json(analyzer_json_text) # Show the raw JSON for debugging

                # --- Step 3: Call Agent 3 (Synthesizer) ---
                st.subheader("Step 2: Synthesizing Final JSON")
                final_json_text = None # Initialize
                if analyzer_json_text: # Only run if analyzer was successful
                    with st.expander("See Synthesizer Details"):
                        # Pass the ANALYZER's JSON output to the Synthesizer
                        synthesizer_response = call_gemini(SYNTHESIZER_AGENT_PROMPT, analyzer_json_text)
                        if not synthesizer_response:
                            st.error("Synthesizer Agent failed.")
                            st.stop()
                        
                        # Clean the response to get pure JSON
                        final_json_text = clean_json_response(synthesizer_response)
                        st.json(final_json_text) # Show the raw JSON for debugging

                # --- Step 4: Build the Final Report (NEW STEP) ---
                st.subheader("Step 3: Building Final Report")
                final_report_markdown = None
                if final_json_text:
                    with st.expander("See Final Report Markdown"):
                        # Use our new function to build the report
                        final_report_markdown = build_report_from_json(final_json_text)
                        st.text(final_report_markdown)
                
                # --- Step 5: Display the Final Product ---
                if final_report_markdown:
                    st.subheader("Your Executive Summary")
                    st.markdown(final_report_markdown)
                    st.balloons()
                else:
                    st.error("Could not generate the final report.")

# This makes the script runnable
if __name__ == "__main__":
    main()