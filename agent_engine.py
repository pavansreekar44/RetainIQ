import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure the Gemini SDK
# Note: Ensure GOOGLE_API_KEY is set in your environment variables.
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Use the fast, cost-effective flash model
model = genai.GenerativeModel('gemini-3.5-flash')

def get_real_agent_intervention(customer_context: dict, risk_score: float, shap_reasons: list) -> dict:
    """
    Calls the Gemini LLM to generate a targeted retention strategy and email draft 
    based on the XGBoost churn risk score and SHAP feature drivers.
    """
    
    prompt = f"""
You are an expert Customer Retention Strategist. 
A customer has been flagged by our ML model with a churn risk score of {risk_score * 100:.1f}%.

Here is the customer's profile:
{json.dumps(customer_context, indent=2)}

Here are the top factors (SHAP drivers) contributing to their current churn risk:
{json.dumps(shap_reasons, indent=2)}

Your task is to review this data and formulate a specific retention offer (e.g., a discount, 
tech support upgrade, contract adjustment, or loyalty perk) that directly counters the SHAP drivers.

You MUST respond strictly in raw JSON format with NO markdown wrapping, NO backticks, and NO extra text.
The JSON must contain exactly these two keys:
1. "recommended_action": A short, strategic directive for the human account manager based on the SHAP data.
2. "generated_email_draft": A personalized, polite email directly addressing the customer, offering the targeted retention incentive.

Return ONLY the raw JSON string.
"""
    
    # Generate the response
    response = model.generate_content(prompt)
    raw_text = response.text.strip()
    
    # Clean the response just in case the LLM ignored instructions and added markdown backticks
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:]
        
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
        
    raw_text = raw_text.strip()
    
    # Parse and return as a Python dictionary
    try:
        intervention = json.loads(raw_text)
        return intervention
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM output. Raw text:\n{raw_text}")
        raise RuntimeError("LLM did not return valid JSON. Check console for output.") from e
