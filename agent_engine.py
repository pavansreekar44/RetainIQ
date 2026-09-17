import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_real_agent_intervention(customer_context: dict, risk_score: float, shap_reasons: list) -> dict:
    """
    Calls the Groq LLM (LLaMA 3) to generate a targeted retention strategy and email draft 
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

You MUST respond strictly in JSON format.
The JSON must contain exactly these two keys:
1. "recommended_action": A short, strategic directive for the human account manager based on the SHAP data.
2. "generated_email_draft": A personalized, polite email directly addressing the customer, offering the targeted retention incentive.
"""
    
    # Generate the response using Groq's lightning-fast model
    response = client.chat.completions.create(
        model="groq/compound-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    raw_text = response.choices[0].message.content.strip()
    
    # Parse and return as a Python dictionary
    try:
        intervention = json.loads(raw_text)
        return intervention
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM output. Raw text:\n{raw_text}")
        raise RuntimeError("LLM did not return valid JSON. Check console for output.") from e
