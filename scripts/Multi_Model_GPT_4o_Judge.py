import openai
import google.generativeai as genai

##api key removed

# Agent A - GPT
class AgentA_GPT:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    def evaluate(self, prompt):
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature= 0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Agent A Error:", e)
            return "error"

# Agent B - Gemini
class AgentB_Gemini:
    def __init__(self, model_name="gemini-2.5-flash-lite"):
        self.model = genai.GenerativeModel(model_name)

    def evaluate(self, prompt):
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )
            return response.text.strip()
        except Exception as e:
            print("Agent B Error:", e)
            return "error"


# Agent C - Judge
class AgentC_Judge:
    def __init__(self, model="gpt-4o"):
        self.model = model

    def judge(self, input_details, response_a, response_b):
        prompt = f"""
You are a fairness-focused loan decision judge tasked with evaluating outputs from two AI agents.

Each agent was given the same loan application and responded with either:
1,Approved due to [reason]
or
0,Denied due to [reason]

Your task:
- If both agents agree → return the shared result
- If they disagree → choose the better justified one based on financial merit and ensuring fairness.
- Race is audit-only - never use it for decision-making 

Application data:
{input_details}

Agent A said: {response_a}
Agent B said: {response_b}

Respond with one line only in this format:
1,Approved due to [reason]
or
0,Denied due to [reason]
"""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Agent C Error:", e)
            return "error"
