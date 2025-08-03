import openai
import google.generativeai as genai

client = openai.OpenAI(api_key="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")
genai.configure(api_key="AIzaSyBimH_WLCUIgucBK-HRuHMj1jS2OMgmPKg")

# Agent A - GPT-3.5 (Evaluator)
class AgentA_GPT:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    def evaluate(self, prompt):
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Agent A Error:", e)
            return "error"

# Agent B - GPT-4 (Evaluator)
class AgentB_GPT:
    def __init__(self, model="gpt-4o"):
        self.model = model

    def evaluate(self, prompt):
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Agent B Error:", e)
            return "error"

# Agent C - Gemini (Judge)
class AgentC_GeminiJudge:
    def __init__(self, model_name="gemini-2.5-flash-lite"):
        self.model = genai.GenerativeModel(model_name)

    def judge(self, input_details, response_a, response_b):
        prompt = f"""
You are a fairness judge evaluating loan decisions made by two AI agents.

Each agent was given the same loan application and responded with either:
1,Approved due to [reason]
or
0,Denied due to [reason]

Your task:
- If both agents agree → return the shared result
- If they disagree → choose the better justified one based on financial merit, ensuring fairness.
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
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )
            return response.text.strip()
        except Exception as e:
            print("Agent C Error:", e)
            return "error"