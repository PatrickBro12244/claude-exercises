from dotenv import load_dotenv
load_dotenv()
import json
import anthropic
import re
import ast

client = anthropic.Anthropic()

#Program for running and evaluating the test data
#!!max_tokens are set to 1000, but this needs to be raised to run properly!!

def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)

def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)

def chat(messages, system=None, stop_sequences=[]):
    params = {
        "model": "claude-haiku-4-5",
        "max_tokens": 1000,
        "messages": messages,
    }
    if system:
        params["system"] = system
    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    response = client.messages.create(**params)
    print(f"\n Input Tokens:{response.usage.input_tokens}")
    print(f"\n Output Tokens:{response.usage.output_tokens}")

    
    return response.content[0].text

#Merges the prompt and test case input, then returns the result
def run_prompt(test_case):
    prompt = f"""
Please solve the following task:

{test_case["task"]}

* Respond only with Python, JSON, or a plain Regex
* Do not add any comments or commentary or explanation

"""

    messages = []
    add_user_message(messages, prompt)
    # Pre-fill the assistant turn with an opening code fence so the model
    # goes straight into raw code, without deciding on a language label
    add_assistant_message(messages, "```code")

    output = chat(messages, stop_sequences=["```"])
    return output

# ---- Syntax validators ----

def validate_json(text):
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0

def validate_python(text):
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0

def validate_regex(text):
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0

def grade_syntax(output, test_case):
    """Dispatches to the right validator based on test_case['format']."""
    format = test_case["format"]

    if format == "json":
        return validate_json(output)
    elif format == "python":
        return validate_python(output)
    elif format == "regex":
        return validate_regex(output)
    else:
        raise ValueError(f"Unknown format: {format}")

# ---- Model-based grading ----

def grade_by_model(test_case, output):
    eval_prompt = f"""
You are an expert code reviewer. Evaluate this AI-generated solution.

Task: {test_case["task"]}
Solution: {output}

Provide your evaluation as a structured JSON object with:
- "strengths": An array of 1-3 key strengths
- "weaknesses": An array of 1-3 key areas for improvement
- "reasoning": A concise explanation of your assessment
- "score": A number between 1-10
"""

    messages = []
    add_user_message(messages, eval_prompt)
    add_assistant_message(messages, "```json")

    eval_text = chat(messages, stop_sequences=["```"])

    try:
        return json.loads(eval_text)
    except json.JSONDecodeError:
        return {
            "strengths": [],
            "weaknesses": [],
            "reasoning": f"Failed to parse grading response: {eval_text[:200]}",
            "score": 0
        }

# ---- Per-test-case runner ----

def run_test_case(test_case):
    output = run_prompt(test_case)

    model_grade = grade_by_model(test_case, output)
    model_score = model_grade["score"]

    syntax_score = grade_syntax(output, test_case)

    score = (model_score + syntax_score) / 2

    return {
        "output": output,
        "test_case": test_case,
        "model_grade": model_grade,
        "syntax_score": syntax_score,
        "score": score
    }

#Loads the dataset and calls run_test_case with each case
def run_eval(dataset):
    results = []

    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)

    return results


with open("dataset_codegrader.json", "r") as f:
    dataset = json.load(f)

results = run_eval(dataset)

print(json.dumps(results, indent=2))