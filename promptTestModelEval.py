from dotenv import load_dotenv
load_dotenv()
import json
import anthropic

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
"""
    
    messages = []
    add_user_message(messages, prompt)
    output = chat(messages)
    return output

#Grades the ouput 
def grade_by_model(test_case, output):
    # Create evaluation prompt
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
    return json.loads(eval_text)

#    Calls run_prompt, then grades the result
def run_test_case(test_case):
    output = run_prompt(test_case)
    evaluation = grade_by_model(test_case, output)
    return {
        "output": output,
        "test_case": test_case,
        "evaluation": evaluation,
        "score": evaluation["score"]
    }

#Loads the dataset and calls run_test_case with each case
def run_eval(dataset):
    
    results = []
    
    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)
    
    return results

with open("dataset.json", "r") as f:
    dataset = json.load(f)
results = run_eval(dataset)

print(json.dumps(results,indent=2))
