from dotenv import load_dotenv
load_dotenv()
import json
import anthropic

client = anthropic.Anthropic()

#Program for looping a prompt evaluation. linked to promptEngineering.py


def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})


def chat(messages, system=None, stop_sequences=[]):
    params = {
        "model": "claude-haiku-4-5",
        "max_tokens": 1500,
        "messages": messages,
    }
    if system:
        params["system"] = system
    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    response = client.messages.create(**params)
    print(f"\n Input Tokens:{response.usage.input_tokens}")
    print(f"\n Output Tokens:{response.usage.output_tokens}")

    if response.stop_reason == "max_tokens":
        print("⚠️  WARNING: response was truncated (hit max_tokens)")

    return response.content[0].text


class PromptEvaluator:
    def __init__(self, max_concurrent_tasks=3):
        # concurrency isn't actually implemented here (calls run sequentially),
        # but kept as a constructor arg to match the intended interface
        self.max_concurrent_tasks = max_concurrent_tasks

    def generate_dataset(self, task_description, prompt_inputs_spec, output_file, num_cases=3):
        spec_lines = "\n".join(
            f'    "{key}": "{desc}"' for key, desc in prompt_inputs_spec.items()
        )

        gen_prompt = f"""
Generate an evaluation dataset for a prompt evaluation.

Task the prompt performs: {task_description}

Generate an array of JSON objects, each representing one test case.
Each object must have exactly these fields:
{{
{spec_lines}
}}

* Vary the values across test cases so the eval covers different real-world scenarios
* Please generate {num_cases} objects
* Return ONLY the JSON array, no other text
"""
        messages = []
        add_user_message(messages, gen_prompt)
        add_assistant_message(messages, "```json")
        text = chat(messages, stop_sequences=["```"])
        dataset = json.loads(text)

        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=2)

        return dataset

    def run_evaluation(self, run_prompt_function, dataset_file, extra_criteria=""):
        with open(dataset_file, "r") as f:
            dataset = json.load(f)

        results = []
        for test_case in dataset:
            output = run_prompt_function(test_case)
            grade = self._grade_by_model(test_case, output, extra_criteria)

            results.append({
                "test_case": test_case,
                "output": output,
                "grade": grade,
                "score": grade["score"]
            })

        avg_score = sum(r["score"] for r in results) / len(results)
        print(f"\nAverage score: {avg_score:.2f}/10")

        return results

    def _grade_by_model(self, test_case, output, extra_criteria):
        eval_prompt = f"""
You are an expert reviewer. Evaluate this AI-generated solution.

Test case inputs: {json.dumps(test_case, indent=2)}

Solution: {output}

{extra_criteria}

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