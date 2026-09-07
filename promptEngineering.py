from promptEngineering_prompteval import PromptEvaluator, add_user_message, chat
import json

# ---- Naive baseline prompt (deliberately basic, per the lesson) ----

def run_prompt(prompt_inputs):
    prompt = f"""
Given this athletes's information, and their goal of maintaining weight and improving power/performance, generate a one day meal plan for them that breaks down the imporant micros and macros as well as total carbs for the day.

- Height: {prompt_inputs["height"]}
- Weight: {prompt_inputs["weight"]}
- Goal: {prompt_inputs["goal"]}
- Dietary restrictions: {prompt_inputs["restrictions"]}
"""
    messages = []
    add_user_message(messages, prompt)
    return chat(messages)


# ---- Run everything ----

evaluator = PromptEvaluator(max_concurrent_tasks=3)

dataset = evaluator.generate_dataset(
    task_description="Write a compact, concise 1 day meal plan for a single athlete",
    prompt_inputs_spec={
        "height": "Athlete's height in cm",
        "weight": "Athlete's weight in kg",
        "goal": "Goal of the athlete",
        "restrictions": "Dietary restrictions of the athlete"
    },
    output_file="dataset.json",
    num_cases=3
)

results = evaluator.run_evaluation(
    run_prompt_function=run_prompt,
    dataset_file="dataset.json",
    extra_criteria="""
The output should include:
- Daily caloric total
- Macronutrient breakdown
- Meals with exact foods, portions, and timing
"""
)

print(json.dumps(results, indent=2))