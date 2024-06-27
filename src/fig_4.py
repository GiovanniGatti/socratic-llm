import json
import os
from openai import OpenAI


def compose_prompt(element):
    o = element['output']
    if "Student" in o:
        o = o.split("Student")[0]

    s = f"####Here is the conversation:#### \n  {element['prompt']}\n #####Here is the answer#### : \n {o}\n\n"

    return s

if __name__ == "__main__":
    path_dpo = r"mathdial_mathdial_with_scores.json"
    path_not_finetuned = r"mathdial_wo_finetuning_with_scores.json"
    prompt_path = r"/home/bonino/Documents2/llama2-fine-tune/evaluation_prompt.txt"

    with open(path_dpo) as f:
        json_file_dpo = json.load(f)

    



    with open(path_not_finetuned) as f:
        json_file_not_finetuned = json.load(f)

    with open(prompt_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    file_content = "You are an expert of mathematics and a teacher" + file_content

    client = OpenAI(
        # This is the default and can be omitted
        api_key=""  # put here your api key
    )

    scores = []
    evals = []
    for i in json_file_dpo:
        print(i)
        out = compose_prompt(i)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": file_content},
                {
                    "role": "user",
                    "content": out,
                }
            ],
            model="gpt-3.5-turbo",
            temperature=0.2,
        )
        print(chat_completion.choices[0].message.content)
        res = json.loads(chat_completion.choices[0].message.content)
        evals.append(res)
        score = calculate_score(res)
        scores.append(score)