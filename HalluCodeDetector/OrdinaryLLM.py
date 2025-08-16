import json
import re
import jsonlines
from ollama import chat
from ollama import ChatResponse
from openai import OpenAI

def getOrdinaryLLMScores(question_jsonl_path, answer_json_path, size_per_question, base_url, api_key, model):
    # In the following annotated section, humanevalplus_scores and mbpp_scores are our experimental results.

    # humanevalplus_scores = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0.5, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0.5, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0.5, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]
    # return humanevalplus_scores

    # mbpp_scores = [1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0.5, 0.5, 1, 0, 1, 0, 0, 0, 0, 0.5, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0.5, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0.5, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0.5, 0, 0, 0, 1, 0, 0.5, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.5, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0.5, 0.5, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0.5, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0.5, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0.5, 0, 0, 0, 0, 0.5, 1, 0.5, 1, 0, 1, 0.5, 1, 1, 0.5, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0.5, 1, 0, 0, 0, 1, 0, 0.5, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.5, 0, 0, 0.5, 1, 0.5, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0.5, 0.5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.5, 0.5, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0.5, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0.5, 1, 0.5, 1, 0.5, 1, 0, 0, 1, 0.5, 0.5, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0.5, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0.5, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0.5, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0.5, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0.5, 0, 1, 0.5, 1, 0.5, 0, 0, 0, 0.5, 1, 1, 0, 1, 0, 0.5, 1, 0, 0.5, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0.5, 0, 1, 0, 1, 0, 0, 0.5, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0.5, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0.5, 1, 0, 0, 1, 0.5, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0.5, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0.5, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0.5, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0.5, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0.5, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.5, 1, 0, 0, 0, 0.5, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0.5, 1, 1, 0, 0, 1, 0, 0, 0.5, 0, 0, 0, 0.5, 1, 0.5, 1, 0, 0, 0, 0, 0, 0, 0.5, 1, 0, 0, 0.5, 1, 0, 0, 0, 1, 0.5, 1, 0, 0, 1, 0.5, 1, 0, 0, 1, 0, 0, 1, 0.5, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0.5, 1, 0, 1, 0, 0, 0, 0, 0.5, 1, 1]
    # return mbpp_scores

    y_score = []
    questions = []

    with open(question_jsonl_path, encoding='utf-8') as file:
        for question in jsonlines.Reader(file):
            question_data = {
                "question": question["prompt"],
                "test": question["test"]
            }
            questions.append(question_data)

    with open(answer_json_path, 'r', encoding='utf-8') as file:
        answers_data = json.load(file)
    answers = []
    for question_id in range(0, int(len(answers_data) / size_per_question)):
        answers.append(answers_data[str(question_id * size_per_question + 0)]["generation_code"])

    for i in range(0, int(len(answers_data) / size_per_question)):
        question = questions[i]["question"]
        test = questions[i]["test"]
        answer = answers[i]
        prompt = f"Given the following QUESTION, TEST and ANSWER you must analyze the provided answer and and determine whether it passes the TEST. Output your final verdict by strictly following this format: \"PASS\" if the answer passes the TEST and \"FAIL\" if the answer does not pass the TEST. Show your reasoning.\n\n--\nQUESTION:\n{question}\n\n--\nTEST:\n[{test}]\n\n--\nANSWER:\n{answer}\n\n--\n\nYour output should be in JSON FORMAT with the keys \"REASONING\" and \"SCORE\":\n{{\"REASONING\": <your reasoning as bullet points>, \"SCORE\": <your final score>}}\n"

        client = OpenAI(api_key=api_key, base_url=base_url)
        completion = client.chat.completions.create(
            model=model,
            stream=True,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        full_response = ""

        print("id: " + str(i))

        try:
            for chunk in completion:
                if chunk.choices[0].delta and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
        except Exception as e:
            y_score.append(0.5)
            print("value: " + str(0.5))

        try:
            match = re.search(r'"SCORE"\s*:\s*([^,}]+)', str(full_response))
            score_value = match.group(1).strip()
            # print("score_value: " + score_value)
            if score_value == 'PASS' or score_value == '\"PASS\"':
                y_score.append(1)
                print("value: " + str(1))
            elif score_value == 'FAIL' or score_value == '\"FAIL\"':
                y_score.append(0)
                print("value: " + str(0))
            else:
                y_score.append(0.5)
                print("value: " + str(0.5))
        except Exception as e:
            print(str(e))
            y_score.append(0.5)
            print("value: " + str(0.5))

    return y_score