import os.path

import tiktoken
import torch
import pandas as pd
from tqdm import tqdm
import re
import evaluate
from openai import OpenAI
import time

key = ''
client = OpenAI(api_key=key)

def add_line_numbers(code_snippet):
    """
    Add line numbers to a code snippet.
    """
    lines = code_snippet.strip().split('\n')
    return '\n'.join([f"{i+1}: {line}" for i, line in enumerate(lines)])

def select_exemplar(training_set):
    """
    Select an exemplar from the training set.
    Here, simply selecting the first entry, but you could implement more complex logic.
    """
    exemplar = training_set.iloc[10]
    return add_line_numbers(exemplar["function"]), {"lines": eval(exemplar["location"])}

def construct_prompt(exemplar_code, exemplar_location, new_function):
    """
    Construct the prompt for the model based on the exemplar and new function.
    """
    prompt = (f"You are a security expert who knows the locations of a software vulnerability."
              f"Here is an exemplar vulnerable function:\n"
              f"{exemplar_code}\n"
              f"You should tell the locations like this: {exemplar_location}\n"
              f"Now there is a new vulnerable function:\n"
              f"{new_function}\n"
              f"Please list the vulnerable line numbers in the same format as the example.")
    return prompt

def analyze_code_for_vulnerabilities(code_snippet):
    """
    Analyze the code snippet with the CodeLlama model to identify line numbers of potential vulnerabilities.
    """

    # Add line numbers to the code snippet
    code_with_lines = add_line_numbers(code_snippet)
    encoded = encoding.encode(code_with_lines)[:max_length]
    code_with_lines = encoding.decode(encoded)

    # Formulate the prompt
    exemplar_code, exemplar_location = select_exemplar(train_set)
    prompt = construct_prompt(exemplar_code, exemplar_location, code_with_lines)
    print(prompt)
    min_cost = 0.5


    while True:
        try:
            completion = client.chat.completions.create(
                temperature=0,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=128
            )

            # print(completion.choices[0].message)
            break
        except Exception as e:
            print(e)
            time.sleep(60)

    reason = completion.choices[0].finish_reason
    if reason == 'stop':
        content = completion.choices[0].message.content
        print(content)
    else:
        print(reason)
        content = completion.choices[0].message.content
        print(content)
    time.sleep(min_cost)
    return content


def extract_line_numbers(generated_text):
    """
    Extract line numbers from the generated text.
    """
    json_pattern = r'\{.*?\}'
    matches = re.findall(json_pattern, generated_text, re.DOTALL)

    if matches:
        json_string = ' '.join(matches)
        generated_text = json_string

    # Find all numbers in the generated text
    line_numbers = re.findall(r'\d+', generated_text)
    # Convert them to integers
    return [int(number) for number in line_numbers]

def create_binary_labels(code_snippet, line_numbers):
    """
    Create a list of binary labels for each line in the code snippet.
    """
    total_lines = len(code_snippet.strip().split('\n'))
    # Initialize all labels to 0 (non-vulnerable)
    labels = [id2label[0]] * total_lines
    last_number = 0
    # Mark lines identified as vulnerable
    for line_number in line_numbers:
        if 1 <= line_number <= total_lines and line_number > last_number:
            labels[line_number - 1] = id2label[1]  # Adjust for zero-based indexing
            last_number = line_number
    return labels


# model_name = "gpt-3.5-turbo"
model_name = "gpt-4"
encoding = tiktoken.encoding_for_model(model_name)
max_length = 3968
train_set = pd.read_csv('../sc_data/train.csv')
file_name = f'{model_name}-one_sc.out'
if not os.path.exists(file_name):
    with open(file_name, 'w'):
        lines = []
else:
    with open(file_name, 'r') as file:
        lines = file.readlines()
cursor = len(lines)
id2label = {0: 'O', 1: 'B-LOC'}
with open(file_name, 'a') as res:
    true_labels = []
    true_predictions = []
    test_set = pd.read_csv('../sc_data/test.csv')
    for (idx, row) in tqdm(test_set.iterrows()):
        # if idx > 10:
        #     break
        if idx < cursor:
            continue
        code_snippet = row['function']
        # Analyze the code
        analysis_results = analyze_code_for_vulnerabilities(code_snippet)
        line_numbers = extract_line_numbers(analysis_results)
        print(line_numbers)
        predicted_labels = create_binary_labels(code_snippet, line_numbers)
        res.write(repr(predicted_labels) + '\n')

        labels = create_binary_labels(code_snippet, sorted(eval(row['location'])))
        true_labels.append(labels)
        true_predictions.append(predicted_labels)

    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    ret =  {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
    print(ret)

