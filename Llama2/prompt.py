from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import pandas as pd
from tqdm import tqdm
import re
import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"
def add_line_numbers(code_snippet):
    """
    Add line numbers to a code snippet.
    """
    lines = code_snippet.strip().split('\n')
    return '\n'.join([f"{i+1}: {line}" for i, line in enumerate(lines)]).strip()

def analyze_code_for_vulnerabilities(code_snippet):
    """
    Analyze the code snippet with the CodeLlama model to identify line numbers of potential vulnerabilities.
    """

    # Add line numbers to the code snippet
    code_with_lines = add_line_numbers(code_snippet)
    input_ids = tokenizer.encode(code_with_lines)
    code_with_lines = tokenizer.decode(input_ids[:max_code_tokens])

    # Formulate the prompt
    prompt = (f"<s>[INST] You are a security expert who knows the locations of software vulnerabilities. Now it's reported that there is one vulnerability in the following function:\n\n{code_with_lines}\n"
              f"Please list the vulnerable line numbers strictly in the format of JSON. [/INST]")

    # Prepare the input
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)

    # Generate the output
    output_ids = model.generate(**inputs, max_new_tokens=128)  # set your generation parameters

    # Decode the generated tokens
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # print(generated_text)
    # Extract only the generated text, excluding the input
    # generated_text = results[0]['generated_text']
    output_only = generated_text.split('[/INST]')[1].strip()  # Removes the first occurrence of prompt
    return output_only


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
    print(line_numbers)
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


# Load the CodeLlama model
model_name = "meta-llama/Llama-2-7b-chat-hf"
file_name = 'zs_llama_7b_sc.out'
# token = "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"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
max_code_tokens = 3968
id2label = {0: 'O', 1: 'B-LOC'}
with open(file_name, 'w') as res:
    true_labels = []
    true_predictions = []
    test_set = pd.read_csv('../sc_data/test.csv')
    for (idx, row) in tqdm(test_set.iterrows()):
        # if idx > 10:
        #     break
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

