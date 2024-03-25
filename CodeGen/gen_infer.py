from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import re
import torch
from datasets import load_dataset
import evaluate
from tqdm import tqdm


def add_line_numbers(function_code):
    lines = function_code.split('\n')
    modified_lines = []
    line_number = 0

    for line in lines:
        line_number += 1
        modified_lines.append(f"{line_number}: {line}")

    code = '\n'.join(modified_lines)
    ids = tokenizer(code, max_length=max_input_length, truncation=True, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(ids)

def is_shorter_than_max_length(example):
    text_column_name = "function"
    tokens = tokenizer.tokenize(example[text_column_name])
    return len(tokens) < max_input_length
def transform_to_binary_labels(line_numbers, total_lines):
    """
    Transform line numbers to binary labels for sequence labeling.
    """
    labels = ['O'] * total_lines
    for line in line_numbers:
        if line <= total_lines:
            labels[line - 1] = 'B-LOC'
    return labels


def predict_location(code_snippet, model, tokenizer):
    """
    Generate predictions for a given code snippet.
    """
    input_text = add_line_numbers(code_snippet)
    input_text = f"{input_text}\n\n### Vulnerable lines: "

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    output_sequences = model.generate(**inputs,
                                      max_new_tokens=max_target_length,
                                      pad_token_id=tokenizer.eos_token_id)
    decoded_prediction = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    decoded_prediction = decoded_prediction.split("### Vulnerable lines:")[1]
    print(decoded_prediction)
    return [int(r) for r in re.findall(r'(\d+):', decoded_prediction)]


device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./codegen-6B-mono-gen/checkpoint-21620"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
max_input_length = 1700
max_target_length = 256

# Load and preprocess test dataset
test_dataset = load_dataset("csv", data_files="../data/test.csv", split="train")
# Filter the dataset
# filtered_test_dataset = test_dataset.filter(is_shorter_than_max_length)
# print(f"Original size: {len(test_dataset)}, Filtered size: {len(filtered_test_dataset)}")
# test_dataset = filtered_test_dataset


# Inference and transformation
predictions = [predict_location(item['function'], model, tokenizer) for item in tqdm(test_dataset)]
predicted_binary_labels = [transform_to_binary_labels(pred, len(item['function'].split('\n'))) for pred, item in zip(predictions, test_dataset)]
true_binary_labels = [transform_to_binary_labels(eval(item['location']), len(item['function'].split('\n'))) for item in test_dataset]

# Load seqeval from evaluate library
seqeval_metric = evaluate.load("seqeval")

# Compute metrics using seqeval from evaluate
results = seqeval_metric.compute(predictions=predicted_binary_labels, references=true_binary_labels)

print("Precision:", results["overall_precision"])
print("Recall:", results["overall_recall"])
print("F1 Score:", results["overall_f1"])
print("Accuracy:", results["overall_accuracy"])