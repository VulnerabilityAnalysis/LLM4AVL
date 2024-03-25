import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
import numpy as np
import evaluate

def add_line_numbers(code_snippet):
    """
    Add line numbers to a code snippet.
    """
    lines = code_snippet.strip().split('\n')
    return '\n'.join([f"{i+1}: {line}" for i, line in enumerate(lines)])
def preprocess_function(examples):
    inputs = [add_line_numbers(doc) for doc in examples["function"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    targets = []
    for ipt, tgt in zip(inputs, examples["location"]):
        lines = ipt.split('\n')
        temp = "Vulnerable Lines: "+'\n'.join([lines[i-1] for i in eval(tgt)])
        targets.append(temp)
    # Setup the tokenizer for targets
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)


    # Function to extract numbers from a string
    def extract_numbers(text):
        return set(map(int, re.findall(r'(\d+):', text)))

    # Function to calculate Jaccard similarity
    def jaccard_similarity(set1, set2):
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if len(union) != 0 else 0

    # Extract numbers and calculate Jaccard similarity for each pair
    jaccard_scores = [jaccard_similarity(extract_numbers(pred), extract_numbers(lbl)) for pred, lbl in zip(decoded_preds, decoded_labels)]

    # Calculate average Jaccard score
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0

    return {"average_jaccard": avg_jaccard}


train_dataset = load_dataset("csv", data_files="../data/train.csv", split="train")
eval_dataset = load_dataset("csv", data_files="../data/valid.csv", split="train")

epochs = 10
batch_size = 8
learning_rate = 5e-5
max_input_length = 512
max_target_length = 256

model_id = 'uclanlp/plbart-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)
seqeval = evaluate.load("seqeval")

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_id)


model_name = model_id.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-gen",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    num_train_epochs=epochs,
    predict_with_generate=True,
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()