import json
import sys
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from modeling_codet5 import CodeT5ForTokenClassification, T5ForTokenClassification



def load_cve():
    ret = {}
    for split_name in ['train', 'valid', 'test']:
        data = []
        with open(f'../sc_data/{split_name}.jsonl', 'r') as reader:
            for line in reader:
                data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
    return DatasetDict(ret)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, padding='longest', max_length=max_length, truncation=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = [-100] * len(word_ids)  # Initialize all labels to -100.
        last_token_position = None

        for word_idx_position, word_idx in enumerate(word_ids):
            if word_idx is not None:
                # If it's the first token of a word or a subsequent token, mark its position.
                last_token_position = word_idx_position
            if word_idx != previous_word_idx and previous_word_idx is not None:
                # Assign the label to the last token of the previous word.
                label_ids[last_token_position] = label[previous_word_idx]
            previous_word_idx = word_idx

        # Ensure the last word's label is also assigned
        if last_token_position is not None and previous_word_idx is not None:
            label_ids[last_token_position] = label[previous_word_idx]
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


epochs = 10
batch_size = 8
learning_rate = 5e-5
max_length = 512

model_id = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)

seqeval = evaluate.load("seqeval")

ds = load_cve()
label2id = {'O': 0, 'B-LOC': 1}
id2label = {v: k for k, v in label2id.items()}
label_list = list(label2id.keys())

tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

model = T5ForTokenClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="sc-ls-model",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()