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
        with open(f'../data/{split_name}.jsonl', 'r') as reader:
            for line in reader:
                data.append(json.loads(line))
        ret[split_name] = Dataset.from_list(data)
    return DatasetDict(ret)

def tokenize_and_create_windows(examples):
    # Tokenize all texts first
    tokenized_inputs = tokenizer(examples["tokens"], truncation=False, is_split_into_words=True,
                                 return_overflowing_tokens=True, stride=stride, padding=False,
                                 return_offsets_mapping=True)
    offset_mapping = tokenized_inputs.pop("offset_mapping")

    # Initialize lists to hold split tokens and labels
    split_input_ids = []
    split_attention_masks = []
    split_labels = []

    for i, offsets in enumerate(offset_mapping):
        # Gather input_ids and attention_mask for the current example
        input_ids = tokenized_inputs["input_ids"][i]
        attention_mask = tokenized_inputs["attention_mask"][i]

        # Initialize list to hold labels for each token in the current example
        labels = []

        # Get the index of the original example this tokenized example is derived from
        sample_index = tokenized_inputs.overflow_to_sample_mapping[i]
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:  # Special tokens
                labels.append(-100)
            elif word_idx != previous_word_idx:  # First token of a new word
                labels.append(examples["ner_tags"][sample_index][word_idx])
            else:  # Subsequent subtokens of a word
                labels.append(-100)  # Ignore subtokens in label alignment
            previous_word_idx = word_idx

        # Split the tokenized example into windows if necessary
        for window_start in range(0, len(input_ids), max_length - stride):
            # Calculate window end index
            window_end = min(window_start + max_length, len(input_ids))

            # Slice the input_ids, attention_mask, and labels to get the current window
            window_input_ids = input_ids[window_start:window_end]
            window_attention_mask = attention_mask[window_start:window_end]
            window_labels = labels[window_start:window_end]

            # Append the current window to the lists
            split_input_ids.append(window_input_ids)
            split_attention_masks.append(window_attention_mask)
            split_labels.append(window_labels)

    # Prepare the final output dictionary
    output = {
        "input_ids": split_input_ids,
        "attention_mask": split_attention_masks,
        "labels": split_labels,
    }

    return output


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
stride = 256

model_id = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)

seqeval = evaluate.load("seqeval")

ds = load_cve()
label2id = {'O': 0, 'B-LOC': 1}
id2label = {v: k for k, v in label2id.items()}
label_list = list(label2id.keys())

tokenized_ds = ds.map(tokenize_and_create_windows, batched=True, remove_columns=['tokens', 'ner_tags'])
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

model = T5ForTokenClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="sliding-ls-model",
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