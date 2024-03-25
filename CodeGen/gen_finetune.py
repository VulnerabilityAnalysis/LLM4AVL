from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import get_peft_model, LoraConfig
from transformers import TrainingArguments
import numpy as np
import re

train_dataset = load_dataset("csv", data_files="../data/train.csv", split="train")
eval_dataset = load_dataset("csv", data_files="../data/valid.csv", split="train")

model_id = 'Salesforce/codegen-6B-mono'
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto').bfloat16()
peft_config = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.1, bias="none",)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
max_code_len = 512
max_loc_len = 256
epochs = 10
batch_size = 4
learning_rate = 5e-5


def add_line_numbers(function_code):
    lines = function_code.split('\n')
    modified_lines = []
    line_number = 0

    for line in lines:
        line_number += 1
        modified_lines.append(f"{line_number}: {line}")

    code = '\n'.join(modified_lines)
    return code

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['function'])):
        func, loc = add_line_numbers(example['function'][i]), example['location'][i]
        lines = func.split('\n')
        tgt = "### Vulnerable lines: "+'\n'.join([lines[i-1] for i in eval(loc)])

        src_ids = tokenizer(func, max_length=max_code_len, truncation=True, add_special_tokens=False)["input_ids"]
        src = tokenizer.decode(src_ids)
        tgt_ids = tokenizer(tgt, max_length=max_loc_len, truncation=True, add_special_tokens=False)["input_ids"]
        tgt = tokenizer.decode(tgt_ids)
        text = f"{src}\n\n{tgt}"
        # print(text)
        output_texts.append(text)
    return output_texts

response_template = "\n### Vulnerable lines:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
model_name = model_id.split("/")[-1]
training_args = TrainingArguments(
    output_dir=f"{model_name}-gen",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    push_to_hub=False,
)
trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator
)

trainer.train()