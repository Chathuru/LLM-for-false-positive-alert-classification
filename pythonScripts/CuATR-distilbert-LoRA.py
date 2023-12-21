# -*- coding: utf-8 -*-
"""CuATR-distilbert.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kN5vVL936CTwkGSBLmD-6bkR-OPNVLX2
"""

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, TaskType


def preprocess_function(data):
    return tokenizer(data["text"], truncation=True, padding=True)

dataset = load_dataset('chathuru/SplunkAttackRangeAlerts')

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "BENIGN", 1: "MALICIOUS"}
label2id = {"BENIGN": 0, "MALICIOUS": 1}

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# accuracy = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=labels)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)
model.config.use_cache = False

peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=8,
    bias="none",
    task_type=TaskType.TOKEN_CLS
)
model.add_adapter(peft_config)

training_args = TrainingArguments(
    output_dir="CuATR-distilbert-LoRA",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    fp16=True,
    optim="paged_adamw_32bit",
    logging_steps=1,
    gradient_accumulation_steps=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("CuATR-distilbert-LoRA")
trainer.push_to_hub("CuATR-distilbert-LoRA")