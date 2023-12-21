import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from peft import LoraConfig, TaskType


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def preprocess_function(p):
    return tokenizer(p["text"], truncation=True, padding=True)


dataset = load_dataset('chathuru/SplunkAttackRangeAlerts')

model_name = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "BENIGN", 1: "MALICIOUS"}
label2id = {"BENIGN": 0, "MALICIOUS": 1}


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    load_in_8bit=True,
    num_labels=2,
)
model.config.use_cache = False


peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=8,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    target_modules=["q_lin", "k_lin", "v_lin"]
)
model.add_adapter(peft_config)

training_args = TrainingArguments(
    output_dir="CuATR-falcon7b",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    fp16=True,
    optim="paged_adamw_32bit",
    logging_steps=1,
    gradient_accumulation_steps=4,
    push_to_hub=True
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

config = AutoConfig.from_pretrained("tiiuae/falcon-7b")

trainer.push_to_hub("CuATR-falcon7b")
model.push_to_hub("CuATR-falcon7b")
model.config.push_to_hub("CuATR-falcon7b")

config.push_to_hub("CuATR-falcon7b")

trainer.save_model("CuATR-falcon7b")
tokenizer.save_pretrained("CuATR-falcon7b")
