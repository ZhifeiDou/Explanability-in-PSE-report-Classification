import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate


#----------------load esnli from the dataset---------------
#todo: ablation study with otehr advanced biomedical models
dataset = load_dataset("esnli")
biobert_ckpt = "dmis-lab/biobert-base-cased-v1.2"
tokenizer = AutoTokenizer.from_pretrained(biobert_ckpt)

#----------format the esnli dataset-----------------------
def preprocess_function(examples):
    result = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    result["labels"] = examples["label"]
    #print(result)
    return result

#-----------------map train, val, test dataset-----------------
encoded_train = dataset["train"].map(preprocess_function, batched=True)
encoded_val = dataset["validation"].map(preprocess_function, batched=True)
encoded_test = dataset["test"].map(preprocess_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    biobert_ckpt,
    num_labels=3
)

#-----------fine-tuning for the biobert------------------------
#todo: more detailed hyperparameter tuning for this biobert
training_args = TrainingArguments(
    output_dir="./bioNLI_ckpt",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5, #tuned
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3, #tuned
    weight_decay=0.01,
    logging_steps=100,
    push_to_hub=False
)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    #print(f'logits {logits}, predictions {predictions}')
    return accuracy_metric.compute(predictions=predictions, references=labels)

#train!
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
eval_results = trainer.evaluate(encoded_test)
print(f'the evaluation result is {eval_results}')
print("Test set evaluation:", eval_results)
