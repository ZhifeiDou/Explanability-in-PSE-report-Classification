import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#--------initialze for prediction-------------------------------------
id2label = {0: "neutral", 1: "contradiction", 2: "entailment"}
#model_dir = "bioNLI_ckpt/checkpoint-68672"
model_dir = "bioNLI_ckpt/checkpoint-34336"
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
directory = 'output'
save_directory = 'biobert_entailment_evaluate_result'

#--------------loop throught the file to predict---------------
#todo: what is the impact on max length to the performance?
for filename in os.listdir(directory):
    if filename.endswith('.csv') and not filename.endswith('_counter.csv'): #skip counter files
        #print(f'the file processing is {filename}')
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        inference_results = []
        for i, row in df.iterrows():
            premise = row["content"]
            #print(f'the premise is {premise}')
            hypothesis = row["predicted_explanation"]
            #print(f'The hypothesis is {hypothesis}')
            inputs = tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=128 #impact on performance?
            )
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predicted_label_id = logits.argmax(axis=-1).item()
            predicted_label = id2label[predicted_label_id]
            inference_results.append(predicted_label)
        #print(inference_results)
        df["bioBERT_nli_label"] = inference_results
        output_file = os.path.join(save_directory, f"biobert_entailment_{filename}")
        df.to_csv(output_file, index=False)
        print(f"Saved inference results to {output_file}")


#-----------------calcualte the contradiction prediction------------------------
results = []
for filename in os.listdir(save_directory):
    if filename.endswith('.csv') and not filename.endswith('_counter.csv'):
        file_path = os.path.join(save_directory, filename)
        df = pd.read_csv(file_path)
        if 'bioBERT_nli_label' in df.columns:
            contradiction_count = (df['bioBERT_nli_label'] == 'contradiction').sum()
            results.append([filename, contradiction_count])

results_df = pd.DataFrame(results, columns=['filename', 'contradiction_count'])
results_df.to_csv('contradiction_summary.csv', index=False)
print("Saved summary to contradiction_summary.csv")
