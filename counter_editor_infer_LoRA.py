import os
import re
import csv
import torch
import random
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import PeftModel

#----------parameter-----------------------------------------
#todo: test more topp and temps for the generator, assume that for the effect of counterfactal, a 3.0 temp is good, but we need more test on this
_EXTRA_TOKEN_INFILL = "<extra_id_0>"

#---------------load editor----------------------------------
def load_editor_model(editor_path, device="cuda"):
    base_model_name = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
    editor_model = PeftModel.from_pretrained(base_model, editor_path)
    editor_model.to(device)
    editor_model.eval()
    return editor_model, tokenizer

#--------------insert masks----------------------------------
def insert_mask_random_position(text):
    words = text.split()
    if len(words) < 4:
        return None
    pos = random.randint(1, len(words) - 1)
    words.insert(pos, _EXTRA_TOKEN_INFILL)
    return " ".join(words)

#-------------generate with editor-----------------------------------------
#todo: even the original author does not suggest how to get the best effort for editing, how do tune this?
def generate_counterfactual_text(editor_model, tokenizer, masked_text, device="cuda"):
    if not masked_text:
        return None
    prompt = f"insert: {masked_text}"
    enc = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = editor_model.generate(
            **enc,
            max_new_tokens=20,
            do_sample=True,
            top_p=0.9, #todo: tuning
            temperature=3.0 #todo: tuing
        )
    fill_str = tokenizer.decode(out[0], skip_special_tokens=True)
    return masked_text.replace(_EXTRA_TOKEN_INFILL, fill_str)

#--------------main---------------------------------------
#todo:
def main():
    model_name = "deepseek_distilled_Qwen14B"
    editor_dir = "editor_ckpt_" + model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(42) #fix seed for reproduction
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    df_infer = pd.read_csv(
        "PSE_test_data.csv",
        header=None,
        names=["label_num", "content", "label", "content_redundant"],
        encoding="utf-8"
    )
    df_infer.drop(columns=["content_redundant"], inplace=True, errors="ignore")
    editor_model, editor_tokenizer = load_editor_model(editor_dir, device=device)
    edited_contents = []
    for _, row in tqdm(df_infer.iterrows(), total=len(df_infer), desc="Editing"):
        orig_text = str(row["content"]).strip()
        masked_text = insert_mask_random_position(orig_text)
        if masked_text is None:
            edited_contents.append(orig_text)
            continue
        cf_text = generate_counterfactual_text(editor_model, editor_tokenizer, masked_text, device=device)
        if cf_text is None:
            cf_text = orig_text
        edited_contents.append(cf_text) #add the edited content
    df_infer["content"] = edited_contents
    df_infer["content_redundant"] = ""
    counter_file_name = f"PSE_counter_test_data_{model_name}.csv"
    df_infer.to_csv(counter_file_name, index=False, header=False, encoding="utf-8")

if __name__ == "__main__":
    main()
