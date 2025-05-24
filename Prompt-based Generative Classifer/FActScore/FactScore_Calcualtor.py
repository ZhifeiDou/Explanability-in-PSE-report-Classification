import os
import re
import csv
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import PeftModel
import random
import numpy as np
from pyserini.search.lucene import LuceneSearcher


#settings-for-reproduction-----------
#todo:For the summer research expand this seed to 5 for average the performance of each model
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

#---model control function----------
MODEL = "Qwen14B"
if MODEL == "Qwen32B":
    MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
    SAVE_DIRECTORY = "local_Qwen_32B"
elif MODEL == "Qwen14B":
    MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
    SAVE_DIRECTORY = "local_Qwen_14B"
elif MODEL == "deepseek_distilled_Qwen32B":
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    SAVE_DIRECTORY = "deepseek_local_Qwen_32B"
elif MODEL == "deepseek_distilled_Qwen14B":
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    SAVE_DIRECTORY = "deepseek_local_Qwen_14B"
elif MODEL == 'QwQ':
    MODEL_NAME = "Qwen/QwQ-32B"
    SAVE_DIRECTORY = "local_QwQ"
else:
    raise ValueError(f"Could not load the model '{MODEL}'.")

#-------parameter controlling function-----------
TOPP = 0.9
TEMP = 0.7
TEST_CSV_PATH = "PSE_test_data.csv"
#OUTPUT_CSV = f"factscore_result/factscore_assist{MODEL}_p{TOPP}_t{TEMP}_seed{SEED}.csv"


#-----Quantization----------------------------------
#only 4bit would allow 32B models to store on GPU vram
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

#-----load model-------------------------------------
#HF load
if os.path.exists(SAVE_DIRECTORY):
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, SAVE_DIRECTORY)
    loaded_online = False
#local load
#todo: find where is this sotring at, may be delete win to release some space for this
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"
    )
    loaded_online = True

#----------------------load the wiki indexer-------------------------------------------
#todo: investigate the performance impact from the hyperparameters
INDEX_DIR = "enwiki-index-storeraw"
searcher = LuceneSearcher(INDEX_DIR)
searcher.set_bm25(k1=1.2, b=0.75)


#-------------------define a independent generation function------------------------------
def run_generation(prompt_text: str, max_tokens=512):
    if MODEL in ["QwQ", "Qwen32B", "Qwen14B"]:
        messages = [{"role": "user", "content": prompt_text}]
        text_for_tokenizer = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text_for_tokenizer], return_tensors="pt")
    else:
        inputs = tokenizer(prompt_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k,v in inputs.items()}
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=TOPP,
            temperature=TEMP
        )
    out_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    #print(f'the output text is: {out_text}')
    return out_text

#--------------------Naive atomic breaking-------------------------------------------------
#todo: as the original author does not give clear guidelines about prompting and tuning for
#the atomic fact breaker, more investigation is needed to explore the impact on performance
#with utilzation of Qwen models
def break_explanation_into_facts(explanation: str) -> list:
    #print(explanation)
    prompt_for_atomic = f"""
        Please break this explanation into separate atomic facts,
        each containing only one piece of information. Example:

        Original: "She was an American nurse who traveled to Europe in 2012."
        Atomic facts:
        1) She was an American nurse.
        2) She traveled to Europe in 2012.

        Now do this for:
        \"{explanation}\"
    """
    generation = run_generation(prompt_for_atomic, max_tokens=300)
    lines = generation.split("\n")
    facts = []
    for line in lines:
        line_clean = re.sub(r"^\d+\)\s*", "", line.strip())
        if len(line_clean) > 5:
            facts.append(line_clean)
    #print(f'THE FACTS ARE:{facts}')
    return facts

#------------------Retrieval function for wikifacts retrieve--------------------------------
def check_fact_support(fact: str) -> bool or None:
    top_k = 1
    hits = searcher.search(fact, k=top_k)
    retrieved_texts = []
    max_retrived_char_length = 100130 #max length Qwen14B can handle from experiment
    #max_retrived_char_length = 100
    for i in range(min(top_k, len(hits))):
        doc = searcher.doc(hits[i].docid)
        if doc:
            content_str = doc.raw()
            if content_str is not None:
                if len(content_str) > max_retrived_char_length:
                    content_str = content_str[:max_retrived_char_length]
                retrieved_texts.append(content_str)
                #print(f'THe length of content string is: {len(content_str)}')
    #print(f'THE RETRIVED TEXT IS: {retrieved_texts}')
    combined_passages = "\n".join(retrieved_texts[:top_k])
    #print(f'The combined_passage is {combined_passages}')
    #print('------------------------------------------------------------------------------')
    #print(f'The fact is {fact}')
    verify_prompt = f"""
    I have the following short fact:
    "{fact}"

    And here are some passages from Wikipedia:
    {combined_passages}

    Based on these passages, is the fact "True" or "False"?
    Please answer with a single word "True" or "False".
    """
    #print(f"PROMPT FOR JUDGE {verify_prompt}")
    verdict_text = run_generation(verify_prompt, max_tokens=50)
    #print(f"OUTPUT FOR JUDGE {verdict_text}")

    lines = verdict_text.strip().splitlines()
    last_line = lines[-1].strip().lower() if lines else ""
    if last_line == "true":
        return True
    elif last_line == "false":
        return False
    #print(f'Assist failed to give answer judge, the last line is {last_line}')
    return None


#---------------------main function to run the file-----------------------------------------------------
for filename in os.listdir('output'):
    if filename.lower().endswith('.csv') and not filename.endswith('_counter.csv'):
        #print(f'the file processing is {filename}')
        filepath = os.path.join('output', filename)
        df_pred = pd.read_csv(filepath)
        file_name_without_ext = os.path.basename(filepath)
        file_name_without_ext = os.path.splitext(file_name_without_ext)[0] #only take the middle of the file name
        joined_output_path = f"factscore_result/factscore_{file_name_without_ext}_(assist{MODEL}_p{TOPP}_t{TEMP}_seed{SEED}).csv"
        if os.path.exists(joined_output_path): #continue logic, if the output file detected, skip to process this file
            continue
        with open(joined_output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(['content', 'predicted_class', 'predicted_explanation', 
                            'fact_score', 'num_facts', 'num_supported_facts'])
            
        for _, row in tqdm(df_pred.iterrows(), total=len(df_pred), desc="FActScore Inference"):
            content = row['content']
            prediction = row['predicted_class']
            explanation = row["predicted_explanation"]
            #print(content, prediction, explanation)
            atomic_facts = break_explanation_into_facts(explanation)
            #print(atomic_facts)
            supported_count = 0
            valid_facts = 0
            for fact in atomic_facts:
                verdict = check_fact_support(fact)
                if verdict is True:
                    #print(f'good!')
                    supported_count += 1
                    valid_facts += 1
                elif verdict is False:
                    valid_facts += 1
                else:
                    continue
            #total_facts = len(atomic_facts)
            fact_score = 0.0
            if valid_facts > 0:
                fact_score = supported_count / valid_facts
            with open(joined_output_path, "a", newline="", encoding="utf-8") as f2:
                writer2 = csv.writer(f2)
                writer2.writerow([
                    content,
                    prediction,
                    explanation,
                    f"{fact_score:.2f}",
                    valid_facts,
                    supported_count
                ])
    else:
        continue

print(f"FActScore computation complete. Results written to {joined_output_path}.")
