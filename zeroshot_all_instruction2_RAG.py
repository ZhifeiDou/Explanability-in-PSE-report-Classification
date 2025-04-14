import os
import re
import csv
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed

)
from peft import PeftModel
import faiss
import random
import numpy as np


#settings-for-reproduction-----------
#todo:For the summer research expand this seed to 5 for average the performance of each model
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

#-------retrieval path for training datasets--------------
#todo: ablation test for various embedding representations hongbo find out RoBERTa
#       with SVM has the highest accuray, maybe try RoBERTa
RETRIEVAL_CSV_PATH = "PSE_train_cv_data.csv"
EMBEDDING_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

#---model control function----------
MODEL = "deepseek_distilled_Qwen32B"
if MODEL == "Qwen32B":
    MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
    SAVE_DIRECTORY = "local_Qwen_32B"
elif MODEL == "Qwen14B":
    MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
    SAVE_DIRECTORY = "local_Qwen_14B"
elif MODEL == "Llama8B":
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    SAVE_DIRECTORY = "local_Llama_8B"
elif MODEL == "deepseek_distilled_Qwen32B":
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    SAVE_DIRECTORY = "deepseek_local_Qwen_32B"
elif MODEL == "deepseek_distilled_Qwen14B":
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    SAVE_DIRECTORY = "deepseek_local_Qwen_14B"
elif MODEL == 'deepseek_distilled_Llama8B':
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    SAVE_DIRECTORY = "deepseek_local_Llama_8B"
elif MODEL == 'MaLlama':
    MODEL_NAME = "YBXL/Med-LLaMA3-8B"
    SAVE_DIRECTORY = "local_Mdllama"
elif MODEL == 'QwQ':
    MODEL_NAME = "Qwen/QwQ-32B"
    SAVE_DIRECTORY = "local_QwQ"   
else:
    raise ValueError(
        f"Could not load the model '{MODEL}'. "
        "Please check that it exists or is spelled correctly."
    )



#-------parameter controlling function-----------
TOPP = 0.9
TEMP = 0.7



#--------counter factual editor function------------
#check everytime before run this code, True is for counter, false is not counter
COUNTER = False
if COUNTER:
    TEST_CSV_PATH = f"counter_data/PSE_counter_test_data_{MODEL}.csv"
    OUTPUT_CSV = f"output/output_RAG_instruction2_{MODEL}_p{TOPP}_t{TEMP}_seed{SEED}_counter.csv"

else:
    TEST_CSV_PATH = "PSE_test_data.csv"
    OUTPUT_CSV = f"output/output_RAG_instruction2_{MODEL}_p{TOPP}_t{TEMP}_seed{SEED}.csv"
    #OUTPUT_CSV = "test.csv"


#-------------LOAD RETRIEVAL DATASETS-------------------------
retrieval_df = pd.read_csv(
    RETRIEVAL_CSV_PATH,
    header=None,
    names=["label_num", "content", "label", "content_redundant"],
    encoding="utf-8"
)
retrieval_df.drop(columns=["content_redundant"], inplace=True)
retrieval_dataset = Dataset.from_pandas(retrieval_df)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
#print(device)
embedding_model.to(device)
embedding_model.eval()


#todo: is 512 impacting the retrieval performance?
def embed_text_batch(text_list):
    #in text, out embedding as numpy
    inputs = embedding_tokenizer(text_list, padding=True, truncation=True, max_length = 512, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = embedding_model(**inputs)
        #print(outputs.shape[])
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy()

def embed_docs(examples):
    return {"embeddings": embed_text_batch(examples["content"])}


#--------faiss retrieval---------------------------------------------------------
retrieval_dataset = retrieval_dataset.map(embed_docs, batched=True, batch_size=16)
retrieval_dataset.add_faiss_index(column="embeddings")

def retrieve_similar_rows(query_text, top_k=2):
    query_emb = embed_text_batch([query_text])
    scores, retrieved = retrieval_dataset.get_nearest_examples(
        "embeddings",
        query_emb,
        k=top_k
    )
    results = []
    for idx in range(len(retrieved["content"])):
        row_content = retrieved["content"][idx]
        row_label = retrieved["label"][idx]
        row_label_num = retrieved["label_num"][idx]
        results.append({
            "label_num": row_label_num,
            "content": row_content,
            "label": row_label
        })
    #print(results)
    return results


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


#---------LOAD DATA--------------------------------------
df_infer = pd.read_csv(
    TEST_CSV_PATH,
    header=None,
    names=["label_num", "content", "label", "content_redundant"],
    encoding="utf-8"
)
df_infer.drop(columns=["content_redundant"], inplace=True)


#---------INITIALIZE OUTPUT FILE-------------------------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["content", "predicted_class", "predicted_explanation"])




#-------MODEL GENERATION-----------------------------------\
#todo: prompt engineering and robust test for prompt on faithfulness during summer.
#todo: investagte on LLama
#todo: prompt engineering for retrieval, what is the impact for prompt on the retrieval embedding?
for _, row in tqdm(df_infer.iterrows(), total=len(df_infer), desc="Inferencing"):
    content = row["content"]
    top_rows = retrieve_similar_rows(content, top_k=2)
    retrieved_text_block = ""
    for r in top_rows:
        retrieved_text_block += f"- Content: {r['content']}\n"
        retrieved_text_block += f"  Label: {r['label']}\n\n"
    #print('the retrieved block is',retrieved_text_block)
    #print('the content is:',content)
    prompt = f"""
        You are a medical classification assistant. 
        Below are some relevant records from our training dataset (content + label):
        {retrieved_text_block}

        Your task is to read the text below and decide which of the following categories best applies:

        Categories (with brief definitions):
        
        Care coordination / communication: Incidents in this category arise from failures in communication or care coordination among healthcare providers, patients, or care teams. These include miscommunications (e.g. during hand-offs, shift changes, or referrals) and poor coordination of care activities, which can lead to errors or omissions. A communication breakdown is any discrepancy in conveying the treatment plan between caregivers or between caregiver and patient. In practice, this might involve a critical test result not being passed to the responsible physician or conflicting information given to a patient during transitions of care. Effective care coordination is the deliberate organization of patient care activities among two or more participants (including the patient) to facilitate appropriate delivery of health services. Example: A failed hand-off where a surgeon is not informed of a patients allergy is a communication error that can jeopardize patient safety.
        
        Laboratory test: Definition: Laboratory test-related incidents encompass errors throughout the lab testing process from test ordering and specimen collection to analysis, result reporting, and interpretation. A laboratory error is defined as “any defect from ordering tests to reporting results and appropriately interpreting and reacting on these”. This means mistakes can occur in the pre-analytic phase (e.g. mislabeling a blood sample), the analytic phase (incorrect calibration of an analyzer), or the post-analytic phase (delayed or erroneous reporting of results). Such errors can lead to misdiagnosis or improper treatment. Example: A blood sample being swapped between patients (leading to a patient receiving anothers lab results) is a lab incident that can result in incorrect diagnosis or treatment.
        
        Medication related: Medication-related incidents are events involving medications that result in patient harm or have the potential to do so. This includes medication errors - any preventable event that may cause or lead to inappropriate medication use or patient harm while the medication is in the control of a healthcare professional, patient, or consumer. These events may be related to prescribing, order communication, product labeling/packaging, dispensing, administration, or monitoring of drugs. They also include adverse drug events and adverse drug reactions. An adverse drug event (ADE) is harm caused by the use of a medication, and an adverse drug reaction (ADR) is a harmful or unpleasant reaction resulting from the use of a medication at normal doses (for example, an anaphylactic reaction to penicillin). Example: A nurse administering 10-fold the intended dose of insulin (a dosing error) leading to hypoglycemia is a medication-related incident.
        
        Omission / errors in assessment or diagnosis or monitoring: These incidents involve failures to properly assess a patient, errors in diagnosis, or inadequate monitoring of a patient's condition. An error of omission is a key concept here - it is an error that results from not taking an action that should have been taken. This can include missing a critical step in patient assessment, failing to order a necessary diagnostic test, not making a timely diagnosis, or not monitoring a patient's status when indicated. Diagnostic errors (missed, wrong, or delayed diagnoses) fall into this category as well. Example: A patient presenting with chest pain who is not evaluated for a possible heart attack (failure to assess and diagnose) represents an omission error that can lead to serious harm. In The Joint Commissions terms, failing to perform an indicated intervention (e.g. delaying a needed emergency cesarean section) or missing a crucial abnormal lab result are errors of omission that may result in adverse outcomes.
        
        Maternal: Maternal safety incidents refer to events that result in harm to a woman during pregnancy, childbirth, or the postpartum period. This includes severe maternal complications and maternal death. The World Health Organization (WHO) defines a maternal death as “the death of a woman while pregnant or within 42 days of termination of pregnancy… from any cause related to or aggravated by the pregnancy or its management, but not from accidental or incidental causes”. In a patient safety context, maternal incidents often involve unanticipated outcomes of labor and delivery. The Joint Commission, for instance, considers any intrapartum maternal death (death during childbirth) a sentinel event, as well as severe maternal morbidity that results in permanent harm. Examples: Maternal hemorrhage requiring emergency intervention, eclampsia (seizures related to high blood pressure in pregnancy), or cardiac arrest in a pregnant patient are maternal incidents. An intrapartum maternal death or a life-threatening postpartum complication (like an uncontrolled bleed) would be among the most serious maternal events.
        
        Equipment / devices:  This category covers incidents involving medical equipment or devices. It includes any event in which a medical device fails, malfunctions, or is used incorrectly in a way that endangers patient safety. According to AHRQ's Common Formats, a device or medical/surgical supply event involves “a defect, failure, or incorrect use of a device” (including health information technology devices) that results in harm or has the potential to harm a patient. These events range from infusion pumps delivering the wrong dose due to malfunction, to equipment not being available when needed, or improper use of devices by staff. Example: A cardiac monitor that freezes or displays incorrect readings, leading clinicians to miss a patient's arrhythmia, is an equipment-related incident. Similarly, a ventilator that stops working (equipment failure) or a tubing misconnection causing delivery of gas to the wrong route are device incidents that can severely harm patients.
        
        Supplies: Supplies-related incidents are safety events involving medical or surgical supplies (disposable or reusable items, implants, etc.), excluding major equipment. This can include using defective or expired supplies, contamination of supplies, or lack of a critical supply when needed. Essentially, any product/equipment/supply management issue that compromises patient care falls here. Examples: A sterile surgical pack that is compromised (e.g., a tear in packaging leading to contamination) and still used can cause infection - a supplies incident. Another example is a shortage or unavailability of a necessary supply (like oxygen canisters or necessary implants) during a procedure, leading to an unsafe makeshift solution or delay in care.(Note: Many taxonomies group “Equipment/Devices” and “Supplies” together because both involve material resources used in care. For clarity, we separate them here: devices typically have mechanical or electronic function, whereas supplies are expendable or single-use items.)
        
        **Instructions**:
        1. Read the text carefully and identify the key points or keywords that hint at one of the above categories.
        2. Decide on the single most relevant category.
        3. Provide a short explanation of how you arrived at this decision, referencing specific keywords or ideas from the text.
        4. Think step by step like a professional doctor
        5. Return the result in **exactly** the following Markdown format:

        **Class:** <Your single best class here>  
        **Explanation:** <Your explanation here>

        Text: {content}
        """

    if MODEL == "QwQ" or "Qwen32B" or "Qwen14B":#accommodate for template of QwQ
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        #print(f'The input prompt is:{text}')
        inputs = tokenizer([text], return_tensors="pt")
    
    else:
        inputs = tokenizer(prompt, return_tensors="pt")    
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens = 3000, #accommodate for QwQ
            do_sample=True,
            top_p=TOPP,
            temperature=TEMP
        )
    
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    #print(output_text)
    pattern = r"\*\*Class:\*\*\s*(.*?)\s*\*\*Explanation:\*\*\s*(.*?)(?=\n\*\*Class:\*\*|$)"
    pairs = re.findall(pattern, output_text, flags=re.DOTALL)
    if pairs:
        final_class, final_explanation = pairs[-1]
        # Take the last match in case there's more than one
        predicted_class = final_class.strip()
        predicted_explanation = final_explanation.strip()
    else:
        predicted_class = ""
        predicted_explanation = ""
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([content, predicted_class, predicted_explanation])

print(f"Inference complete for {MODEL}. Results written to {OUTPUT_CSV}.")
