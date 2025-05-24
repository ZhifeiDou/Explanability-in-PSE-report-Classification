#############################################################################################################################
#This code is modified from: https://github.com/copenlu/nle_faithfulness
#Modification: Replaced teacher model with more sophescitated models, QLoRA fine-tune the T5 to allow 32B teacher load in 4090
#############################################################################################################################
'''
Code usage:
python counter_editor_4bitLoRA.py \
  --csv_path PSE_train_cv_data.csv \
  --save_dir editor_ckpt_Qwen32B \
  --num_train_epochs 3 \
  --train_batch_size 1 \
  --lr 1e-4 \
  --bnb_4bit True \
  --base_model Qwen32B

python counter_editor_4bitLoRA.py \
  --csv_path PSE_train_cv_data.csv \
  --save_dir editor_ckpt_deepseek_distilled_Qwen32B \
  --num_train_epochs 3 \
  --train_batch_size 1 \
  --lr 1e-4 \
  --bnb_4bit True \
  --base_model deepseek_distilled_Qwen32B

python counter_editor_4bitLoRA.py \
  --csv_path PSE_train_cv_data.csv \
  --save_dir editor_ckpt_QwQ \
  --num_train_epochs 3 \
  --train_batch_size 1 \
  --lr 1e-4 \
  --bnb_4bit True \
  --base_model QwQ

python counter_editor_4bitLoRA.py \
  --csv_path PSE_train_cv_data.csv \
  --save_dir editor_ckpt_Qwen14B \
  --num_train_epochs 3 \
  --train_batch_size 1 \
  --lr 1e-4 \
  --bnb_4bit True \
  --base_model Qwen14B


python counter_editor_4bitLoRA.py \
  --csv_path PSE_train_cv_data.csv \
  --save_dir editor_ckpt_deepseek_distilled_Qwen14B \
  --num_train_epochs 3 \
  --train_batch_size 1 \
  --lr 1e-4 \
  --bnb_4bit True \
  --base_model deepseek_distilled_Qwen14B

'''

import argparse
import copy
import os
import random
import json
import torch
import numpy as np
import pandas as pd
import re
import logging
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModel, TaskType



#-----------Parameter setting-------------------------
_EXTRA_TOKEN_INFILL = "<extra_id_0>"
TEACHER_TOPP = 0.9
TEACHER_TEMP = 0.7

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#-----------set the mannual seed------------------------
def enforce_reproducibility(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

#-----------read the data used to train the counterfactual editor---------------------------
def read_pse_data(csv_path="PSE_train_cv_data.csv", max_samples=2000):
    df = pd.read_csv(csv_path, header=None, names=["label_num","content","label","redundant"], encoding="utf-8")
    df.drop(columns=["redundant"], inplace=True, errors="ignore")
    #print(df.head())
    examples = []
    for i, row in df.iterrows():
        if i >= max_samples:
            break
        ex = {
            "idx": i,
            "content": str(row["content"]),
            "original_label": str(row["label"]),
        }
        examples.append(ex)
    #print(examples)
    return examples

#-------------randomly add masked content into the original content----------------------------------------
def add_insertions_pse(examples):
    #print(examples)
    for ex in examples:
        ex["masked_instances"] = []
        masked_text = _insert_random(ex["content"])
        if masked_text is not None:
            inst = copy.deepcopy(ex)
            inst["content"] = masked_text
            inst["original_content"] = ex["content"]
            inst["masked"] = True
            inst["replacement"] = None
            ex["masked_instances"].append(inst)
    #print(examples)
    return examples


#----------------mask the content------------------------------------
def _insert_random(text, mask_token=_EXTRA_TOKEN_INFILL):
    #print(text)
    words = text.split()
    if len(words) < 4: #no mask less than 4 to align with the original work
        return None
    pos = random.randint(1, len(words)-1)
    words.insert(pos, mask_token)
    #print(" ".join(words))
    return " ".join(words)


#---------------load the teacher model which------------------------
#NOTICE: must align with the model needs to be tested
def load_qwen_teacher(teacher_model, bnb_4bit=True):
    if teacher_model == "deepseek_distilled_Qwen14B":
        DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    elif  teacher_model ==   "deepseek_distilled_Qwen32B":
        DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    elif teacher_model == "Qwen14B":
        DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
    elif teacher_model == "Qwen32B":
        DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct" 
    elif teacher_model == "Llama8B":
        DEFAULT_MODEL = 'meta-llama/Llama-3.1-8B'
    elif teacher_model == "QwQ":
        DEFAULT_MODEL = 'Qwen/QwQ-32B'
    MODEL_NAME = DEFAULT_MODEL
    #print(f"the teacher model loaded is {MODEL_NAME}")
    if bnb_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        teacher_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map="auto"
        )
    teacher_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    teacher_model.eval()
    return teacher_model, teacher_tokenizer



#--------------------prompt the teacher model to predict and generate explanation-----------------------------------
#todo: 1000 is the maxima for token length for how if load 32B and the T5 same time, try longer when 5090 arrived.
#todo: after the teacher model is fed with generated word content, for some reason it always return 'assistent:', needs investiagate
def model_specific_predictions_explanations(args, in_filled_examples, teacher_model, teacher_tokenizer):
    teacher_predictions = []
    teacher_explanations = []
    for ex in in_filled_examples:
        content_text = ex["content"]
        #print(f'The content is {content_text}')
        prompt = f"""
                You are a medical classification assistant. 
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

                Text: {content_text}
                """
        
        #apply template
        if teacher_model in ["Qwen14B" , "Qwen32B",'QwQ']:
            messages = [{"role": "user", "content": prompt}]
            text = teacher_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = teacher_tokenizer([text], return_tensors="pt")
            #print(inputs)
        else:
            inputs = teacher_tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            for k,v in inputs.items():
                inputs[k] = v.to("cuda")
        with torch.no_grad():
            out_tokens = teacher_model.generate(
                **inputs,
                max_new_tokens=1000, #1000 is maximum working here
                do_sample=True,
                top_p=TEACHER_TOPP,
                temperature=TEACHER_TEMP
            )
        out_text = teacher_tokenizer.decode(out_tokens[0], skip_special_tokens=True)
        logger.info(f"Teacher model output: {out_text} for input: {content_text}")
        
        
        pattern = r"(?:Assistant:\s*)?\*\*Class:\*\*\s*(.*?)\s*\*\*Explanation:\*\*\s*(.*?)(?=\n(?:Assistant:\s*)?\*\*Class:\*\*|$)"
        pairs = re.findall(pattern, out_text, flags=re.DOTALL)
        #pattern = r"\*\*Class:\*\*\s*(.*?)\s*\*\*Explanation:\*\*\s*(.*?)(?=\n\*\*Class:\*\*|$)"
        #pairs = re.findall(pattern, out_text, flags=re.DOTALL)
        #pattern2 = r"Assistant: \*\*Class:\*\*\s*(.*?)\s*\*\*Explanation:\*\*\s*(.*?)(?=\n\*\*Class:\*\*|$)"
        #pairs2 = re.findall(pattern2, out_text, flags=re.DOTALL)

        if pairs:
            final_class, final_explanation = pairs[-1]
            pred_class = final_class.strip()
            pred_expl = final_explanation.strip()
        
        #elif pairs2:
        #    final_class, final_explanation = pairs2[-1]
        #    pred_class = final_class.strip()
        #    pred_expl = final_explanation.strip()
        
        else:
            pred_class = "UNKNOWN"
            pred_expl = "No explanation found"
        teacher_predictions.append(pred_class)
        teacher_explanations.append(pred_expl)
        logger.info(f"teacher_predictions: {pred_class} teacher_explanations: {pred_expl}")
    return teacher_explanations, teacher_predictions


#-----------------collate function for the report------------------------------------------------------------
def pse_collate_fn(batch, tokenizer, device="cuda"):
    inputs = []
    labels = []
    original_examples = []
    for ex in batch:
        inp_text = f"insert: {ex['content']}"
        tgt_text = "SOME_FILL" #as the instruction from original paper and the code, the target should be a dummy target like this
        inputs.append(inp_text)
        labels.append(tgt_text)
        original_examples.append(ex)
    #print(f'the inputs are {inputs})
    #print(f'the labels are {labels})
    tokenized_in = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
    tokenized_out = tokenizer(labels, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = tokenized_in["input_ids"]
    attention_mask = tokenized_in["attention_mask"]
    label_ids = tokenized_out["input_ids"].clone()
    label_ids[label_ids == tokenizer.pad_token_id] = -100 #<-------
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": label_ids.to(device),
        "original_examples": original_examples
    }


#-----------------lets train----------------------------------------------------------------------
def train_editor(args, editor_model, tokenizer, train_data, teacher_model, teacher_tokenizer):
    from torch.utils.data import DataLoader
    editor_model.train()
    collate_fn = partial(pse_collate_fn, tokenizer=tokenizer, device=args.device)
    loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(editor_model.parameters(), lr=args.lr)
    for epoch in range(args.num_train_epochs):
        total_loss = 0.0
        for batch_dict in tqdm(loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch_dict["input_ids"]
            attention_mask = batch_dict["attention_mask"]
            labels = batch_dict["labels"]
            original_examples = batch_dict["original_examples"]
            out = editor_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            fill_loss = out.loss #<-----------------filling loss
            #print(fill_loss)
            logits_infill = out.logits
            imitation_loss_val = 0.0
            adv_value = 0.0
            
            #teacher model generate with editor model's mask fillings
            with torch.no_grad():
                gen_out = editor_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=5
                )
                #print(f'the editor model generatrion is {gen_out}')
                fill_texts = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
                logger.info(f"Original ex: {original_examples} \nEditor gen: {fill_texts} (epoch {epoch+1})")
                in_filled = []
                for i, ex in enumerate(original_examples):
                    ex_copy = copy.deepcopy(ex)
                    ex_copy["replacement"] = fill_texts[i]
                    ex_copy["content"] = ex_copy["content"].replace(_EXTRA_TOKEN_INFILL, fill_texts[i])
                    in_filled.append(ex_copy)
                teacher_expls, teacher_preds = model_specific_predictions_explanations(
                    args,
                    in_filled,
                    teacher_model,
                    teacher_tokenizer
                ) #get teacher models generation
                for j, exf in enumerate(in_filled):
                    exf["teacher_explanation"] = teacher_expls[j]
                    exf["teacher_label"] = teacher_preds[j]
            student_inputs = []
            student_targets = []
            for exf in in_filled:
                s_inp = f"explain: {exf['content']}"
                s_tar = exf["teacher_explanation"]
                student_inputs.append(s_inp)
                student_targets.append(s_tar)
            
            
            #student (editor) model to learn with teacher forcing on teacher's explanation
            if len(student_inputs) > 0:
                s_inp_enc = tokenizer(student_inputs, padding=True, truncation=True, max_length=512, return_tensors="pt")
                s_tar_enc = tokenizer(student_targets, padding=True, truncation=True, max_length=512, return_tensors="pt")
                s_labels = s_tar_enc["input_ids"].clone()
                s_labels[s_labels == tokenizer.pad_token_id] = -100
                s_inp_enc = {k: v.to(args.device) for k,v in s_inp_enc.items()}
                s_out = editor_model(
                    input_ids=s_inp_enc["input_ids"],
                    attention_mask=s_inp_enc["attention_mask"],
                    labels=s_labels.to(args.device)
                )
                imitation_loss_val = s_out.loss.item() #<----------------imitation loss
                expl_logits = s_out.logits
                fill_mean = torch.mean(logits_infill.float(), dim=[1,2])
                expl_mean = torch.mean(expl_logits.float(), dim=[1,2])
                diff = torch.abs(fill_mean - expl_mean)
                adv_value = -torch.mean(diff) #<--------------------adv loss
            logger.info(f"Filling loss: {fill_loss}, Imitation loss: {imitation_loss_val}, Adv loss: {adv_value}")
            
            
            #calculate the total loss
            total_loss_val = (
                args.filling_loss_weight * fill_loss +
                args.imitation_loss_weight * imitation_loss_val +
                args.adversary_loss_weight * adv_value
            )
            optimizer.zero_grad()
            total_loss_val.backward() #back propogate
            optimizer.step()
            total_loss += total_loss_val.item()
        logger.info(f"Epoch {epoch+1}, avg loss: {total_loss / len(loader):.4f}")

#-----------------fill in masks with the editor model-----------------------
def generate_alternative_input(editor_model, tokenizer, text, device="cuda"):
    words = text.split()
    if len(words) < 4:
        return text
    pos = random.randint(1, len(words)-1)
    words.insert(pos, _EXTRA_TOKEN_INFILL)
    masked_text = " ".join(words)
    prompt = f"insert: {masked_text}"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = editor_model.generate(
            **enc,
            max_new_tokens=20,
            num_beams=3,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )
    fill_str = tokenizer.decode(out[0], skip_special_tokens=True)
    alt_text = masked_text.replace(_EXTRA_TOKEN_INFILL, fill_str)
    #print(alt_text)
    return alt_text

#----------------use argparse following the original author's format---------------
#todo: delete the load ckpt logic
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="PSE_train_cv_data.csv", type=str)
    parser.add_argument("--save_dir", default="editor_ckpt", type=str)
    parser.add_argument("--num_train_epochs", default=2, type=int)
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--filling_loss_weight", default=0.1, type=float)
    parser.add_argument("--imitation_loss_weight", default=1.0, type=float)
    parser.add_argument("--adversary_loss_weight", default=0.1, type=float)
    parser.add_argument("--bnb_4bit", default=True, type=bool)
    parser.add_argument("--log_file", default="editor.log", type=str)
    parser.add_argument("--base_model", type=str)
    args = parser.parse_args()

    #let up loggers for debug check
    logging.basicConfig(
        filename=args.log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    #fix seed
    enforce_reproducibility(args.seed)

    #read and randomly mask the PSE report content
    examples = read_pse_data(args.csv_path, max_samples=200)
    examples = add_insertions_pse(examples)
    train_data = []
    for ex in examples:
        train_data += ex["masked_instances"]
    logger.info(f"Total training data: {len(train_data)}")

    #load the teacher model
    teacher_model, teacher_tokenizer = load_qwen_teacher(args.base_model, bnb_4bit=args.bnb_4bit)
    if args.bnb_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    #delete this logic, if there is a trained editor, just to to the inference file, if goes in this if, then a useless ckpt is loaded
    if os.path.exists(args.save_dir) and os.path.isfile(os.path.join(args.save_dir, "adapter_config.json")):
        '''
        from peft import PeftModel
        editor_model = PeftModel.from_pretrained(
            pretrained_model_name_or_path=args.save_dir,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info(f"Loaded existing LoRA-based T5 model from {args.save_dir}")
        '''
        print('forget to delect useless ckpt')
    else:
        #logger.info("No editor found, training from scratch")
        base_t5 = T5ForConditionalGeneration.from_pretrained(
            "t5-base",
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto"
        )
        #load t5 in LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=["q", "k", "v", "o"]
        )
        editor_model = get_peft_model(base_t5, lora_config)
        logger.info("PEFT/LoRA model created. Starting training...")
        editor_tokenizer = AutoTokenizer.from_pretrained("t5-base")
        editor_tokenizer.pad_token = editor_tokenizer.eos_token
        
        #train the editor
        train_editor(
            args,
            editor_model=editor_model,
            tokenizer=editor_tokenizer,
            train_data=train_data,
            teacher_model=teacher_model,
            teacher_tokenizer=teacher_tokenizer
        )
        editor_model.save_pretrained(args.save_dir)
        editor_tokenizer.save_pretrained(args.save_dir)
        logger.info(f"editor saved in {args.save_dir}")

    editor_tokenizer = AutoTokenizer.from_pretrained(args.save_dir)
    test_text = "The patient displayed severe chest pain."
    alt = generate_alternative_input(editor_model, editor_tokenizer, test_text, device=args.device)
    logger.info(f"Original: {test_text}")
    logger.info(f"Edited  : {alt}")

if __name__ == "__main__":
    main()
