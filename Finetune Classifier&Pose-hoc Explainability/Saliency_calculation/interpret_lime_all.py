########################################################################################################################
#    This code is modified on top of: https://github.com/copenlu/xai-benchmark  for academic research purpose          #
########################################################################################################################

#--------Load Packages----------------------------------------------------------------------------------------------s
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm

from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    DebertaConfig,
    DebertaForSequenceClassification,
    DebertaTokenizer,
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
)

from data_loader import get_dataset
from model_builder import CNN_MODEL, LSTM_MODEL
from peft import LoraConfig, get_peft_model, TaskType


#-----Model Wrappers-------------------------------------------------------------------------
class RobertaModelWrapper(nn.Module):
    def __init__(self, model, device, tokenizer, args):
        super(RobertaModelWrapper, self).__init__()
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, token_ids):
        results = []
        token_ids = [
            [int(i) for i in instance_ids.split(' ') if i != '']
            for instance_ids in token_ids
        ]
        for i in tqdm(
            range(0, len(token_ids), self.args.batch_size),
            desc='Building a local approximation...'
        ):
            batch_ids = token_ids[i:i + self.args.batch_size]
            max_batch_id = min(max(len(_l) for _l in batch_ids), 512)
            batch_ids = [_l[:max_batch_id] for _l in batch_ids]
            padded_batch_ids = [
                _l + [self.tokenizer.pad_token_id] * (max_batch_id - len(_l))
                for _l in batch_ids
            ]
            tokens_tensor = torch.tensor(padded_batch_ids).to(self.device)
            logits = self.model(
                tokens_tensor.long(),
                attention_mask=(tokens_tensor.long() > 0)
            )
            results += logits[0].detach().cpu().numpy().tolist()
        return np.array(results)


class DebertaModelWrapper(nn.Module):
    def __init__(self, model, device, tokenizer, args):
        super(DebertaModelWrapper, self).__init__()
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, token_ids):
        results = []
        token_ids = [
            [int(i) for i in instance_ids.split(' ') if i != '']
            for instance_ids in token_ids
        ]
        for i in tqdm(
            range(0, len(token_ids), self.args.batch_size),
            desc='Building a local approximation...'
        ):
            batch_ids = token_ids[i:i + self.args.batch_size]
            max_batch_id = min(max(len(_l) for _l in batch_ids), 512)
            batch_ids = [_l[:max_batch_id] for _l in batch_ids]
            padded_batch_ids = [
                _l + [self.tokenizer.pad_token_id] * (max_batch_id - len(_l))
                for _l in batch_ids
            ]
            tokens_tensor = torch.tensor(padded_batch_ids).to(self.device)
            logits = self.model(
                tokens_tensor.long(),
                attention_mask=(tokens_tensor.long() != self.tokenizer.pad_token_id)
            )
            results += logits[0].detach().cpu().numpy().tolist()
        return np.array(results)


class GPT2ModelWrapper(nn.Module):
    def __init__(self, model, device, tokenizer, args):
        super(GPT2ModelWrapper, self).__init__()
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, token_ids):
        results = []
        token_ids = [
            [int(i) for i in instance_ids.split(' ') if i != '']
            for instance_ids in token_ids
        ]
        for i in tqdm(
            range(0, len(token_ids), self.args.batch_size),
            desc='Building a local approximation...'
        ):
            batch_ids = token_ids[i:i + self.args.batch_size]
            max_batch_id = min(max(len(_l) for _l in batch_ids), 512)
            batch_ids = [_l[:max_batch_id] for _l in batch_ids]
            padded_batch_ids = [
                _l + [self.tokenizer.pad_token_id] * (max_batch_id - len(_l))
                for _l in batch_ids
            ]
            tokens_tensor = torch.tensor(padded_batch_ids).to(self.device)
            logits = self.model(
                tokens_tensor.long(),
                attention_mask=(tokens_tensor.long() != self.tokenizer.pad_token_id)
            )
            results += logits[0].detach().cpu().numpy().tolist()
        return np.array(results)


#--------Saliency Generation---------------------------------------------------
def generate_saliency(model_path, saliency_path):
    test = get_dataset(path=args.dataset_dir, mode=args.split, dataset=args.dataset)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**checkpoint['args'])

    #load model
    if args.model == 'roberta':
        model_args.batch_size = 7
        base_model_name = 'roberta-base'
        config = RobertaConfig.from_pretrained(base_model_name, num_labels=model_args.labels)
        model = RobertaForSequenceClassification.from_pretrained(base_model_name, config=config).to(device)
        model.load_state_dict(checkpoint['model'])
        modelw = RobertaModelWrapper(model, device, tokenizer, model_args)
    
    elif args.model == 'roberta_lora':
        model_args.batch_size = 7
        config = RobertaConfig.from_pretrained(
            'roberta-base',
            num_labels=model_args.labels
        )
        # Load base roberta
        base_model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            config=config
        )

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
        lora_model = get_peft_model(base_model, lora_config)
        lora_model.load_state_dict(checkpoint['model'])
        lora_model = lora_model.to(device)
        modelw = RobertaModelWrapper(lora_model, device, tokenizer, model_args)

    elif args.model == 'bioroberta':
        model_args.batch_size = 7
        model_name = 'RoBERTa-base-PM-M3-Voc-distill-align-hf'
        config = RobertaConfig.from_pretrained(model_name, num_labels=model_args.labels)
        model = RobertaForSequenceClassification.from_pretrained(model_name, config=config).to(device)
        model.load_state_dict(checkpoint['model'])
        modelw = RobertaModelWrapper(model, device, tokenizer, model_args)


    elif args.model == 'roberta_large':
        model_args.batch_size = 7
        base_model_name = 'roberta-large'
        config = RobertaConfig.from_pretrained(base_model_name, num_labels=model_args.labels)
        model = RobertaForSequenceClassification.from_pretrained(base_model_name, config=config).to(device)
        model.load_state_dict(checkpoint['model'])
        modelw = RobertaModelWrapper(model, device, tokenizer, model_args)

    elif args.model == 'deberta':
        model_args.batch_size = 7
        base_model_name = 'microsoft/deberta-base'
        config = DebertaConfig.from_pretrained(base_model_name, num_labels=model_args.labels)
        model = DebertaForSequenceClassification.from_pretrained(base_model_name, config=config).to(device)
        model.load_state_dict(checkpoint['model'])
        modelw = DebertaModelWrapper(model, device, tokenizer, model_args)
    
    elif args.model == 'gpt2':
        model_args.batch_size = 7
        base_model_name = 'gpt2'
        config = GPT2Config.from_pretrained(
            base_model_name,
            num_labels=model_args.labels,
            pad_token_id=tokenizer.pad_token_id
        )
        model = GPT2ForSequenceClassification.from_pretrained(base_model_name, config=config).to(device)
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(checkpoint['model'])
        modelw = GPT2ModelWrapper(model, device, tokenizer, model_args)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    modelw.eval()
    explainer = LimeTextExplainer()
    saliency_flops = []
    
    #Generate Saliency
    with open(saliency_path, 'w') as out:
        for instance in tqdm(test, desc="Generating LIME explanations"):
            saliencies = []
            if args.dataset in ['imdb', 'tweet', 'PSE']:
                text_str = instance[0]
                token_ids = tokenizer.encode(text_str)
            elif args.dataset == 'snli':
                text_str = instance[0] + " " + instance[1]
                token_ids = tokenizer.encode(text_str)
            else:
                text_str = instance[0]
                token_ids = tokenizer.encode(text_str)

            if len(token_ids) < 6:
                token_ids += [tokenizer.pad_token_id] * (6 - len(token_ids))

            try:
                exp = explainer.explain_instance(
                    " ".join(str(i) for i in token_ids),
                    modelw,
                    num_features=len(token_ids),
                    top_labels=args.labels
                )
            except Exception as e:
                print("LIME encountered an exception:", e)
                for token_id in token_ids:
                    token_id = int(token_id)
                    token_saliency = {'token': tokenizer.convert_ids_to_tokens(token_id)}
                    for cls_ in range(args.labels):
                        token_saliency[int(cls_)] = 0
                    saliencies.append(token_saliency)
                out.write(json.dumps({'tokens': saliencies}) + '\n')
                out.flush()
                continue

            explanation = {}
            for cls_ in range(args.labels):
                cls_expl = {}
                for (w, s) in exp.as_list(label=cls_):
                    cls_expl[int(w)] = s
                explanation[cls_] = cls_expl

            for token_id in token_ids:
                token_id = int(token_id)
                token_saliency = {'token': tokenizer.convert_ids_to_tokens(token_id)}
                for cls_ in range(args.labels):
                    token_saliency[int(cls_)] = explanation[cls_].get(token_id, 0.0)
                saliencies.append(token_saliency)

            out.write(json.dumps({'tokens': saliencies}) + '\n')
            out.flush()

    return saliency_flops

#-----------Main body-----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_time", action='store_true', default=False)
    parser.add_argument("--dataset", help="Which dataset", default='snli', type=str,
                        choices=['snli', 'imdb', 'tweet', 'PSE'])
    parser.add_argument("--dataset_dir", help="Path to the directory with the datasets",
                        default='data/e-SNLI/dataset/', type=str)
    parser.add_argument("--model", help="Which model architecture to load",
                        default='roberta',
                        choices=['roberta', 'roberta_large', 'deberta', 'gpt2','bioroberta','roberta_lora'],
                        type=str)
    parser.add_argument("--model_path", help="Path to the model checkpoint (.pt or .bin)",
                        default='snli_roberta', type=str)
    parser.add_argument("--output_dir", help="Path where the saliency will be serialized",
                        default='', type=str)
    parser.add_argument("--gpu", action='store_true', default=False)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--labels", type=int, default=3)
    parser.add_argument("--split", default='test', type=str,
                        choices=['train', 'test', 'val'])
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu_id}") if args.gpu else torch.device("cpu")

    if args.model == 'roberta' or args.model == 'roberta_lora':
        base_model_name = 'roberta-base' 
        tokenizer = RobertaTokenizer.from_pretrained(base_model_name)
    
    elif args.model == 'bioroberta':
        base_model_name = 'RoBERTa-base-PM-M3-Voc-distill-align-hf' 
        tokenizer = RobertaTokenizer.from_pretrained(base_model_name)

    elif args.model == 'roberta_large':
        base_model_name = 'roberta-large' 
        tokenizer = RobertaTokenizer.from_pretrained(base_model_name)
    elif args.model == 'deberta':
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    elif args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.add_tokens(['[MASK]'])
        tokenizer.mask_token = '[MASK]'
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids('[MASK]')
    else:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    model_path = args.model_path
    all_flops = generate_saliency(
        model_path,
        os.path.join(args.output_dir, f'{os.path.basename(model_path)}_lime')
    )
    print(
        'FLOPS',
        np.average(all_flops) if all_flops else 0.0,
        np.std(all_flops) if all_flops else 0.0
    )
