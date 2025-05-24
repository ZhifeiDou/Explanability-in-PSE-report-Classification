########################################################################################################################
#    This code is modified on top of: https://github.com/copenlu/xai-benchmark  for academic research purpose          #
########################################################################################################################

"""Script for training a Transformer model for the e-SNLI dataset with optional cross-validation and early stopping."""
#----------Initial Package Loadings---------------------------------------------------------------------------------------
import argparse
import random
from functools import partial
from typing import Dict

from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm
from transformers import AdamW, BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import get_constant_schedule_with_warmup
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config
from transformers import DebertaForSequenceClassification, DebertaTokenizer, DebertaConfig
from sklearn.model_selection import KFold

from data_loader import BucketBatchSampler, PSEDataset, collate_PSE
from model_builder import EarlyStopping


#----------Model Training Module-------------------------------------------------------------------
def train_model(model: torch.nn.Module,
                train_dl: BatchSampler, dev_dl: BatchSampler,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
                n_epochs: int,
                labels: int = 7,
                early_stopping: EarlyStopping = None) -> (Dict, Dict):
    best_val, best_model_weights = {'val_f1': 0}, None
    for ep in range(n_epochs):
        for batch in tqdm(train_dl, desc='Training'):
            model.train()
            optimizer.zero_grad()
            loss, _ = model(batch[0], attention_mask=batch[1], labels=batch[2].long())[:2]
            loss.backward()
            optimizer.step()
            scheduler.step()

        val_p, val_r, val_f1, val_loss, _, _ = eval_model(model, dev_dl, labels)
        current_val = {'val_f1': val_f1, 'val_p': val_p, 'val_r': val_r, 'val_loss': val_loss, 'ep': ep}
        print(current_val, flush=True)
        if current_val['val_f1'] > best_val['val_f1']:
            best_val = current_val
            best_model_weights = model.state_dict()

        if early_stopping and early_stopping.step(val_f1):
            print('Early stopping...')
            break
    return best_model_weights, best_val



#------------Model Evaluation Module--------------------------------------------------------------
def eval_model(model: torch.nn.Module, test_dl: BatchSampler, labels, measure=None):
    model.eval()
    with torch.no_grad():
        labels_all = []
        logits_all = []
        losses = []
        for batch in tqdm(test_dl, desc="Evaluation"):
            loss, logits_val = model(batch[0], attention_mask=batch[1], labels=batch[2].long())[:2]
            losses.append(loss.item())
            labels_all += batch[2].detach().cpu().numpy().tolist()
            logits_all += logits_val.detach().cpu().numpy().tolist()

        prediction = np.argmax(np.asarray(logits_all).reshape(-1, labels), axis=-1)
        if measure == 'acc':
            p, r = None, None
            f1 = accuracy_score(labels_all, prediction)
        else:
            p, r, f1, _ = precision_recall_fscore_support(labels_all, prediction, average='macro')
            print(confusion_matrix(labels_all, prediction), flush=True)
        return p, r, f1, np.mean(losses), labels_all, prediction.tolist()


#---------Model Accuracy Calculation Module--------------------------------------------
#todo: since the 'accuracy_score function suffering from error under python 3.12 env, investigate why
def acc_model(model: torch.nn.Module, test_dl: BatchSampler, labels):
    model.eval()
    with torch.no_grad():
        labels_all = []
        logits_all = []
        for batch in tqdm(test_dl, desc="Evaluation"):
            _, logits_val = model(batch[0], attention_mask=batch[1], labels=batch[2].long())[:2]
            labels_all += batch[2].detach().cpu().numpy().tolist()
            logits_all += logits_val.detach().cpu().numpy().tolist()

        prediction = np.argmax(np.asarray(logits_all).reshape(-1, labels), axis=-1)
        return accuracy_score(labels_all, prediction)


#--------Model Initializing Module------------------------------------------------------
def create_model(args, device):
    if args.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        collate_fn = partial(collate_PSE, tokenizer=tokenizer, device=device,
                             return_attention_masks=True, pad_to_max_length=False)
        config = RobertaConfig.from_pretrained('roberta-base', num_labels=args.labels)
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=config).to(device)
        return model, tokenizer, collate_fn

    elif args.model == 'roberta_large':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        collate_fn = partial(collate_PSE, tokenizer=tokenizer, device=device,
                             return_attention_masks=True, pad_to_max_length=False)
        config = RobertaConfig.from_pretrained('roberta-large', num_labels=args.labels)
        model = RobertaForSequenceClassification.from_pretrained('roberta-large', config=config).to(device)
        return model, tokenizer, collate_fn
    
    elif args.model == 'roberta_lora':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        collate_fn = partial(
            collate_PSE, 
            tokenizer=tokenizer, 
            device=device,
            return_attention_masks=True, 
            pad_to_max_length=False
        )
        base_config = RobertaConfig.from_pretrained('roberta-base', num_labels=args.labels)
        base_model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=base_config)

        #initialize lora config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "value"],  
            lora_dropout=0.05,
            bias="none",  
            task_type=TaskType.SEQ_CLS 
        )

        lora_model = get_peft_model(base_model, lora_config)
        
        for name, param in lora_model.named_parameters():
            if 'classifier' not in name.lower() and 'lora' not in name.lower():
                param.requires_grad = False
        
        lora_model.to(device)

        return lora_model, tokenizer, collate_fn

    elif args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.add_tokens(['[MASK]'])
        tokenizer.mask_token = '[MASK]'
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids('[MASK]')
        config = GPT2Config.from_pretrained('gpt2', num_labels=args.labels, pad_token_id=tokenizer.pad_token_id)
        model = GPT2ForSequenceClassification.from_pretrained('gpt2', config=config).to(device)
        model.resize_token_embeddings(len(tokenizer))
        collate_fn = partial(collate_PSE, tokenizer=tokenizer, device=device,
                             return_attention_masks=True, pad_to_max_length=False)
        return model, tokenizer, collate_fn

    elif args.model == 'deberta':
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        collate_fn = partial(collate_PSE, tokenizer=tokenizer, device=device,
                             return_attention_masks=True, pad_to_max_length=False)
        config = DebertaConfig.from_pretrained('microsoft/deberta-base', num_labels=args.labels)
        model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', config=config).to(device)
        return model, tokenizer, collate_fn

    elif args.model == 'deberta_large':
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-large')
        collate_fn = partial(collate_PSE, tokenizer=tokenizer, device=device,
                             return_attention_masks=True, pad_to_max_length=False)
        config = DebertaConfig.from_pretrained('microsoft/deberta-large', num_labels=args.labels)
        model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large', config=config).to(device)
        return model, tokenizer, collate_fn

    elif args.model == 'bioroberta':
        model_path = 'RoBERTa-base-PM-M3-Voc-distill-align-hf'
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        collate_fn = partial(collate_PSE, tokenizer=tokenizer, device=device,
                             return_attention_masks=True, pad_to_max_length=False)
        config = RobertaConfig.from_pretrained(model_path, num_labels=args.labels)
        model = RobertaForSequenceClassification.from_pretrained(model_path, config=config).to(device)
        return model, tokenizer, collate_fn


    else:
        print('You have entered a wrong model')
        exit()


#--------Optimizer Initialization Module----------------------------------------------
def get_optimizer(model, lr):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 1e-4
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0
        }
    ]
    return AdamW(optimizer_grouped_parameters, lr=lr)


#------Cross Validate Module (Experimental/Optional)----------------------------------
def cross_validate(args, device, n_splits=5):
    dataset = PSEDataset(args.dataset_dir, type='cv_train')
    idxs = list(range(len(dataset)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    all_f1 = []
    best_fold_perf = 0.0
    best_fold_checkpoint = None
    best_fold_idx = None

    for fold_idx, (train_index, dev_index) in enumerate(kf.split(idxs)):
        model, tokenizer, collate_fn = create_model(args, device)
        optimizer = get_optimizer(model, args.lr)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0.05)
        early_stopping = EarlyStopping(patience=6)

        train_subset = [dataset[i] for i in train_index]
        dev_subset = [dataset[i] for i in dev_index]

        sort_key = lambda x: len(x[0])
        train_dl = BucketBatchSampler(
            batch_size=args.batch_size,
            sort_key=sort_key,
            dataset=train_subset,
            collate_fn=collate_fn
        )
        dev_dl = BucketBatchSampler(
            batch_size=args.batch_size,
            sort_key=sort_key,
            dataset=dev_subset,
            collate_fn=collate_fn
        )

        best_model_w, best_perf = train_model(
            model, train_dl, dev_dl, optimizer, scheduler,
            args.epochs, labels=args.labels,
            early_stopping=early_stopping
        )
        all_f1.append(best_perf['val_f1'])
        print(f'Fold {fold_idx}, best val_f1: {best_perf["val_f1"]}')

        if best_perf['val_f1'] > best_fold_perf:
            best_fold_perf = best_perf['val_f1']
            best_fold_checkpoint = {
                'performance': best_perf,
                'args': vars(args),
                'model': best_model_w,
            }
            best_fold_idx = fold_idx

    print(f'Mean F1 across folds: {np.mean(all_f1)}, Std: {np.std(all_f1)}')
    if best_fold_checkpoint is not None:
        torch.save(best_fold_checkpoint, args.model_path[0])
        print(
            f"Best model from fold {best_fold_idx} saved to {args.model_path[0]} "
            f"with val_f1 = {best_fold_perf}"
        )


#----------Main Body-----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", type=int, default=7)
    parser.add_argument("--dataset_dir", default='data/e-SNLI/dataset/', type=str)
    parser.add_argument("--model_path", default='nli_roberta', nargs='+', type=str)
    parser.add_argument("--model", type=str, default='bert',
                        choices=[
                            'roberta','gpt2','deberta','roberta_large',
                            'bioroberta','deberta_large', 'roberta_lora'
                        ])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--mode", type=str, default='train',
                        choices=['train', 'test', 'acc', 'cv'])
    parser.add_argument("--init_only", action='store_true', default=False)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    #fix seed for reproduction
    seed = random.randint(0, 10000)
    args.seed = seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = torch.device("cuda") if args.gpu else torch.device("cpu")
    print(f'The entered model is {args.model}')

    if args.mode == 'cv':  # Cross-validation (optional)
        cross_validate(args, device, n_splits=args.folds)

    elif args.mode == 'test':
        model, tokenizer, collate_fn = create_model(args, device)
        test = PSEDataset(args.dataset_dir, type='test')
        sort_key = lambda x: len(x[0])
        test_dl = BucketBatchSampler(
            batch_size=args.batch_size,
            sort_key=sort_key,
            dataset=test,
            collate_fn=collate_fn
        )
        scores = []
        for mp in args.model_path:
            checkpoint = torch.load(mp)
            model.load_state_dict(checkpoint['model'])
            p, r, f1, loss, _, _ = eval_model(model, test_dl, args.labels)
            scores.append((p, r, f1, loss))
        for i, name in zip(range(len(scores[0])), ['p', 'r', 'f1', 'loss']):
            vals = [model_scores[i] for model_scores in scores]
            print(name, np.average(vals), np.std(vals))

    elif args.mode == 'acc':
        model, tokenizer, collate_fn = create_model(args, device)
        test = PSEDataset(args.dataset_dir, type='test')
        sort_key = lambda x: len(x[0])
        test_dl = BucketBatchSampler(
            batch_size=args.batch_size,
            sort_key=sort_key,
            dataset=test,
            collate_fn=collate_fn
        )
        for mp in args.model_path:
            checkpoint = torch.load(mp)
            model.load_state_dict(checkpoint['model'])
            print(f'The accuracy for model: {mp} is {acc_model(model, test_dl, args.labels)}')

    else:
        model, tokenizer, collate_fn = create_model(args, device)
        train_data = PSEDataset(args.dataset_dir, type='train')
        dev_data = PSEDataset(args.dataset_dir, type='dev')
        sort_key = lambda x: len(x[0])
        train_dl = BucketBatchSampler(
            batch_size=args.batch_size,
            sort_key=sort_key,
            dataset=train_data,
            collate_fn=collate_fn
        )
        dev_dl = BucketBatchSampler(
            batch_size=args.batch_size,
            sort_key=sort_key,
            dataset=dev_data,
            collate_fn=collate_fn
        )
        optimizer = get_optimizer(model, args.lr)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0.05)
        # es = EarlyStopping(patience=6) # Optionally enable early stopping

        if args.init_only:
            best_model_w, best_perf = model.state_dict(), {'val_f1': 0}
        else:
            best_model_w, best_perf = train_model(
                model, train_dl, dev_dl, optimizer, scheduler,
                args.epochs, labels=args.labels
                # , early_stopping=es
            )
        checkpoint = {
            'performance': best_perf,
            'args': vars(args),
            'model': best_model_w,
        }
        print(f'Best performance is: {best_perf}')
        print(args)
        print(args.model_path)
        torch.save(checkpoint, args.model_path[0])
