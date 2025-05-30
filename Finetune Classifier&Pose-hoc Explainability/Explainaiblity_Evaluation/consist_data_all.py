########################################################################################################################
#    This code is modified on top of: https://github.com/copenlu/xai-benchmark  for academic research purpose          #
########################################################################################################################


import argparse
import json
import os
import traceback
from functools import partial
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from transformers import (
    BertTokenizer, BertConfig, BertForSequenceClassification,
    RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification,
    DebertaTokenizer, DebertaConfig, DebertaForSequenceClassification,
    GPT2Tokenizer, GPT2Config, GPT2ForSequenceClassification
)
from data_loader import get_collate_fn, get_dataset
import csv
import time

# Retrieve the layer names from the model--------------------------------------------
def get_layer_names(model_type, dataset):
    if model_type == 'roberta':
        layers = [f'roberta.encoder.layer.{i}' for i in range(12)] + ['classifier']
    elif model_type == 'roberta_large':
        layers = [f'roberta.encoder.layer.{i}' for i in range(24)] + ['classifier']
    elif model_type == 'deberta':
        layers = [f'deberta.encoder.layer.{i}' for i in range(12)] + ['classifier']
    elif model_type == 'gpt2':
        layers = [f'transformer.h.{i}' for i in range(12)] + ['score']
    else:
        layers = [f'bert.encoder.layer.{i}' for i in range(12)] + ['classifier']
    return layers

#Loading models from local----------------------------------------------------------
def get_model(model_path, device, model_type, tokenizer, num_labels=2):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    if 'args' in checkpoint:
        if hasattr(checkpoint['args'], 'num_labels'):
            num_labels = checkpoint['args'].num_labels

    if model_type == 'roberta':
        base_name = 'roberta-base'
        config = RobertaConfig.from_pretrained(base_name, num_labels=num_labels)
        model_cp = RobertaForSequenceClassification.from_pretrained(base_name, config=config).to(device)
    elif model_type == 'roberta_large':
        base_name = 'roberta-large'
        config = RobertaConfig.from_pretrained(base_name, num_labels=num_labels)
        model_cp = RobertaForSequenceClassification.from_pretrained(base_name, config=config).to(device)
    elif model_type == 'deberta':
        base_name = 'microsoft/deberta-base'
        config = DebertaConfig.from_pretrained(base_name, num_labels=num_labels)
        model_cp = DebertaForSequenceClassification.from_pretrained(base_name, config=config).to(device)
    elif model_type == 'gpt2':
        base_name = 'gpt2'
        config = GPT2Config.from_pretrained(base_name, num_labels=num_labels, pad_token_id=tokenizer.pad_token_id)
        model_cp = GPT2ForSequenceClassification.from_pretrained(base_name, config=config).to(device)
        model_cp.resize_token_embeddings(len(tokenizer))
    else:
        base_name = 'bert-base-uncased'
        config = BertConfig.from_pretrained(base_name, num_labels=num_labels)
        model_cp = BertForSequenceClassification.from_pretrained(base_name, config=config).to(device)

    model_cp.load_state_dict(checkpoint['model'], strict=False)
    return model_cp, checkpoint.get('args', None)

# Load Saliency from Local------------------------------------------------------------
def get_saliencies(saliency_path, test, tokenizer, dataset, n_labels=2):
    result = []
    tokens = []
    with open(saliency_path, 'r') as out:
        for i, line in enumerate(out):
            if i >= len(test):
                break
            instance_saliency = json.loads(line)
            saliency = instance_saliency['tokens']
            instance = test[i]

            if dataset == 'snli':
                token_ids = tokenizer.encode(instance[0], instance[1])
            else:
                token_ids = tokenizer.encode(instance[0])

            token_pred_saliency = []
            for cls_id in range(n_labels):
                for record in saliency:
                    token_pred_saliency.append(record[str(cls_id)])

            result.append(token_pred_saliency)
            tokens.append(token_ids)

    return result, tokens


def save_activation(self, inp, out):
    global activations
    activations.append(out)


def get_layer_activation(layer, model, instance, collate_fn):
    handle = None
    for name, module in model.named_modules():
        if name == layer:
            handle = module.register_forward_hook(save_activation)
            break

    global activations
    activations = []
    with torch.no_grad():
        batch = collate_fn([instance])
        if len(batch) == 3:
            model(batch[0], attention_mask=batch[1], labels=batch[2])
        elif len(batch) == 2:
            model(batch[0], attention_mask=batch[1])
        else:
            model(batch[0])

    if handle is not None:
        handle.remove()

    activ1 = None
    try:
        captured = activations[0]
        if isinstance(captured, tuple) and len(captured) == 1:
            captured = captured[0]
        if args.model in ['roberta','bioroberta','gpt2', 'deberta', 'roberta_large']: #somehow the last condition does not wrok on all tested models thus adding this
            captured = captured[0]
        activ1 = captured.detach().cpu().numpy().ravel().tolist()
    except Exception as e:
        print(e, flush=True)
        print(traceback.format_exc(), flush=True)
        time.sleep(200)
    return activ1


def get_model_distv2(model, x, y, collate_fn, model_type, dataset):
    dist = []
    layer_names = get_layer_names(model_type, dataset)
    for layer in layer_names:
        act1 = get_layer_activation(layer, model, x, collate_fn)
        act2 = get_layer_activation(layer, model, y, collate_fn)
        if act1 is None or act2 is None:
            continue
        dist.append(np.mean(np.array(act1) - np.array(act2)))
    return dist

# Retrieve the model's embedding size----------------------------------------
def get_model_embedding_emb_size(model):
    if hasattr(model, 'bert'):
        return model.bert.embeddings.word_embeddings.weight.shape[0]
    if hasattr(model, 'roberta'):
        return model.roberta.embeddings.word_embeddings.weight.shape[0]
    if hasattr(model, 'deberta'):
        return model.deberta.embeddings.word_embeddings.weight.shape[0]
    if hasattr(model, 'transformer'):
        return model.transformer.wte.weight.shape[0]
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir_trained", help="Directory with trained models", type=str)
    parser.add_argument("--saliency_dir_trained", help="Directory with saliencies for trained models", type=str)
    parser.add_argument('--saliencies', help='Names of the saliencies to be considered', type=str, nargs='+')
    parser.add_argument('--model', help='Which model: roberta, roberta_large, deberta, gpt2, or fallback to BERT', type=str, default='bert')
    parser.add_argument("--dataset_dir", help="Path to the directory with the datasets", default='data/e-SNLI/dataset/', type=str)
    parser.add_argument("--gpu", help="Run on GPU if set", action='store_true', default=False)
    parser.add_argument("--dataset", help='Which dataset: snli, imdb, tweet, PSE', choices=['snli', 'imdb', 'tweet','PSE'], default='imdb')
    args = parser.parse_args()

    np.random.seed(1)

    if args.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif args.model == 'roberta_large':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
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
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if args.dataset == 'imdb':
        n_labels = 2
    elif args.dataset == 'snli':
        n_labels = 3
    else:
        n_labels = 7

    test = get_dataset(args.dataset_dir, args.dataset)
    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    coll_call = get_collate_fn(args.dataset, args.model)
    collate_fn = partial(
        coll_call,
        tokenizer=tokenizer,
        device=device,
        return_attention_masks=True,
        pad_to_max_length=True
    )

    models_trained = [
        _m for _m in os.listdir(args.model_dir_trained)
        if not _m.endswith('.predictions')
    ]
    full_model_paths_trained = [
        os.path.join(args.model_dir_trained, _m)
        for _m in models_trained
    ]
    
    #loop through all saliencies and calculate-----------------------------
    for saliency_name in args.saliencies:
        all_scores = []
        for i, model_path in enumerate(full_model_paths_trained):
            model_cp, model_args = get_model(
                model_path, device, args.model, tokenizer, num_labels=n_labels
            )
            model_size = get_model_embedding_emb_size(model_cp) or 30522

            saliency_path = os.path.join(
                args.saliency_dir_trained,
                models_trained[i] + f'_{saliency_name}'
            )
            saliencies, token_ids = get_saliencies(
                saliency_path, test, tokenizer, args.dataset, n_labels
            )

            dataset_ids = []
            pair_file = f'selected_pairs_{args.dataset}.tsv'
            if os.path.exists(pair_file):
                with open(pair_file) as f:
                    for line in f:
                        idx1, idx2 = line.strip().split()
                        dataset_ids.append((int(idx1), int(idx2)))
            else:
                dataset_ids = [(i, i + 1) for i in range(0, len(test) - 1, 2)]

            dist_dir = f'consist_data/{args.dataset}_{os.path.basename(model_path)}.json'
            if not os.path.exists(dist_dir):
                diff_activation = []
                for (ind1, ind2) in tqdm(dataset_ids, desc='Calculating Model Differences'):
                    dist_vals = get_model_distv2(
                        model_cp, test[ind1], test[ind2],
                        collate_fn, args.model, args.dataset
                    )
                    diff_activation.append(dist_vals)
                with open(dist_dir, 'w') as out_file:
                    json.dump(diff_activation, out_file)
            else:
                diff_activation = json.load(open(dist_dir))

            diff_saliency = []
            for (ind1, ind2) in tqdm(dataset_ids, desc='Calculating Saliency Differences'):
                if ind1 >= len(saliencies) or ind2 >= len(saliencies):
                    continue

                size_needed = max(
                    max(token_ids[ind1]) if token_ids[ind1] else 0,
                    max(token_ids[ind2]) if token_ids[ind2] else 0
                ) + 1

                pair_word_mask = [0.0] * size_needed
                mult1 = [1e-7] * size_needed
                mult2 = [1e-7] * size_needed

                for t_id, sal_val in zip(token_ids[ind1], saliencies[ind1]):
                    if t_id < size_needed:
                        mult1[t_id] = sal_val
                        pair_word_mask[t_id] = 1.0

                for t_id, sal_val in zip(token_ids[ind2], saliencies[ind2]):
                    if t_id < size_needed:
                        mult2[t_id] = sal_val
                        pair_word_mask[t_id] = 1.0

                mult1 = np.array([v for i, v in enumerate(mult1) if pair_word_mask[i] != 0])
                mult2 = np.array([v for i, v in enumerate(mult2) if pair_word_mask[i] != 0])

                sal_dist = np.mean(np.abs(mult1 - mult2))
                diff_saliency.append(sal_dist)

            diff_activation_scaled = MinMaxScaler().fit_transform(
                [[np.mean(np.abs(_d))] for _d in diff_activation]
            )
            diff_activation_scaled = [_d[0] for _d in diff_activation_scaled]

            diff_saliency_scaled = MinMaxScaler().fit_transform(
                [[_d] for _d in diff_saliency]
            )
            diff_saliency_scaled = [_d[0] for _d in diff_saliency_scaled]

            diff_activation_scaled = np.nan_to_num(diff_activation_scaled)
            diff_saliency_scaled = np.nan_to_num(diff_saliency_scaled)

            if len(diff_activation_scaled) != len(diff_saliency_scaled):
                continue

            sr = spearmanr(diff_activation_scaled, diff_saliency_scaled)
            all_scores.append([sr.correlation, sr.pvalue])

        if all_scores:
            mean_corr = np.mean([score[0] for score in all_scores])
            mean_pval = np.mean([score[1] for score in all_scores])
            print(
                f"\nOverall {saliency_name} Spearmanr: {mean_corr:.3f} "
                f"({mean_pval:.1e})\n",
                flush=True
            )

            csv_file = f"evaluation_results/data_consistency/data_consistency_results_{args.model}.csv"
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            write_header = not os.path.exists(csv_file)

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["saliency", "spearman_correlation", "p_value"])
                writer.writerow([
                    saliency_name,
                    f"{mean_corr:.3f}",
                    f"{mean_pval:.1f}"
                ])
        else:
            print(f"No scores computed for {saliency_name}.")
