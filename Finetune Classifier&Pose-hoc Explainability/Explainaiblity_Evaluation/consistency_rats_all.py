import argparse
import json
import os
import random
import traceback
from argparse import Namespace
from functools import partial

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from transformers import (
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
    DebertaConfig, DebertaForSequenceClassification, DebertaTokenizer,
    GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer
)
from data_loader import get_collate_fn, get_dataset


def get_model(model_path, device, tokenizer):
    """
    Load the model from a given path, applying the appropriate config based on args.model.
    """
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_args = Namespace(**checkpoint['args'])

    # Distinguish roberta, roberta-large, deberta, gpt2
    if args.model == 'roberta':
        config = RobertaConfig.from_pretrained('roberta-base', num_labels=7)
        model_cp = RobertaForSequenceClassification.from_pretrained(
            'roberta-base', config=config
        ).to(device)

    elif args.model == 'roberta_large':
        config = RobertaConfig.from_pretrained('roberta-large', num_labels=7)
        model_cp = RobertaForSequenceClassification.from_pretrained(
            'roberta-large', config=config
        ).to(device)

    elif args.model == 'deberta':
        config = DebertaConfig.from_pretrained('microsoft/deberta-base', num_labels=7)
        model_cp = DebertaForSequenceClassification.from_pretrained(
            'microsoft/deberta-base', config=config
        ).to(device)

    elif args.model == 'gpt2':
        config = GPT2Config.from_pretrained(
            'gpt2', num_labels=7, pad_token_id=tokenizer.pad_token_id
        )
        model_cp = GPT2ForSequenceClassification.from_pretrained(
            'gpt2', config=config
        ).to(device)
        model_cp.resize_token_embeddings(len(tokenizer))

    else:
        print("invalid model")
        return None, None

    model_cp.load_state_dict(checkpoint['model'])
    return model_cp, model_args


def get_saliencies(saliency_path):
    """
    Load token-level saliency values from a JSON-lines file.
    """
    max_classes = 7
    result = []
    with open(saliency_path) as out:
        for line in out:
            instance_saliency = json.loads(line)
            saliency = instance_saliency['tokens']
            token_pred_saliency = []
            for _cls in range(max_classes):
                for record in saliency:
                    token_pred_saliency.append(record[str(_cls)])
            result.append(token_pred_saliency)
    return result


def save_activation(self, inp, out):
    """
    A forward hook used to collect the activations in a global variable.
    """
    global activations
    activations.append(out)


def get_layer_activation(layer, model, instance):
    """
    Pass a single instance through the model, hooking on a specific layer to
    record its activations.
    """
    handle = None
    for name, module in model.named_modules():
        if name == layer:
            handle = module.register_forward_hook(save_activation)

    global activations
    activations = []
    with torch.no_grad():
        batch = collate_fn([instance])
        model(batch[0], attention_mask=batch[1], labels=batch[2])

    if handle:
        handle.remove()

    activ1 = None
    try:
        activations_data = activations[0]
        if isinstance(activations_data, tuple):
            activations_data = activations_data[0]
        activ1 = activations_data.detach().cpu().numpy().ravel().tolist()
    except Exception as e:
        print(e, flush=True)
        print(traceback.format_exc(), flush=True)
    return activ1


def get_model_dist(model1, model2, x, layers):
    """
    Compute the average activation difference between two models across the specified layers.
    """
    dist = []
    for layer in layers:
        act1 = get_layer_activation(layer, model1, x)
        act2 = get_layer_activation(layer, model2, x)
        if not act1 or not act2:
            continue
        dist.append(np.mean(np.array(act1).ravel() - np.array(act2).ravel()))
    return np.mean(np.abs(dist))


def get_layer_names(model, dataset):
    """
    Return the relevant layer names for each model type, including roberta-large with 24 layers.
    """
    if model == 'roberta':
        layers = [f'roberta.encoder.layer.{i}' for i in range(12)] + ['classifier']
    elif model == 'roberta_large':
        layers = [f'roberta.encoder.layer.{i}' for i in range(24)] + ['classifier']
    elif model == 'deberta':
        layers = [f'deberta.encoder.layer.{i}' for i in range(12)] + ['classifier']
    elif model == 'gpt2':
        layers = [f'transformer.h.{i}' for i in range(12)] + ['score']
    else:
        raise ValueError("invalid model")
    return layers


def get_sal_dist(sal1, sal2):
    """
    Compute the average saliency difference between two sets of saliency values.
    """
    return np.mean(np.abs(np.array(sal1).reshape(-1) - np.array(sal2).reshape(-1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir_trained", type=str)
    parser.add_argument("--model_dir_random", type=str)
    parser.add_argument("--saliency_dir_trained", type=str)
    parser.add_argument("--saliency_dir_random", type=str)
    parser.add_argument('--saliencies', type=str, nargs='+')
    # ADDED 'roberta-large' as a valid choice
    parser.add_argument('--model', type=str, choices=['roberta','roberta_large','deberta','gpt2'])
    parser.add_argument("--gpu", action='store_true', default=False)
    parser.add_argument("--dataset", choices=['PSE'])
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--per_layer", action='store_true', default=False)
    args = parser.parse_args()

    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    np.random.seed(1)

    # Configure tokenizer for the chosen model
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

    # For each specified saliency, compute the correlation across precomputed activation differences
    for saliency in args.saliencies:
        models_trained = [
            _m for _m in os.listdir(args.model_dir_trained)
            if not _m.endswith('.predictions')
        ]
        saliency_trained = [
            os.path.join(args.saliency_dir_trained, _m + f'_{saliency}')
            for _m in models_trained
        ]

        models_rand = [
            _m for _m in os.listdir(args.model_dir_random)
            if not _m.endswith('.predictions')
        ]
        saliency_rand = [
            os.path.join(args.saliency_dir_random, _m + f'_{saliency}')
            for _m in models_rand
        ]

        device = torch.device("cuda") if args.gpu else torch.device("cpu")
        test = get_dataset(mode='test', dataset=args.dataset, path=args.dataset_dir)
        coll_call = get_collate_fn(dataset=args.dataset, model=args.model)

        # Collate function with partial arguments
        global collate_fn
        collate_fn = partial(
            coll_call,
            tokenizer=tokenizer,
            device=device,
            return_attention_masks=True,
            pad_to_max_length=False,
            collate_orig=coll_call,
            n_classes=7
        )

        layers = get_layer_names(args.model, args.dataset)

        # Collect precomputed activation differences
        precomputed = []
        for _f in os.scandir('consist_rat'):
            _f = _f.name
            if args.model in _f and args.dataset in _f and _f.startswith('precomp_'):
                precomputed.append(_f)

        diff_activation, diff_saliency = [], []
        for f in precomputed:
            act_distances = json.load(open('consist_rat/' + f))
            ids = [int(_n) for _n in f.split('_') if _n.isdigit()]
            model_p = f.split('_')[3]

            if model_p == 'not':
                s1 = get_saliencies(saliency_trained[ids[0]])
                s2 = get_saliencies(saliency_trained[ids[1]])
            elif model_p == 'rand':
                s1 = get_saliencies(saliency_rand[ids[0]])
                s2 = get_saliencies(saliency_rand[ids[1]])
            else:
                s1 = get_saliencies(saliency_rand[ids[0]])
                s2 = get_saliencies(saliency_trained[ids[1]])

            for inst_id in range(len(test)):
                try:
                    sal_dist = get_sal_dist(s1[inst_id], s2[inst_id])
                    act_dist = act_distances[inst_id]
                    diff_activation.append(act_dist)
                    diff_saliency.append(sal_dist)
                except:
                    continue

        # If requested, compute Spearman correlations per layer
        if args.per_layer:
            # For each layer index, gather correlation across all instances
            for i in range(len(diff_activation[0])):
                # Activation diffs for this layer
                acts = [np.abs(_dist[i]) for _dist in diff_activation]
                diff_act = MinMaxScaler().fit_transform([[_d] for _d in acts])
                diff_act = [_d[0] for _d in diff_act]

                # Saliency diffs for the same set
                diff_sal = MinMaxScaler().fit_transform([[_d] for _d in diff_saliency])
                diff_sal = [_d[0] for _d in diff_sal]

                sp = spearmanr(diff_act, diff_sal)
                print(f'{sp[0]:.3f}', flush=True, end=' ')

        # Overall correlation across all layers
        acts = [np.abs(np.mean(_dist)) for _dist in diff_activation]
        diff_act = MinMaxScaler().fit_transform([[_d] for _d in acts])
        diff_act = [_d[0] for _d in diff_act]

        diff_sal = MinMaxScaler().fit_transform([[_d] for _d in diff_saliency])
        diff_sal = [_d[0] for _d in diff_sal]

        sp = spearmanr(diff_act, diff_sal)

        import csv
        csv_file = f"evaluation_results/rational_consistency/rational_consistency_results_{args.model}.csv"
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        write_header = not os.path.exists(csv_file)

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "saliency",
                    "spearman_correlation",
                    "p_value"
                ])
            writer.writerow([
                saliency,
                f"{sp[0]:.3f}",
                f"{sp[1]:.3e}"
            ])
