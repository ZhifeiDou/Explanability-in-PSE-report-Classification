########################################################################################################################
#    This code is modified on top of: https://github.com/copenlu/xai-benchmark  for academic research purpose          #
########################################################################################################################


# Loading Libs----------------------------------------------------------------------------------
import csv
import argparse
import os
import random
from argparse import Namespace
from functools import partial

import numpy as np
import torch
from sklearn.metrics import auc
from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    GPT2Config,
    DebertaForSequenceClassification,
    DebertaTokenizer,
    DebertaConfig,
)

import train_lstm_cnn
import train_transformers
import train_PSE_all
from data_loader import (
    BucketBatchSampler,
    DatasetSaliency,
    collate_threshold,
    get_collate_fn,
    get_dataset
)
from model_builder import CNN_MODEL, LSTM_MODEL


def get_model(model_path):
    return model_cp, model_args


# Main body-----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for running on gpu",
                        action='store_true', default=False)
    parser.add_argument("--saliency", help='Name of the saliency', type=str)
    parser.add_argument("--dataset", choices=['snli', 'imdb', 'tweet','PSE'])
    parser.add_argument("--dataset_dir",
                        help="Path to the direcory with the datasets",
                        default='data/e-SNLI/dataset/',
                        type=str)
    parser.add_argument("--test_saliency_dir",
                        help="Path to the saliency files", type=str)
    parser.add_argument("--model_path", help="Directory with all of the models",
                        type=str, nargs='+')
    parser.add_argument("--models_dir", help="Directory with all of the models",
                        type=str)
    parser.add_argument("--model", help="Type of model",
                        choices=['roberta', 'deberta', 'gpt2','roberta_large'])

    args = parser.parse_args()
    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    thresholds = list(range(0, 110, 10))
    aucs = []

    coll_call = get_collate_fn(dataset=args.dataset, model=args.model)
    return_attention_masks = True
    eval_fn = train_PSE_all.eval_model

    #load prediction logits from local
    for model_path in os.listdir(args.models_dir):
        if model_path.endswith('.predictions'):
            continue
        print('Model', model_path, flush=True)
        model_full_path = os.path.join(args.models_dir, model_path)
        checkpoint = torch.load(
            model_full_path,
            map_location=lambda storage, loc: storage
        )
        model_args = Namespace(**checkpoint['args'])

        #major modification: load model options, GPT2 needs more accommodateion
        if args.model == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            transformer_config = RobertaConfig.from_pretrained(
                'roberta-base',
                num_labels=model_args.labels
            )
            model_cp = RobertaForSequenceClassification.from_pretrained(
                'roberta-base',
                config=transformer_config
            ).to(device)

        elif args.model == 'roberta_large':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
            transformer_config = RobertaConfig.from_pretrained(
                'roberta-large',
                num_labels=model_args.labels
            )
            model_cp = RobertaForSequenceClassification.from_pretrained(
                'roberta-large',
                config=transformer_config
            ).to(device)

        elif args.model == 'deberta':
            tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
            transformer_config = DebertaConfig.from_pretrained(
                'microsoft/deberta-base',
                num_labels=model_args.labels
            )
            model_cp = DebertaForSequenceClassification.from_pretrained(
                'microsoft/deberta-base',
                config=transformer_config
            ).to(device)

        elif args.model == 'gpt2':
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.add_tokens(['[MASK]'])
            tokenizer.mask_token = '[MASK]'
            tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids('[MASK]') 
            #same as before, since GPT2 as a docoder only model, does not have a mask token, we would like to create a new one
            transformer_config = GPT2Config.from_pretrained(
                'gpt2',
                num_labels=model_args.labels,
                pad_token_id=tokenizer.pad_token_id
            )
            model_cp = GPT2ForSequenceClassification.from_pretrained(
                'gpt2',
                config=transformer_config
            ).to(device)
            model_cp.resize_token_embeddings(len(tokenizer))

        else:
            print('You have entered the wrong model type.')
            continue

        model_cp.load_state_dict(checkpoint['model'])

        #fix seed for reproduction
        random.seed(model_args.seed)
        torch.manual_seed(model_args.seed)
        torch.cuda.manual_seed_all(model_args.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(model_args.seed)

        model_scores = []
        for threshold in thresholds:
            collate_fn = partial(
                collate_threshold,
                tokenizer=tokenizer,
                device=device,
                return_attention_masks=return_attention_masks,
                pad_to_max_length=False,
                threshold=threshold,
                collate_orig=coll_call,
                n_classes=3 if args.dataset in ['snli', 'tweet'] else 7
            )
            #form the saliency path for loading
            saliency_path_test = os.path.join(
                args.test_saliency_dir,
                f'{model_path}_{args.saliency}'
            )
            test = get_dataset(
                mode='test',
                dataset=args.dataset,
                path=args.dataset_dir
            )
            test = DatasetSaliency(test, saliency_path_test)
            test_dl = BucketBatchSampler(
                batch_size=model_args.batch_size,
                dataset=test,
                collate_fn=collate_fn
            )
            #evaluate the result again for measure the dropping
            results = eval_fn(model_cp, test_dl, model_args.labels)
            model_scores.append(results[2])

        print(thresholds, model_scores)
        aucs.append(auc(thresholds, model_scores))

    csv_file = f"evaluation_results/faithfulness/faithfulness_results_{args.model}.csv"
    write_header = not os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "saliency", "mean_value", "std_value"])
        writer.writerow([
            args.model,
            args.saliency,
            f"{np.mean(aucs):.2f}",
            f"{np.std(aucs):.2f}"
        ])
