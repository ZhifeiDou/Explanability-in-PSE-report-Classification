########################################################################################################################
#    This code is modified on top of: https://github.com/copenlu/xai-benchmark  for academic research purpose          #
########################################################################################################################


#--------Package loading-----------------------------------------------
import argparse
import json
import os
import random
from argparse import Namespace
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from captum.attr import (
    DeepLift,
    GuidedBackprop,
    InputXGradient,
    Occlusion,
    Saliency,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    RobertaForSequenceClassification,
    AutoTokenizer,
    RobertaConfig,
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    GPT2Config,
    DebertaForSequenceClassification,
    DebertaTokenizer,
    DebertaConfig,
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    RobertaTokenizer
)

from peft import LoraConfig, get_peft_model, TaskType

from data_loader import get_collate_fn, get_dataset
from model_builder import CNN_MODEL, LSTM_MODEL
import time


#---------Aggregation Method Module---------------------------------------------------
def summarize_attributions(attributions, type='mean', model=None, tokens=None):
    if type == 'none':
        return attributions
    elif type == 'dot':
        embeddings = get_model_embedding_emb(model)(tokens)
        attributions = torch.einsum('bwd, bwd->bw', attributions, embeddings)
    elif type == 'mean':
        attributions = attributions.mean(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
    elif type == 'l2':
        attributions = attributions.norm(p=1, dim=-1).squeeze(0)
    return attributions



#--------Following Modules are Model Wrappers---------------------------------------
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, input, attention_mask, labels):
        try:
            return self.model(input, attention_mask=attention_mask)[0]
        except:
            print('error, the input shape is:', input.size())
            time.sleep(200)


class GPT2ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPT2ModelWrapper, self).__init__()
        self.model = model

    def forward(self, input, attention_mask, labels):
        if input.dim() == 2 and input.dtype == torch.long:
            return self.model(input_ids=input, attention_mask=attention_mask)[0]
        elif input.dim() == 3 and input.dtype in (torch.float, torch.float32, torch.float16):
            return self.model(inputs_embeds=input, attention_mask=attention_mask)[0]
        else:
            raise ValueError(
                f"Unrecognized input shape {input.shape} or dtype {input.dtype}. "
                "Expected (batch, seq) for token IDs or (batch, seq, emb) for embeddings."
            )


class RobertaModelWrapper(torch.nn.Module):
    def __init__(self, roberta_model):
        super(RobertaModelWrapper, self).__init__()
        self.roberta_model = roberta_model

    def forward(self, input, attention_mask=None, labels=None):
        if input.dim() == 2 and input.dtype == torch.long:
            outputs = self.roberta_model(
                input_ids=input,
                attention_mask=attention_mask,
                labels=labels
            )
            return outputs.logits
        elif input.dim() == 3 and input.dtype in (torch.float, torch.float32, torch.float16):
            outputs = self.roberta_model(
                inputs_embeds=input,
                attention_mask=attention_mask,
                labels=labels
            )
            return outputs.logits
        else:
            raise ValueError(
                f"Unrecognized input shape {input.shape} or dtype {input.dtype}. "
                "Expected (batch_size, seq_len) for token IDs "
                "or (batch_size, seq_len, hidden_dim) for embeddings."
            )


#-----------Embedding Extraction Module-----------------------------
def get_model_embedding_emb(model):

    if args.model == 'deberta':
        return model.deberta.embeddings.word_embeddings
    elif args.model == 'gpt2':
        return model.transformer.wte
    elif args.model in ['roberta', 'roberta_large', 'bioroberta', 'roberta_lora']:
        return model.roberta_model.roberta.embeddings.word_embeddings
    else:
        raise ValueError("Unsupported model type or incorrect model name.")


#-----------Gradient Based Saliency Production---------------------
def generate_saliency(model_path, saliency_path, saliency, aggregation):
    print('load from', model_path)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_args = Namespace(**checkpoint['args'])

    if args.model == 'deberta':
        transformer_config = DebertaConfig.from_pretrained(
            'microsoft/deberta-base',
            num_labels=model_args.labels
        )
        model_cp = DebertaForSequenceClassification.from_pretrained(
            'microsoft/deberta-base', config=transformer_config
        ).to(device)
        model_cp.load_state_dict(checkpoint['model'])
        model = ModelWrapper(model_cp)

    elif args.model == 'gpt2':
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
        model_cp.load_state_dict(checkpoint['model'])
        model = GPT2ModelWrapper(model_cp)

    elif args.model == 'roberta_lora':
        transformer_config = RobertaConfig.from_pretrained(
            'roberta-base',
            num_labels=model_args.labels
        )
        # Load base roberta
        base_model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            config=transformer_config
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
        model = RobertaModelWrapper(lora_model)

    elif args.model == 'roberta':
        transformer_config = RobertaConfig.from_pretrained(
            'roberta-base',
            num_labels=model_args.labels
        )
        model_cp = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            config=transformer_config
        ).to(device)
        model_cp.load_state_dict(checkpoint['model'])
        model = RobertaModelWrapper(model_cp)

    elif args.model == 'roberta_large':
        transformer_config = RobertaConfig.from_pretrained(
            'roberta-large',
            num_labels=model_args.labels
        )
        model_cp = RobertaForSequenceClassification.from_pretrained(
            'roberta-large',
            config=transformer_config
        ).to(device)
        model_cp.load_state_dict(checkpoint['model'])
        model = RobertaModelWrapper(model_cp)

    elif args.model == 'bioroberta':
        transformer_config = RobertaConfig.from_pretrained(
            "RoBERTa-base-PM-M3-Voc-distill-align-hf",
            num_labels=model_args.labels
        )
        model_cp = RobertaForSequenceClassification.from_pretrained(
            "RoBERTa-base-PM-M3-Voc-distill-align-hf",
            config=transformer_config
        ).to(device)
        model_cp.load_state_dict(checkpoint['model'])
        model = RobertaModelWrapper(model_cp)

    else:
        print("Wrong type of code entered")
        return

    model.train()

    # Choose Captum method
    if saliency == 'deeplift':
        ablator = DeepLift(model)
    elif saliency == 'guided':
        ablator = GuidedBackprop(model)
    elif saliency == 'sal':
        ablator = Saliency(model)
    elif saliency == 'inputx':
        ablator = InputXGradient(model)
    elif saliency == 'occlusion':
        ablator = Occlusion(model)
    else:
        raise ValueError(f"Unknown saliency method: {saliency}")

    #preprosess the model and dataset
    coll_call = get_collate_fn(dataset=args.dataset, model=args.model)
    collate_fn = partial(
        coll_call,
        tokenizer=tokenizer,
        device=device,
        return_attention_masks=True,
        pad_to_max_length=False
    )
    test = get_dataset(path=args.dataset_dir, mode=args.split, dataset=args.dataset)
    batch_size = args.batch_size if args.batch_size is not None else model_args.batch_size
    test_dl = DataLoader(
        batch_size=batch_size,
        dataset=test,
        shuffle=False,
        collate_fn=collate_fn
    )

    #load model predictions and embeddings for saliency calculation
    predictions_path = model_path + '.predictions'
    if not os.path.exists(predictions_path):
        predictions = defaultdict(lambda: [])
        for batch in tqdm(test_dl, desc='Running test prediction... '):
            logits = model(batch[0], attention_mask=batch[1], labels=batch[2].long())
            logits = logits.detach().cpu().numpy().tolist()
            predicted = np.argmax(np.array(logits), axis=-1)
            predictions['class'] += predicted.tolist()
            predictions['logits'] += logits
        with open(predictions_path, 'w') as out:
            json.dump(predictions, out)

    if saliency != 'occlusion':
        if args.model == 'deberta':
            embedding_layer_name = 'model.deberta.embeddings'
        elif args.model in ['roberta', 'roberta_large', 'bioroberta', 'roberta_lora']:
            embedding_layer_name = 'roberta_model.roberta.embeddings.word_embeddings'
        elif args.model == 'gpt2':
            embedding_layer_name = 'model.transformer.wte'
        else:
            raise ValueError('Wrong model entered for interpretable embedding.')

        interpretable_embedding = configure_interpretable_embedding_layer(model, embedding_layer_name)

    class_attr_list = defaultdict(lambda: [])
    token_ids = []
    saliency_flops = []

    #calculate the saliency
    for batch in tqdm(test_dl, desc='Running Saliency Generation...'):
        additional = (batch[1], batch[2])
        token_ids += batch[0].detach().cpu().numpy().tolist()

        if saliency != 'occlusion':
            input_embeddings = interpretable_embedding.indices_to_embeddings(batch[0])

        for cls_ in range(checkpoint['args']['labels']):
            if saliency == 'occlusion':
                attributions = ablator.attribute(
                    batch[0],
                    sliding_window_shapes=(args.sw,),
                    target=cls_,
                    additional_forward_args=additional
                )
            else:
                attributions = ablator.attribute(
                    input_embeddings,
                    target=cls_,
                    additional_forward_args=additional
                )
            attributions = summarize_attributions(
                attributions,
                type=aggregation,
                model=model,
                tokens=batch[0]
            ).detach().cpu().numpy().tolist()

            class_attr_list[cls_] += [[_li for _li in _l] for _l in attributions]

    if saliency != 'occlusion':
        remove_interpretable_embedding_layer(model, interpretable_embedding)

    with open(saliency_path, 'w') as out:
        for instance_i, _ in enumerate(test):
            saliencies = []
            for token_i, token_id in enumerate(token_ids[instance_i]):
                token_sal = {'token': tokenizer.convert_ids_to_tokens(token_id)}
                for cls_ in range(checkpoint['args']['labels']):
                    token_sal[int(cls_)] = class_attr_list[cls_][instance_i][token_i]
                saliencies.append(token_sal)
            out.write(json.dumps({'tokens': saliencies}) + '\n')
            out.flush()

    return saliency_flops

#-----Main Body------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default='PSE',
        type=str,
        choices=['snli', 'imdb', 'tweet', 'PSE']
    )
    parser.add_argument("--dataset_dir", default='data/PSE/dataset/', type=str)
    parser.add_argument("--split", default='test', type=str, choices=['train', 'test'])
    parser.add_argument("--no_time", action='store_true', default=False)
    parser.add_argument(
        "--model",
        default='deberta',
        choices=['deberta', 'roberta', 'roberta_large', 'gpt2', 'bioroberta', 'roberta_lora'],
        type=str
    )
    parser.add_argument("--models_dir", default='data/models/PSE/deberta/transformer_PSE', type=str)
    parser.add_argument("--gpu", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--output_dir", default='data/saliency/PSE/deberta/transformer/', type=str)
    parser.add_argument("--sw", type=int, default=1)
    parser.add_argument("--saliency", nargs='+')
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    #fix seed for reproduction
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    #load tokenizer
    if args.model == 'deberta':
        tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
    elif args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.add_tokens(['[MASK]'])
        tokenizer.mask_token = '[MASK]'
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids('[MASK]')
    elif args.model == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    elif args.model == 'roberta_large':
        tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    elif args.model == 'bioroberta':
        tokenizer = RobertaTokenizer.from_pretrained("RoBERTa-base-PM-M3-Voc-distill-align-hf")
    elif args.model == 'roberta_lora':
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    else:
        print('Wrong model entered')

    for saliency in args.saliency:
        if saliency in ['guided', 'sal', 'inputx', 'deeplift']:
            aggregations = ['mean', 'l2']
        else:
            aggregations = ['none'] #Occlusion -> none
        for aggregation in aggregations:
            flops = []
            models_dir = args.models_dir
            base_model_name = models_dir.split('/')[-1]
            for m_i in range(1, 6):
                curr_flops = generate_saliency(
                    os.path.join(models_dir + f'_{m_i}'),
                    os.path.join(args.output_dir, f'{base_model_name}_{m_i}_{saliency}_{aggregation}'),
                    saliency,
                    aggregation
                )
                flops.append(np.average(curr_flops) if len(curr_flops) > 0 else 0)
            print(np.average(flops), np.std(flops))
