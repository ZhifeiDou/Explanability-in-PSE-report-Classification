########################################################################################################################
#    This code is modified on top of: https://github.com/copenlu/xai-benchmark  for academic research purpose          #
########################################################################################################################


"""Evaluate confidence measure."""
import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import csv
from scipy.special import softmax
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler


# Sample Function--------------------------------------------
def sample(X, y, mode='up'):
    buckets_idx = defaultdict(lambda: [])
    buckets_size = defaultdict(lambda: 0)
    for i, _y in enumerate(y):
        buckets_size[int(_y * 10)] += 1
        buckets_idx[int(_y * 10)].append(i)

    if mode == 'up':
        sample_size = max(list(buckets_size.values()))
    elif mode == 'down':
        sample_size = min(list(buckets_size.values()))
    elif mode == 'mid':
        sample_size = (
            max(list(buckets_size.values()))
            - min(list(buckets_size.values()))
        ) // 2
    else:
        sample_size = min(list(buckets_size.values()))

    new_idx = []

    for _, bucket_ids in buckets_idx.items():
        do_replace = True
        if sample_size <= len(bucket_ids):
            do_replace = False
        chosen = np.random.choice(bucket_ids, sample_size, replace=do_replace)
        new_idx += chosen.tolist()

    random.shuffle(new_idx)
    return X[new_idx], y[new_idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", help="Path where the models can be found", default='snli_roberta', type=str)
    parser.add_argument("--saliency_dir", help="Directory where saliencies are serialized", type=str)
    parser.add_argument("--saliency", help="Saliency name", nargs='+')
    parser.add_argument("--upsample", choices=['up', 'none'])

    args = parser.parse_args()
    np.random.seed(1)

    print(args, flush=True)

    test_scores = []

    for saliency in args.saliency:
        print(saliency)
        
        per_model_scores = []

        for model_path in os.listdir(args.models_dir):
            if model_path.endswith('.predictions'):
                continue

            full_model_path = os.path.join(args.models_dir, model_path)
            predictions_path = full_model_path + '.predictions'
            saliency_path = os.path.join(args.saliency_dir, f"{model_path}_{saliency}")

            predictions = json.load(open(predictions_path))
            class_preds = predictions['class']
            logits = predictions['logits']

            classes = [0, 1, 2]
            if 'imdb' in args.saliency_dir:
                classes = [0, 1]
            if 'PSE' in args.saliency_dir:
                classes = [0, 1, 2, 3, 4, 5, 6] #modify this to 6 classes to accommodate PSE dataset

            features = []
            y = []

            saliencies = []
            with open(saliency_path) as out:
                for i, line in enumerate(out):
                    try:
                        instance_saliency = json.loads(line)
                    except:
                        continue

                    instance_sals = []
                    for _cls in classes:
                        cls_sals = []
                        for _token in instance_saliency['tokens']:
                            if _token['token'] == '[PAD]':
                                break
                            cls_sals.append(_token[str(_cls)])
                        instance_sals.append(cls_sals)
                    saliencies.append(instance_sals)

            for i, instance in enumerate(saliencies):
                pred_cls = class_preds[i]
                instance_saliency = saliencies[i]
                instance_logits = softmax(logits[i])

                confidence_pred = instance_logits[pred_cls]
                saliency_pred = np.array(instance_saliency[pred_cls])

                left_classes = classes.copy()
                left_classes.remove(pred_cls)
                other_sals = [np.array(instance_saliency[c_]) for c_ in left_classes]

                feats = []
                if len(classes) == 2:
                    feats.append(sum(saliency_pred - other_sals[0]))
                    feats.append(sum(saliency_pred - other_sals[0]))
                    feats.append(sum(saliency_pred - other_sals[0]))
                else:
                    feats.append(
                        sum(
                            np.max([saliency_pred - other_sals[0],
                                    saliency_pred - other_sals[1]], axis=0)
                        )
                    )
                    feats.append(
                        sum(
                            np.mean([saliency_pred - other_sals[0],
                                     saliency_pred - other_sals[1]], axis=0)
                        )
                    )
                    feats.append(
                        sum(
                            np.min([saliency_pred - other_sals[0],
                                    saliency_pred - other_sals[1]], axis=0)
                        )
                    )

                y.append(confidence_pred)
                features.append(feats)

            features = np.array(features)
            features = MinMaxScaler().fit_transform(features)
            y = np.array(y)

            rs = ShuffleSplit(n_splits=5, random_state=2)
            fold_scores = []
            for train_index, test_index in rs.split(features):
                X_train, y_train = features[train_index], y[train_index]
                X_test, y_test = features[test_index], y[test_index]

                if args.upsample == 'up':
                    X_train, y_train = sample(X_train, y_train, mode='up')

                reg = LinearRegression().fit(X_train, y_train)
                test_pred = reg.predict(X_test)

                fold_mae = mean_absolute_error(y_test, test_pred)
                fold_max = max_error(y_test, test_pred)
                fold_scores.append([fold_mae, fold_max])

            per_model_avg = [
                np.mean([fs[0] for fs in fold_scores]),
                np.mean([fs[1] for fs in fold_scores])
            ]
            per_model_scores.append(per_model_avg)

        for metric_i, metric_name in enumerate(["MAE", "MaxError"]):
            metric_vals = [ms[metric_i] for ms in per_model_scores]
            mean_val = np.mean(metric_vals)
            std_val = np.std(metric_vals)
            print(
                f"{saliency} | {metric_name} = {mean_val:.3f} "
                f"(± {std_val:.3f})",
                flush=True
            )
        path = args.models_dir
        model = path.split('/')[-2]
        csv_file = f"evaluation_results/confidence/confidence_results_{model}.csv"
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)

        write_header = not os.path.exists(csv_file)

        mae_vals = [ms[0] for ms in per_model_scores]
        maxerr_vals = [ms[1] for ms in per_model_scores]
        mae_mean = np.mean(mae_vals)
        mae_std = np.std(mae_vals)
        maxerr_mean = np.mean(maxerr_vals)
        maxerr_std = np.std(maxerr_vals)

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "saliency",
                    "mae_mean",
                    "mae_std",
                    "maxerr_mean",
                    "maxerr_std"
                ])
            writer.writerow([
                saliency,
                f"{mae_mean:.3f}",
                f"{mae_std:.3f}",
                f"{maxerr_mean:.3f}",
                f"{maxerr_std:.3f}"
            ])
