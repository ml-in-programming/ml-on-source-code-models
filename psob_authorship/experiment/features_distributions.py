import os
from collections import defaultdict

from typing import Dict, Tuple, List

import torch
import yaml
import matplotlib.pyplot as plt

CONFIG = {
    'experiment_name': os.path.basename(__file__).split('.')[0],
    'labels_features_common_name': "../calculated_features/extracted_for_each_file",
    'path_save_plots': "../experiment_result/"
}


def get_features_data() -> Tuple[torch.Tensor, torch.Tensor, List[str], Dict[int, str]]:
    with open(CONFIG['labels_features_common_name'] + "_additional_data") as f:
        lines = f.readlines()
        feature_names = yaml.load(lines[1])
        author_names = yaml.load(lines[-1])
    return torch.load(CONFIG['labels_features_common_name'] + "_features.tr"), \
        torch.load(CONFIG['labels_features_common_name'] + "_labels.tr"), \
        feature_names, author_names


def draw_sorted_feature(feature_by_author: Dict[int, List[float]], feature_name: str, author_names: Dict[int, str]):
    plt.title(feature_name)
    plt.ylabel("Feature value")
    plt.xlabel("file(or group of files) index")
    for label, features in feature_by_author.items():
        features = sorted(features)
        x_axis = [i for i in range(len(features))]
        plt.plot(x_axis, features, '-o', label=str(label) + ":" + author_names[label])
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(CONFIG['path_save_plots'] + feature_name + "_feature_plot",
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()


def draw_features_sorted():
    features, labels, feature_names, author_names = get_features_data()
    for feature_id in range(features.shape[1]):
        feature_by_author = defaultdict(lambda: list())
        for i in range(labels.shape[0]):
            feature_by_author[labels[i].item()].append(features[i][feature_id].item())
        draw_sorted_feature(feature_by_author, feature_names[feature_id], author_names)


if __name__ == '__main__':
    draw_features_sorted()
