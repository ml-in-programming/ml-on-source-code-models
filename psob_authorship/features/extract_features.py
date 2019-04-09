import os
import random

from typing import Tuple, Dict, List

import torch
from tqdm import tqdm

from psob_authorship.features.java.MetricsCalculator import MetricsCalculator
from psob_authorship.features.utils import chunks, ZERO_DIVISION_RETURN

CALCULATED_FEATURES_ROOT = "../calculated_features/"
N = 60


def get_labeled_data(author_max_files=None, split=None, dataset_path=os.path.abspath("../dataset"),
                     ast_path=os.path.abspath("../asts")) -> Tuple[torch.Tensor, torch.Tensor, List[str], Dict[int, str]]:
    metrics_calculator = MetricsCalculator(dataset_path, ast_path)
    features = []
    labels = []
    author_names = {}
    for author_id, author in enumerate(tqdm(os.listdir(dataset_path))):
        author_names[author_id] = author
        proccessed_files = 0
        for root, dirs, files in os.walk(os.path.join(dataset_path, author)):
            if len(files) == 0:
                continue
            random.shuffle(files)
            splitted_files = chunks(files, len(files)) if split is None else chunks(files, min(split, len(files)))
            for chunk in splitted_files:
                if author_max_files is not None and proccessed_files >= author_max_files:
                    break
                proccessed_files += len(chunk)
                filepaths = {os.path.abspath(os.path.join(root, file)) for file in chunk}
                files_features = metrics_calculator.get_metrics(filepaths)
                features.append(files_features)
                labels.append(author_id)
            if author_max_files is not None and proccessed_files >= author_max_files:
                break
    return torch.stack(tuple(features)), torch.tensor(labels), metrics_calculator.get_metric_names(), author_names


def extract_no_more_n_files_per_author(n):
    fileprefix = "no_more_" + str(n) + "_files_per_author_without_5th_features"
    filepath = os.path.join(CALCULATED_FEATURES_ROOT, fileprefix)
    features, labels, feature_names, author_names = get_labeled_data(n)
    save_extracted_features(filepath, features, labels, feature_names, author_names)


def extract_no_more_n_chunks(n):
    fileprefix = "no_more_" + str(n) + "_chunks"
    filepath = os.path.join(CALCULATED_FEATURES_ROOT, fileprefix)
    features, labels, feature_names, author_names = get_labeled_data(split=n)
    save_extracted_features(filepath, features, labels, feature_names, author_names)


def extract_for_each_file():
    fileprefix = "extracted_for_each_file"
    filepath = os.path.join(CALCULATED_FEATURES_ROOT, fileprefix)
    features, labels, feature_names, author_names = get_labeled_data()
    save_extracted_features(filepath, features, labels, feature_names, author_names)


def extract_with_transformation(fileprefix, transformation):
    filepath = os.path.join(CALCULATED_FEATURES_ROOT, fileprefix)
    features, labels, feature_names, author_names = get_labeled_data()
    transformation(ZERO_DIVISION_RETURN, features)
    save_extracted_features(filepath, features, labels, feature_names, author_names)


def extract_with_mean_default_values_for_each_file():
    extract_with_transformation("extracted_for_each_file",
                                MetricsCalculator.transform_metrics_default_values_to_mean)


def extract_with_zero_default_values_for_each_file():
    extract_with_transformation("extracted_for_each_file",
                                MetricsCalculator.transform_metrics_default_values_to_zero)


def extract_with_one_default_values_for_each_file():
    extract_with_transformation("extracted_for_each_file",
                                MetricsCalculator.transform_metrics_default_values_to_one)


def save_extracted_features(filepath: str, features: torch.Tensor, labels: torch.Tensor,
                            feature_names: List[str], author_names: Dict[int, str]):
    print("Shape of features: " + str(features.shape))
    print("Shape of labels: " + str(labels.shape))
    torch.save(features, filepath + "_features.tr")
    torch.save(labels, filepath + "_labels.tr")
    with open(filepath + "_additional_data", "w") as f:
        f.write("Feature names (order in features file is the same):\n")
        f.write(str(feature_names) + "\n")
        f.write("<label> <author_name>\n")
        f.write(str(author_names))


if __name__ == '__main__':
    extract_with_zero_default_values_for_each_file()
