import os

import torch

from skorch import NeuralNetClassifier
from tqdm import tqdm

from psob_authorship.features.java.MetricsCalculator import MetricsCalculator
from psob_authorship.model.Model import Model


def get_labeled_data():
    dataset_path = "../dataset"
    ast_path = "../asts"
    metrics_calculator = MetricsCalculator(dataset_path, ast_path)
    features = []
    labels = []
    for author_id, author in enumerate(tqdm(os.listdir(dataset_path))):
        for root, dirs, files in os.walk(os.path.join(dataset_path, author)):
            for file in files:
                filepaths = {os.path.abspath(os.path.join(root, file))}
                files_features = metrics_calculator.get_metrics(filepaths)
                features.append(torch.tensor(files_features))
                labels.append(author_id)
    return torch.stack(tuple(features)), torch.tensor(labels)


def run_train():
    features, labels = get_labeled_data()
    net = NeuralNetClassifier(Model, max_epochs=20, lr=0.1)
    net.fit(features, labels)
    print(net.predict(features[:10]))
    print(labels[:10])
    print(net.predict_proba(features[:10]))


if __name__ == '__main__':
    run_train()
