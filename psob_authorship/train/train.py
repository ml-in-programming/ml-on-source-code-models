import os

import torch

from skorch import NeuralNetClassifier
from tqdm import tqdm

from psob_authorship.features.java_language_features import get_all_metrics
from psob_authorship.model.Model import Model


def get_labeled_data():
    root_dataset = "../dataset"
    features = []
    labels = []
    metrics = get_all_metrics()
    ast_path = "../asts"
    for author_id, author in enumerate(tqdm(os.listdir(root_dataset))):
        for root, dirs, files in os.walk(os.path.join(root_dataset, author)):
            for file in files:
                features.append([metric(os.path.join(root, file), ast_path) for metric in metrics])
                labels.append(author_id)
    return torch.tensor(features), torch.tensor(labels)


def run_train():
    features, labels = get_labeled_data()
    net = NeuralNetClassifier(Model, max_epochs=20, lr=0.1)
    net.fit(features, labels)
    print(net.predict(features[:10]))
    print(labels[:10])
    print(net.predict_proba(features[:10]))


if __name__ == '__main__':
    run_train()
