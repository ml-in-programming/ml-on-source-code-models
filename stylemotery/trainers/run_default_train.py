import argparse
import random

import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.model_selection import StratifiedKFold

from stylemotery.ast_generators.AstGeneratorsFactory import AstGeneratorsFactory
from stylemotery.dataset.AuthorsFilesDataset import AuthorsFilesDataset
from stylemotery.models.ModelConfiguration import ModelConfiguration
from stylemotery.models.ModelsFactory import ModelsFactory
from stylemotery.trainers.Trainer import Trainer


def parse_arguments():
    parser = argparse.ArgumentParser(description='Runs training of stylemotery')
    parser.add_argument('--dataset', '-d', type=str,
                        help='Root path of dataset where all files to train on are listed', required=True)
    parser.add_argument('--language', '-lang', type=str,
                        help='Language of files, used to generate AST', required=True)
    parser.add_argument('--batch_size', '-b', type=int, default=1,
                        help='Number of examples in each mini batch')
    parser.add_argument('--num_workers', '-nw', type=int, default=4,
                        help='Number of workers to load data (specific for DataLoader in PyTorch)')
    parser.add_argument('--shuffle', '-sh', type=bool, default=True,
                        help='Shuffle data (true or false)')
    parser.add_argument('--n_folds', '-nf', type=int, default=3,
                        help='Number of folds in k-fold cross validation')
    parser.add_argument('--shuffle_folds', '-sf', type=bool, default=True,
                        help='Shuffle data before splitting into folds')
    parser.add_argument('--seed', '-sd', type=int, default=random.randint(0, 4294967295),
                        help='Seed for shuffle in k-fold cross validation')
    parser.add_argument('--model', '-m', type=str, default="lstm",
                        help='Model used for this experiment')
    parser.add_argument('--units', '-u', type=int,default=100,
                        help='Number of hidden units')
    parser.add_argument('--dropout', '-dr', type=float, default=0.2,
                        help='Dropout coefficient')
    parser.add_argument('--cell', '-cl', type=str, default="lstm",
                        help='lstm')
    parser.add_argument('--residual', '-r', action='store_true', default=False,
                        help='Number of examples in each mini batch')
    parser.add_argument('--layers', '-l', type=int, default=1,
                        help='Number of Layers for LSTMs')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01,
                        help='Learning rate for SGD optimizer')
    parser.add_argument('--optimizer_momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--epoch', '-ep', type=int, default=10,
                        help='Number of epochs')
    return parser.parse_args()


def run_train():
    """
    Runs training of model.
    Parses a lot of arguments, please see code for more explanations.
    Essential arguments are:
    -d - path to dataset root
    :return: None
    """
    args = parse_arguments()
    ast_generator = AstGeneratorsFactory.create(args.language)
    authors_files_dataset = AuthorsFilesDataset(args.dataset, ast_generator)

    trees = np.array([authors_files_dataset[i]["ast"] for i in range(0, len(authors_files_dataset))])
    labels = np.array([authors_files_dataset[i]["label"] for i in range(0, len(authors_files_dataset))])
    classes = np.unique(labels)

    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=args.shuffle_folds, random_state=args.seed)
    train_indices, test_indices = next(cv.split(trees, labels))
    train_trees, train_labels = trees[train_indices], labels[train_indices]
    test_trees, test_labels = trees[test_indices], labels[test_indices]

    model_configuration = ModelConfiguration(classes, args.units, args.layers, args.dropout,
                                             ast_generator.get_node_id, args.cell, args.residual)
    model = ModelsFactory.create(args.model, model_configuration)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate, momentum=args.optimizer_momentum)
    trainer = Trainer(model, criterion, optimizer, classes, args.batch_size)

    best_scores = (-1, -1, -1)
    for epoch in range(1, args.epoch + 1):
        print('Epoch: {0:d} / {1:d}'.format(epoch, args.epoch))
        print("optimizer lr = ", optimizer.lr)
        print('Train')
        train_accuracy, train_loss = trainer.train(train_trees, train_labels)
        print("\tAccuracy: %0.2f " % train_accuracy)
        print('Test')
        test_accuracy, test_loss = trainer.test(test_trees, test_labels)
        print("\tAccuracy: %0.2f " % test_accuracy)
        print()

        if args.save > 0 and epoch > 0:
            epoch_, loss_, acc_ = best_scores
            if test_accuracy > acc_ or (test_accuracy >= acc_ and test_loss <= loss_):
                best_scores = (epoch, test_loss, test_accuracy)

        if epoch >= 5 and (test_loss < 0.001 or test_accuracy >= 1.0):
            print("\tEarly Stopping")
            break


if __name__ == "__main__":
    run_train()
