import random

import torch
from sklearn.metrics import accuracy_score


class Trainer(object):
    def __init__(self, model, criterion, optimizer, classes, batch_size, shuffle=True):
        super(Trainer, self).__init__()
        self.classes = classes
        self.batch_size = batch_size
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.shuffle = shuffle
        self.epoch = 0

    def train(self, trees, labels):
        self.model.train()
        self.optimizer.zero_grad()
        batch_loss = 0
        total_loss = torch.tensor()
        predict = torch.tensor()

        if self.shuffle:
            indices = torch.arange(len(labels))
            random.shuffle(indices)
            trees = trees[indices]
            labels = labels[indices]

        for idx, tree in enumerate(trees):
            output = self.model(tree)
            target = [1 if labels[idx] == a_class else 0 for a_class in self.classes]
            batch_loss += self.criterion.loss(output, target)
            _, predicted = torch.max(output)
            predict.append(predicted)
            if (idx + 1) % self.batch_size == 0:
                total_loss.append(float(batch_loss.data) / self.batch_size)
                self.optimizer.backward()
                self.optimizer.step()
                self.optimizer.zerograds()
                batch_loss = 0
        predict = torch.tensor(predict)
        accuracy = accuracy_score(predict, labels)
        return accuracy, torch.mean(total_loss)


    def test(self, trees, labels):
        self.model.eval()
        with torch.no_grad():
            batch_loss = 0
            total_loss = torch.tensor()
            predict = torch.tensor()
            for idx, tree in enumerate(trees):
                output = self.model(tree)
                target = [1 if labels[idx] == a_class else 0 for a_class in self.classes]
                batch_loss += self.criterion.loss(output, target)
                _, predicted = torch.max(output)
                predict.append(predicted)
                if idx % self.batch_size == 0:
                    total_loss.append(float(batch_loss.data) / self.batch_size)
                    batch_loss = 0
            accuracy = accuracy_score(predict, labels)
            mean_loss = torch.mean(total_loss)
            return accuracy, mean_loss
