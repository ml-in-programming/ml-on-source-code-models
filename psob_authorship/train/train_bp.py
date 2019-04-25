import torch

from psob_authorship.features.PsobDataset import PsobDataset
from psob_authorship.train.utils import print_model_accuracy_before_train


def train_bp(model, train_features, train_labels, test_features, test_labels, config):
    print_info = config['pso_options']['print_info']
    print_model_accuracy_before_train(model, test_features, test_labels, print_info)

    criterion = config['criterion']()
    optimizer = config['optimizer'](model.parameters(), lr=config['lr'])

    trainloader = torch.utils.data.DataLoader(
        PsobDataset(train_features, train_labels),
        batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        PsobDataset(test_features, test_labels),
        batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=2
    )

    best_accuracy = -1.0
    current_duration = 0
    for epoch in range(config['epochs']):
        for data in trainloader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                features, labels = data
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        if best_accuracy >= accuracy:
            current_duration += 1
        else:
            current_duration = 0
        best_accuracy = max(best_accuracy, accuracy)
        if current_duration > config['early_stopping_rounds']:
            print_info("On epoch " + str(epoch) + " training was early stopped")
            break
        if epoch % 100 == 0:
            print_info(
                "CHECKPOINT EACH 100th EPOCH " + str(epoch) + ": current accuracy " + str(accuracy) + " , best "
                + str(best_accuracy))

    correct = 0
    total = 0
    labels_dist = torch.zeros(config['number_of_authors'])
    labels_correct = torch.zeros(config['number_of_authors'])
    with torch.no_grad():
        for data in testloader:
            features, labels = data
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i, label in enumerate(labels):
                labels_dist[label] += 1
                labels_correct[label] += predicted[i] == labels[i]
    print_info('Finished training')
    print_info('Best accuracy: ' + str(max(best_accuracy, correct / total)))
    print_info('Accuracy of the last validation of the network: %d / %d = %d %%' %
               (correct, total, 100 * correct / total))
    print_info("Correct labels / labels for each author of last validation:\n" +
               str(torch.stack((labels_correct, labels_dist), dim=1)))
