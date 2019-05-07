import torch

from psob_authorship.train.utils import print_evaluation_before_train


def train_pso(model, train_features, train_labels, test_features, test_labels, config):
    print_info = config['pso_options']['print_info']

    criterion = config['criterion']
    optimizer = config['pso_optimizer'](model, criterion, config['pso_options'], config['n_particles'])

    print_evaluation_before_train(model, criterion,
                                  train_features, train_labels,
                                  test_features, test_labels,
                                  print_info)
    loss, _ = optimizer.optimize(train_features, train_labels,
                                 test_features, test_labels,
                                 config['pso_iters'], config['pso_velocity_clamp'])
    print_info("Train loss after PSO optimizing = " + str(loss))

    correct = 0
    total = 0
    outputs = model(train_features)
    _, predicted = torch.max(outputs.data, 1)
    total += train_labels.size(0)
    correct += (predicted == train_labels).sum().item()
    print_info('Train accuracy of the network: %d / %d = %d %%' % (correct, total, 100 * correct / total))

    correct = 0
    total = 0
    labels_dist = torch.zeros(config['number_of_authors'])
    labels_correct = torch.zeros(config['number_of_authors'])
    outputs = model(test_features)
    _, predicted = torch.max(outputs.data, 1)
    total += test_labels.size(0)
    correct += (predicted == test_labels).sum().item()
    for i, label in enumerate(test_labels):
        labels_dist[label] += 1
        labels_correct[label] += predicted[i] == test_labels[i]
    final_accuracy = correct / total
    print_info('Final accuracy of the network: %d / %d = %d %%' % (correct, total, 100 * final_accuracy))
    print_info("Correct labels / labels for each author:\n" + str(torch.stack((labels_correct, labels_dist), dim=1)))
    print_info("END OF PSO TRAINING")
    return final_accuracy
