import torch


def print_model_accuracy_and_loss_before_train(model, criterion,
                                               train_features, train_labels, test_features, test_labels,
                                               print_info):
    correct = 0
    total = 0
    outputs = model(test_features)
    _, predicted = torch.max(outputs.data, 1)
    total += test_labels.size(0)
    correct += (predicted == test_labels).sum().item()
    accuracy = correct / total
    outputs = model(train_features)
    train_loss = criterion(outputs, train_labels).item()
    outputs = model(test_features)
    test_loss = criterion(outputs, test_labels).item()
    print_info("INITIAL DATA: " + "accuracy " + str(accuracy) +
               ", train loss " + str(train_loss) + ", test loss " + str(test_loss))
