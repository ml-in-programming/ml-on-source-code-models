import torch


def print_model_accuracy_before_train(model, test_features, test_labels, print_info):
    correct = 0
    total = 0
    outputs = model(test_features)
    _, predicted = torch.max(outputs.data, 1)
    total += test_labels.size(0)
    correct += (predicted == test_labels).sum().item()
    accuracy = correct / total
    print_info("Accuracy of model before training = " + str(accuracy))
