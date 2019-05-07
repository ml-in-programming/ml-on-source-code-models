import torch


def get_model_accuracy_and_loss(model, criterion, features, labels):
    with torch.no_grad():
        outputs = model(features)
        loss = criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        correct = 0
        total = 0
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy


def get_model_accuracy_and_loss_for_train_and_test(model, criterion,
                                                   train_features, train_labels,
                                                   test_features, test_labels):
    train_loss, train_accuracy = get_model_accuracy_and_loss(model, criterion, train_features, train_labels)
    test_loss, test_accuracy = get_model_accuracy_and_loss(model, criterion, test_features, test_labels)
    return train_loss, test_loss, train_accuracy, test_accuracy


def print_100th_checkpoint_evaluation(epoch,
                                      model, criterion,
                                      train_features, train_labels,
                                      test_features, test_labels,
                                      print_info):
    train_loss, test_loss, train_accuracy, test_accuracy = \
        get_model_accuracy_and_loss_for_train_and_test(model, criterion,
                                                       train_features, train_labels,
                                                       test_features, test_labels)
    print_info(
        "CHECKPOINT EACH 100th EPOCH " + str(epoch) + ". Train Loss: " + str(train_loss) +
        " , Test Loss: " + str(test_loss) + ", Train Accuracy: " + str(train_accuracy) +
        ", Test Accuracy: " + str(test_accuracy)
    )


def print_evaluation_before_train(model, criterion,
                                  train_features, train_labels,
                                  test_features, test_labels,
                                  print_info):
    train_loss, test_loss, train_accuracy, test_accuracy = \
        get_model_accuracy_and_loss_for_train_and_test(model, criterion,
                                                       train_features, train_labels,
                                                       test_features, test_labels)
    print_info(
        "INITIAL DATA. Train Loss: " + str(train_loss) +
        " , Test Loss: " + str(test_loss) + ", Train Accuracy: " + str(train_accuracy) +
        ", Test Accuracy: " + str(test_accuracy)
    )
