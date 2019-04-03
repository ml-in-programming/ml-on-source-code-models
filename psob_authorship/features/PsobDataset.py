from torch.utils.data import Dataset


class PsobDataset(Dataset):
    def __init__(self, features, labels, metrics=None) -> None:
        super().__init__()
        self.features = features
        self.labels = labels
        if metrics is None:
            self.metrics = [i for i in range(self.features.shape[1])]
        else:
            self.metrics = metrics

    def __getitem__(self, index):
        return self.features[index][self.metrics], self.labels[index]

    def __len__(self):
        return len(self.labels)
