from __future__ import print_function, division

import os
import torch
from torch.utils.data import Dataset


class AuthorsFilesDataset(Dataset):
    """
    Class used to iterate on dataset of files written by specific authors.
    Can be used for author identification problem.

    Dataset must contain list of files of following names: <filename>.<author>.<extension>

    Example of dataset with python codejam solutions:
        root_path = /dataset/data/py/
        root_path contains the following files:
            p1483485.Bastiandantilus0.py
            p1483485.Fizu0.py
            p1483485.J3ffreySmith0.py
            p1483488.Bastiandantilus0.py
            p1483488.Fizu0.py
            p1483488.J3ffreySmith0.py
    """

    def __init__(self, root_path, ast_generator) -> None:
        """
        Constructor of dataset

        :param root_path: path to the root of dataset
        :param ast_generator: a function that generates ast of given file, must receive file_path and return AST node
        """
        super().__init__()
        self.root_path = root_path
        self.ast_generator = ast_generator
        self.labels = [name.split('.')[1] for name in os.listdir(root_path)]
        self.file_paths = [os.path.join(root_path, name) for name in os.listdir(root_path)
                           if os.path.isfile(os.path.join(root_path, name))]

    def __getitem__(self, index):
        """
        Overridden method of PyTorch Dataset.
        :param index: index in data sequence
        :return: pair:ast node and author name of it (aka label).
        """
        return {"ast": self.ast_generator(self.file_paths[index]), "label": self.labels[index]}

    def __len__(self):
        return len(self.file_paths)
