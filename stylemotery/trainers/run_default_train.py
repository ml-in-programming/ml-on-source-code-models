import argparse

from torch.utils.data import DataLoader

from stylemotery.ast_generators.AstGeneratorsFactory import AstGeneratorsFactory
from stylemotery.dataset.AuthorsFilesDataset import AuthorsFilesDataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='Runs training of stylemotery')
    parser.add_argument('--dataset', '-d', type=str,
                        help='Root path of dataset where all files to train on are listed', required=True)
    parser.add_argument('--language', '-l', type=str,
                        help='Language of files, used to generate AST', required=True)
    parser.add_argument('--batch_size', '-b', type=int, default=1,
                        help='Number of examples in each mini batch')
    parser.add_argument('--num_workers', '-nw', type=int, default=4,
                        help='Number of workers to load data (specific for DataLoader in PyTorch)')
    parser.add_argument('--shuffle', '-sh', type=bool, default=True,
                        help='Shuffle data (true or false)')
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
    data_loader = DataLoader(authors_files_dataset, batch_size=args.batch_size,
                             shuffle=args.shuffle, num_workers=args.num_workers)
    a = next(iter(data_loader))
    for inputs, labels in data_loader:
        print("Inputs: " + inputs)
        print("Labels: " + labels)


if __name__ == "__main__":
    run_train()
