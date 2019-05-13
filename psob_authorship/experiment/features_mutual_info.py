import datetime
import logging
import os
import time

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from torch import optim, nn

from psob_authorship.experiment.utils import make_experiment_reproducible
from psob_authorship.features.java.MetricsCalculator import MetricsCalculator
from psob_authorship.features.utils import configure_logger_by_default
from psob_authorship.model.Model import Model
from psob_authorship.train.train_bp import train_bp

CONFIG = {
    'experiment_name': os.path.basename(__file__).split('.')[0],
    'experiment_notes': "draw bar of features mutual info",
    'number_of_authors': 40,
    'labels_features_common_name': "../calculated_features/extracted_for_each_file",
    'random_state': 4562,
    'pso_options': {}
}
INPUT_FEATURES = torch.load(CONFIG['labels_features_common_name'] + "_features.tr").numpy()
INPUT_LABELS = torch.load(CONFIG['labels_features_common_name'] + "_labels.tr").numpy()
make_experiment_reproducible(CONFIG['random_state'])


def get_features_mutual_info(file_to_print):
    logger = logging.getLogger(CONFIG['experiment_name'])
    configure_logger_by_default(logger)
    logger.info("START " + CONFIG['experiment_name'])

    def print_info(info):
        logger.info(info)
        print(info)
        file_to_print.write(info + "\n")

    descrete_features = [False for _ in range(19)]
    descrete_features[9] = True
    descrete_features[18] = True
    mutual_info = mutual_info_classif(INPUT_FEATURES, INPUT_LABELS, discrete_features=descrete_features, random_state=CONFIG['random_state'])
    print_info("Mutual info: ")
    print_info(str(list(zip(range(mutual_info.shape[0]), zip(MetricsCalculator.ALL_METRICS, mutual_info)))))
    logger.info("END " + CONFIG['experiment_name'])
    return mutual_info, MetricsCalculator.ALL_METRICS


def draw_mutual_info_bar(mutual_info, metric_names):
    import matplotlib.pyplot as plt
    index = np.arange(len(metric_names))
    plt.bar(index, mutual_info)
    plt.xlabel('Metric')
    plt.ylabel('Mutual Info')
    plt.xticks(index, metric_names, rotation=90)
    plt.title('Mutual Info of Features')
    img_path_to_save_plot = "../experiment_result/" + CONFIG['experiment_name'] + "_"\
                            + str(datetime.datetime.now()) + ".png"
    plt.savefig(img_path_to_save_plot, bbox_inches='tight')


def conduct_features_mutual_info_experiment():
    with open("../experiment_result/" + CONFIG['experiment_name'] + "_" + str(datetime.datetime.now()), 'w') as f:
        f.write("Config: " + str(CONFIG) + "\n")
        start = time.time()
        mutual_info, metric_names = get_features_mutual_info(f)
        draw_mutual_info_bar(mutual_info, metric_names)
        end = time.time()
        execution_time = end - start
        f.write("Execution time: " + str(datetime.timedelta(seconds=execution_time)) + "\n")


if __name__ == '__main__':
    conduct_features_mutual_info_experiment()
