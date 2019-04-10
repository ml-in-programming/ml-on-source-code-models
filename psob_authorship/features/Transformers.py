import torch

from psob_authorship.features.utils import ZERO_DIVISION_RETURN


class Transformers:
    @staticmethod
    def set_mean_to_indices(default_value_indices, mean_values: torch.Tensor, features: torch.Tensor):
        for row in range(features.shape[0]):
            for col in range(features.shape[1]):
                if default_value_indices[row][col] == 1:
                    features[row][col] = mean_values[col]

    @staticmethod
    def set_mean_to_zero_division(mean_values: torch.Tensor, features: torch.Tensor):
        default_value_indices = features == ZERO_DIVISION_RETURN
        Transformers.set_mean_to_indices(default_value_indices, mean_values, features)

    @staticmethod
    def transform_metrics_default_values_to_mean(default_value, features: torch.Tensor) -> torch.Tensor:
        default_value_indices = features == default_value
        features[default_value_indices] = 0
        mean_values = torch.mean(features, dim=0)
        Transformers.set_mean_to_indices(default_value_indices, mean_values, features)
        return mean_values

    @staticmethod
    def transform_metrics_zero_division_to_mean(features: torch.Tensor) -> torch.Tensor:
        return Transformers.transform_metrics_default_values_to_mean(ZERO_DIVISION_RETURN, features)

    @staticmethod
    def transform_metrics_zero_division_to_new_value(new_value, features: torch.Tensor):
        Transformers.transform_metrics_default_values_to_new_value(ZERO_DIVISION_RETURN, new_value, features)

    @staticmethod
    def transform_metrics_default_values_to_new_value(default_value, new_value, features: torch.Tensor):
        default_value_indices = features == default_value
        features[default_value_indices] = new_value

    @staticmethod
    def transform_metrics_zero_division_to_zero(features: torch.Tensor):
        Transformers.transform_metrics_zero_division_to_new_value(0, features)

    @staticmethod
    def transform_metrics_zero_division_to_one(features: torch.Tensor):
        Transformers.transform_metrics_zero_division_to_new_value(1, features)
