from stylemotery.models.RecursiveLSTM import RecursiveLSTM


class ModelsFactory:
    @staticmethod
    def create(name, configuration):
        if name == "lstm":
            return RecursiveLSTM(configuration)
