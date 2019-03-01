class ModelConfiguration:
    def __init__(self, classes, n_units, n_layers, dropout, node_id_getter, cell, residual) -> None:
        super().__init__()
        self.cell = cell
        self.feature_dict = node_id_getter
        self.dropout = dropout
        self.n_layers = n_layers
        self.residual = residual
        self.classes = classes
        self.n_labels = len(classes)
        self.n_units = n_units
