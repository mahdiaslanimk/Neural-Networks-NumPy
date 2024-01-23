import numpy as np
from Dense import Dense
from Dropout import Dropout


class Sequential:
    def __init__(self):
        self.layers = []
        self.metrics = None
        self.history = {"loss": []}

    def add(self, layer):
        if len(self.layers) > 0 and layer.w.shape[1] != self.layers[-1].w.shape[0]:
            raise Exception(
                "New layer's shape is  not true \n"
                + "New  layer: %s\n" % (str(self.layers[-1].w.shape))
                + "Last layer: %s" % (str(layer.w.shape))
            )
        else:
            self.layers.append(layer)

    def compile(self, optimizer, loss, metrics=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def forward(self, x, training=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
        return x

    def fit(
        self,
        x_train,
        y_train,
        epochs,
        batch_size=32,
        validation_split=0,
        validation_data=None,
    ):
        num_layers = len(self.layers)
        num_samples = x_train.shape[0]

        if validation_data:
            self.history["validation_data_loss"] = []

        if self.metrics:
            # Split the data into training and validation sets
            split_index = int((1 - validation_split) * num_samples)

            if validation_split == 0:
                x_valid, y_valid = x_train, y_train
            elif validation_split > 0:
                self.history["val_loss"] = []
                x_valid, y_valid = x_train[split_index:], y_train[split_index:]
                x_train, y_train = x_train[:split_index], y_train[:split_index]
            for metric in self.metrics:
                self.history[metric.name] = []
                self.history["val_" + metric.name] = []

        # training the network
        for epoch in range(epochs):
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]

                # forward pass
                y_hat = self.forward(x_batch, training=True)  # NN's output (pred.)

                # backward pass
                sens = self.loss.compute_derivative(y_batch, y_hat)
                for layer in self.layers[::-1]:
                    layer.s = sens * layer.df
                    sens = layer.backward()

                # Update parameters using optimizer
                params = [
                    param for layer in self.layers[1:] for param in layer.get_params()
                ]
                gradients = [
                    grad for layer in self.layers[1:] for grad in layer.get_gradients()
                ]
                self.optimizer.update(params, gradients)

            y_hat_train = self.forward(x_train)
            self.history["loss"].append(self.loss.compute_loss(y_train, y_hat_train))

            # Metrics
            if self.metrics:
                if validation_split > 0:
                    y_hat_valid = self.forward(x_valid)
                    self.history["val_loss"].append(
                        self.loss.compute_loss(y_valid, y_hat_valid)
                    )
                for metric in self.metrics:
                    self.history[metric.name].append(
                        metric.compute(y_train, y_hat_train)
                    )
                    if validation_split > 0:
                        self.history["val_" + metric.name].append(
                            metric.compute(y_valid, y_hat_valid)
                        )

                loss_print = (
                    f"Epoch {epoch + 1:2d} / {epochs:2d}, "
                    + f'loss: {self.history["loss"][-1]:2.3f}'
                    + (
                        f', val_loss: {self.history["val_loss"][-1]:2.3f}'
                        if validation_split > 0
                        else ""
                    )
                )

                metrics_print = ", ".join(
                    [
                        ", %s: %2.3f" % (metric.name, self.history[metric.name][-1])
                        + (
                            ", %s: %2.3f "
                            % (
                                "val_" + metric.name,
                                self.history["val_" + metric.name][-1],
                            )
                            if validation_split > 0
                            else ""
                        )
                        for metric in self.metrics
                    ]
                )

                print(loss_print + metrics_print)
            else:
                print(
                    f"Epoch {epoch + 1:2d} /{epochs:2d}, "
                    f'loss: {self.history["loss"][-1]:2.3f}'
                )

            if validation_data:
                self.history["validation_data_loss"].append(
                    self.loss.compute_loss(
                        validation_data[1], self.forward(validation_data[0])
                    )
                )
