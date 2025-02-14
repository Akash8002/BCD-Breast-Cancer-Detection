from keras.callbacks import CSVLogger



class CustomCSVLogger(CSVLogger):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': self.model.optimizer.lr.numpy()})
        super().on_epoch_end(epoch, logs)
