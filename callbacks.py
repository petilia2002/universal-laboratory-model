import tensorflow as tf
import os

# If you place a stop file in the working directory, the model will stop training.


class MessagerCallback(tf.keras.callbacks.Callback):

    def __init__(self, **kwargs):
        pass

    def on_epoch_end(self, epoch, logs={}):
        if os.path.exists("stop"):
            self.model.stop_training = True
        return
