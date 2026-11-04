"""wine: Logistic Regression using Keras 3 (torch backend, single Dense layer)."""
import os
os.environ.setdefault('KERAS_BACKEND', 'torch')
import keras
from shared.datasets import EPOCHS, LEARNING_RATE, SEED

def train(X, y, n_classes):
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential([
        keras.layers.Input(shape=(X.shape[1],)),
        keras.layers.Dense(n_classes, activation=None,
                           kernel_initializer='zeros', bias_initializer='zeros'),
    ])
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    model.fit(X.astype('float32'), y, epochs=EPOCHS,
              batch_size=len(X), shuffle=False, verbose=0)
    return model
