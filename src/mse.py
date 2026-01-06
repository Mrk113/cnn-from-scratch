from core import *

X_train, y_train, X_test, y_test = load("mnist",
                                        flatten=True,
                                        normalize=True,
                                        label_encoding="one_hot")

network = [
    FC(784, 256),
    ReLU(),
    FC(256, 128),
    ReLU(),
    FC(128, 10),
    Softmax()
]

training_config = TrainConfig(
    epochs=30,
    batch_size=64,
    learning_rate=0.01,
    verbose=True,
)

history = train(
    network,
    MSE(),
    Data(X_train, y_train),
    config=training_config,
    val_data=Data(X_test, y_test)
)

log(history)