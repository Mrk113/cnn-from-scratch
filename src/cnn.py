from core import *

X_train, y_train, X_test, y_test = load("cifar10", 
                                        flatten=False, 
                                        normalize=True, 
                                        label_encoding="index")

network = [
    Conv(3, 64, 3),
    ReLU(),
    Conv(64, 64, 3),
    ReLU(),
    MaxPool(2),

    Conv(64, 128, 3),
    ReLU(),
    Conv(128, 128, 3),
    ReLU(),
    MaxPool(2),

    Conv(128, 256, 3),
    ReLU(),
    Conv(256, 256, 3),
    ReLU(),
    MaxPool(2),

    Flatten(),
    FC(256 * 4 * 4, 512),
    ReLU(),
    FC(512, 10),
]

training_config = TrainConfig(
    epochs=80,
    batch_size=128,
    learning_rate=0.001,
    verbose=True,
    eval_every=1,
)

history = train(
    network,
    CrossEntropy(),
    Data(X_train, y_train),
    config=training_config,
    val_data=Data(X_test, y_test)
)