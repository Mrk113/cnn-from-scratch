from core.utils.train import train
from core.losses.mse import MSE


X, Y = load()  

# miniVGG
network = [
    Conv,
    ReLu,
    Conv,
    ReLu,

    Pool,

    Conv,
    ReLu,
    Conv,
    ReLu,

    Pool,

    FC,
    ReLu,
    FC,
    ReLu,
]

train(network, MSE, X, Y, epochs=250, learning_rate=0.01, batch_size=32, verbose=True)