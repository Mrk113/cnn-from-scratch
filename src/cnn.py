from core import Model
from core.layers import *
from core.transforms import *
from core.datasets import CIFAR10
from core.losses import CrossEntropy

model = Model([
    Conv(3, 64, 3, padding=1),
    BatchNorm(),
    ReLU(),
    Conv(64, 64, 3, padding=1),
    BatchNorm(),
    ReLU(),
    MaxPool(2),

    Conv(64, 128, 3, padding=1),
    BatchNorm(),
    ReLU(),
    Conv(128, 128, 3, padding=1),
    BatchNorm(),
    ReLU(),
    MaxPool(2),

    Conv(128, 256, 3, padding=1),
    BatchNorm(),
    ReLU(),
    Conv(256, 256, 3, padding=1),
    BatchNorm(),
    ReLU(),
    MaxPool(2),

    Flatten(),
    FC(256 * 4 * 4, 512),
    ReLU(),
    FC(512, 10),
])

train_transform = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomCrop(size=(32, 32), padding=4),
    Scale(),
    Normalize(mean=CIFAR10.mean, std=CIFAR10.std, axis=1),
])

train_data = CIFAR10(
  root = "data",
  train = True,
  download = True,
  transform = train_transform
)

test_data = CIFAR10(
  root = "data",
  train = False,
  download = True,
)

model.config(
    loss=CrossEntropy(),
    lr=0.01
)

history = model.fit(
    train_data,
    None,
    epochs=10,
    batch_size=64,
)

history.plot()
history.wandb()