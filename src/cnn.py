from core import *

network = [
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
]

train_transform = [
    RandomHorizontalFlip(prob=0.5),
    RandomCrop(padding=4),
    Normalize()
]

label_transform = [
    OneHotEncode(10)
]

train_data = CIFAR10(
  root = "data",
  train=True,
  download=True,
  transform=train_transform
  target_transform=label_transform,
)

test_data = CIFAR10(
  root = "data",
  train=False,
  download=True,
)

config = TrainConfig(
    epochs=80,
    batch_size=64,
    optimzer=SGD()
    lr_scheduler=CosineAnnealingLR(0.0001),
    loss=CrossEntropy(),
)

train = Trainer(
    config,
    callbacks=[
      ProgessBar(),
      CSVLogger(),
      PythonPlot(),
      WandbLogger(),
    ]
)

train.fit(model, train_data, test_data)