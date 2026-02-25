from core import *
from core.layers import *
from core.transforms import *
from core.logging import Wandb
from core.lr_scheduler import CosineAnnealing
from core import Model
from core.layers import *
from core.transforms import *
from core.datasets import CIFAR10
from core.losses import CrossEntropy

model = Model([
    Conv(3, 64, 3, padding=1),
    BatchNorm2d(),
    ReLU(),
    Conv(64, 64, 3, padding=1),
    BatchNorm2d(),
    ReLU(),
    MaxPool(2),

    Conv(64, 128, 3, padding=1),
    BatchNorm2d(),
    ReLU(),
    Conv(128, 128, 3, padding=1),
    BatchNorm2d(),
    ReLU(),
    MaxPool(2),

    Conv(128, 256, 3, padding=1),
    BatchNorm2d(),
    ReLU(),
    Conv(256, 256, 3, padding=1),
    BatchNorm2d(),
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

test_transform = Compose([
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
  transform = test_transform
)

trainer = Trainer(
    model=model,
    loss=CrossEntropy(),
    lr_sched=CosineAnnealing(lr_max=0.1, lr_min=0.0, T_max=80),
    logger=Wandb(project_name="cnn-from-scratch", run_name="cifar10-cnn")  
)

trainer.fit(
    train_data,
    test_data,
    epochs=80,
    batch_size=64,
)
