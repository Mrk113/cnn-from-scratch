from core import *
from core.layers import *
from core.transforms import *
from core.logging import Wandb
from core.datasets import CIFAR10
from core.losses import CrossEntropy
from core.lr_scheduler import StepLR


# Simple 3-layer MLP for CIFAR-10 (input 3x32x32 -> 10 classes)
model = Model([
    Flatten(),
    FC(3 * 32 * 32, 1024),
    ReLU(),
    FC(1024, 512),
    ReLU(),
    FC(512, 10),
])


train_transform = Compose([
    Scale(),
    Normalize(mean=CIFAR10.mean, std=CIFAR10.std, axis=1),
])

test_transform = Compose([
    Scale(),
    Normalize(mean=CIFAR10.mean, std=CIFAR10.std, axis=1),
])


train_data = CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=train_transform,
)

test_data = CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=test_transform,
)


trainer = Trainer(
    model=model,
    loss=CrossEntropy(),
    lr_sched=StepLR(lr=0.01),
    logger=Wandb(project_name="cnn-from-scratch", run_name="mlp-base"),
)


trainer.fit(
    train_data,
    test_data,
    epochs=80,
    batch_size=64,
)
