from core import *
from core.layers import *
from core.transforms import *
from core.logging import Wandb
from core.lr_scheduler import CosineAnnealing, StepLR
from core import Model
from core.layers import *
from core.transforms import *
from core.datasets import CIFAR10
from core.losses import CrossEntropy

# Mini-VGG: 3 conv blocks + 2 FC layers for CIFAR-10
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

# Training augmentation then normalization
train_transform = Compose([
    Scale(),
    Normalize(mean=CIFAR10.mean, std=CIFAR10.std, axis=1),
])

# Eval pipeline keeps only resizing and normalization
test_transform = Compose([
    Scale(),
    Normalize(mean=CIFAR10.mean, std=CIFAR10.std, axis=1),
])

# Load CIFAR-10 with our transforms
train_data = CIFAR10(
    root = "data",
    train = True,
    download = True,
    transform = train_transform
)

# Eval dataset with test transforms
test_data = CIFAR10(
    root = "data",
    train = False,
    download = True,
    transform = test_transform
)

# Trainer combines model, loss, LR schedule, and logging backend
trainer = Trainer(
    model=model,
    loss=CrossEntropy(),
    lr_sched=StepLR(lr=0.01),
    logger=Wandb(project_name="cnn-from-scratch", run_name="cnn-batchnorm")
)

# Start training with eval each epoch
trainer.fit(
    train_data,
    test_data,
    epochs=80,
    batch_size=64,
)