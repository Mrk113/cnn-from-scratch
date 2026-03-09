from core import *
from core.layers import *
from core.transforms import *
from core.lr_scheduler import StepLR
from core.logging import Wandb
from core.datasets import MNIST
from core.losses import MSE

# Simple MLP for MNIST
model = Model([
    Flatten(),
    FC(784, 256),
    ReLU(),
    FC(256, 128),
    ReLU(),
    FC(128, 10),
    Softmax()
])

# Resize and normalize grayscale inputs
train_transform = Compose([
    Scale(),
    Normalize(mean=MNIST.mean, std=MNIST.std, axis=0),
])

# Same preprocessing for eval
test_transform = Compose([
    Scale(),
    Normalize(mean=MNIST.mean, std=MNIST.std, axis=0),
])

# Load MNIST with our transforms and one-hot encode targets
train_data = MNIST(
    root = "data",
    train = True,
    download = True,
    transform = train_transform,
    target_transform = OneHot(num_classes=10)
)

# Eval dataset with test transforms and one-hot targets
test_data = MNIST(
    root = "data",
    train = False,
    download = True,
    transform = test_transform,
    target_transform = OneHot(num_classes=10)
)

# Trainer combines model, loss, LR schedule, and logging
trainer = Trainer(model,
                  loss=MSE(),
                  lr_sched=StepLR(0.01),
                  logger=Wandb(project_name="nn-from-scratch", run_name="mlp-mnist")  
                )

# Train for a few epochs with periodic evaluation
trainer.fit(
    train_data,
    test_data,
    epochs = 6,
    batch_size = 64,
    test_interval = 2
)
