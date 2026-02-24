from core import *
from core.layers import *
from core.transforms import *
from core.lr_scheduler import StepLR
from core.logging import Wandb
from core.datasets import MNIST
from core.losses import MSE

model = Model([
    Flatten(),
    FC(784, 256),
    ReLU(),
    FC(256, 128),
    ReLU(),
    FC(128, 10),
    Softmax()
])

train_transform = Compose([
    Scale(),
    Normalize(mean=MNIST.mean, std=MNIST.std, axis=0),
])

test_transform = Compose([
    Scale(),
    Normalize(mean=MNIST.mean, std=MNIST.std, axis=0),
])

train_data = MNIST(
  root = "data",
  train = True,
  download = True,
  transform = train_transform,
  target_transform = OneHot(num_classes=10)
)

test_data = MNIST(
  root = "data",
  train = False,
  download = True,
  transform = test_transform,
  target_transform = OneHot(num_classes=10)
)

trainer = Trainer(model,
                  loss=MSE(),
                  lr_sched=StepLR(0.01),
                  logger=Wandb(project_name="nn-from-scratch", run_name="mnist-mlp")  
                )

trainer.fit(
    train_data,
    test_data,
    epochs = 6,
    batch_size = 64,
    test_interval = 2
)
