import cupy as cp

from core import trainer
from core.trainer import Trainer


class FakeDataSet:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]


class FakeModel:
	def __init__(self):
		self.weight = 1.0

	def forward(self, x):
		return x

	def backward(self, grad, lr):
		self.weight -= lr * grad


class FakeLoss:
	def forward(self, pred, target):
		return cp.array(0.5)

	def backward(self, pred, target):
		return cp.array(0.1)
	
class FakeScheduler:
	def __call__(self, epoch):
		return 0.5
	
def test_fit():
    x = cp.array([1.0, 2.0])
    y = cp.array([1.5, 2.5])
    data = FakeDataSet(x, y)

    model = FakeModel()

    trainer = Trainer(model=model,
					  loss=FakeLoss(),
					  lr_sched=FakeScheduler()
					)

    trainer.fit(train_data=data, epochs=1, batch_size=1)

    assert cp.isclose(model.weight, 0.9)
    assert trainer.logs["train_loss"] == 0.5
    assert trainer.logs["epoch"] == 1
    assert trainer.logs["lr"] == 0.5


def test_evaluate():
	x = cp.array([[0.0, 0.0], [0.0, 1.0]])
	y = cp.array([0, 1])
	data = FakeDataSet(x, y)

	trainer = Trainer(model=FakeModel(), loss=FakeLoss(), lr_sched=FakeScheduler())

	avg_loss, acc = trainer.evaluate(data, batch_size=1)

	assert acc == 1.0
	assert cp.isclose(avg_loss, 0.5)
