import math

from core.utils.lr_scheduler import ConstantLR, CosineAnnealingLR, StepLR, TrainState


def test_constant_lr():
    sched = ConstantLR()
    sched.setup(base_lr=0.01, epochs=2, steps_per_epoch=3)

    s0 = TrainState(epoch=0, batch=0, global_step=0, epochs=2, steps_per_epoch=3)
    s5 = TrainState(epoch=1, batch=2, global_step=5, epochs=2, steps_per_epoch=3)

    assert sched.lr(s0) == 0.01
    assert sched.lr(s5) == 0.01


def test_step_lr_epoch_based():
    sched = StepLR(step_size=2, gamma=0.5)
    sched.setup(base_lr=0.01, epochs=10, steps_per_epoch=4)

    s0 = TrainState(epoch=0, batch=0, global_step=0, epochs=10, steps_per_epoch=4)
    s1 = TrainState(epoch=1, batch=0, global_step=4, epochs=10, steps_per_epoch=4)
    s2 = TrainState(epoch=2, batch=0, global_step=8, epochs=10, steps_per_epoch=4)
    s4 = TrainState(epoch=4, batch=0, global_step=16, epochs=10, steps_per_epoch=4)

    assert sched.lr(s0) == 0.01
    assert sched.lr(s1) == 0.01
    assert sched.lr(s2) == 0.005
    assert sched.lr(s4) == 0.0025


def test_cosine_annealing_endpoints():
    sched = CosineAnnealingLR(min_lr=1e-4)
    sched.setup(base_lr=1e-2, epochs=2, steps_per_epoch=5)  # total_steps=10

    s0 = TrainState(epoch=0, batch=0, global_step=0, epochs=2, steps_per_epoch=5)
    s_end = TrainState(epoch=1, batch=4, global_step=10, epochs=2, steps_per_epoch=5)

    assert abs(sched.lr(s0) - 1e-2) < 1e-12
    assert abs(sched.lr(s_end) - 1e-4) < 1e-12


def test_cosine_annealing_monotonic_decrease_initially():
    sched = CosineAnnealingLR(min_lr=0.0)
    sched.setup(base_lr=1.0, epochs=1, steps_per_epoch=10)

    lrs = [
        sched.lr(TrainState(epoch=0, batch=i, global_step=i, epochs=1, steps_per_epoch=10))
        for i in range(0, 6)
    ]

    # First half should be strictly decreasing for this setup.
    assert all(lrs[i] > lrs[i + 1] for i in range(len(lrs) - 1))
    assert math.isclose(lrs[0], 1.0, rel_tol=0, abs_tol=1e-12)
