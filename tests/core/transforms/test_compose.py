from core.transforms import Compose

def test_compose():

    transform = Compose([
        lambda x: x + 1,
        lambda x: x * 2,
    ])

    x = 3
    out = transform(x)
    assert out == 8  # (3 + 1) * 2 = 8
    assert len(transform.transforms) == 2