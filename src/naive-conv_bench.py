"""Benchmark NaiveConv."""
from core.layers import NaiveConv
from core.logging import Wandb
from core.utils import benchmark_layer

input_shape = (3, 32, 32)
output_shape = (16, 32, 32)

logger = Wandb(project_name="cnn-from-scratch", run_name="naive-conv-benchmark")
naive_layer = NaiveConv(3, 16, 3, padding=1)

benchmark_layer(naive_layer,
                step=8,
                max_batch=64,
                input_shape=input_shape,
                output_shape=output_shape,
                logger=logger
                )