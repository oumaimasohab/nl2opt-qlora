"""Run mlx_lm.lora and print MLX peak memory at exit (mlx-lm 0.32 does not print it). Same CLI as mlx_lm.lora."""
import sys, atexit
import mlx.core as mx
from mlx_lm import lora
atexit.register(lambda: print(f"MLX_PEAK_GB {mx.get_peak_memory()/1e9:.2f}", flush=True))
sys.argv = ["mlx_lm.lora"] + sys.argv[1:]
lora.main()
