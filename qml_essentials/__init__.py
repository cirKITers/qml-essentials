import jax

# 64-bit precision is NOT enabled by default — this matches JAX's own
# default (complex64 / float32) and halves memory usage, which is critical
# for density-matrix simulation at 13+ qubits.
#
# Users who need double precision (complex128 / float64) can opt in via:
#   jax.config.update("jax_enable_x64", True)   # before any computation
# or by setting the environment variable:
#   JAX_ENABLE_X64=1
