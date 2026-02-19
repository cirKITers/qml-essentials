import jax

# Enable 64-bit precision globally — must happen before any JAX computation.
# Centralised here so every submodule inherits the setting automatically.
jax.config.update("jax_enable_x64", True)
