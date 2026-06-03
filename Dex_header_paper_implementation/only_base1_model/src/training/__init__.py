"""Training loop, checkpointing, and evaluation (Phases 5–6).

Import submodules directly (e.g. ``from src.training.train import run_training``).
Do not import train/evaluate here: ``python -m src.training.train`` loads this
package first; eager imports register ``src.training.train`` in sys.modules before
runpy executes it, which triggers RuntimeWarning and spams logs from DataLoader workers.
"""
