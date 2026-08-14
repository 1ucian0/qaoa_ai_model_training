# Copyright IBM Corp. 2026
# License: Apache-2.0

# Import juliacall BEFORE torch to avoid segfault warning
# This must be at the package level to ensure it's loaded before any submodules
# See: https://github.com/pytorch/pytorch/issues/78829
try:
    import juliacall
except ImportError:
    pass  # juliacall is optional
