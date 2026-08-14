# Copyright IBM Corp. 2026
# License: Apache-2.0

# Import juliacall BEFORE torch to avoid segfault warning
try:
    import juliacall
except ImportError:
    pass  