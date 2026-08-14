# Copyright IBM Corp. 2026
# License: Apache-2.0

"""Install and load PauliPropagation.jl into the Julia environment juliacall uses.

Required only for QAOA depth p > 1, where ``training/eval_step.py`` evaluates energies
with ``PPEvaluator``, which is backed by the Julia package PauliPropagation.jl
(https://github.com/MSRudolph/PauliPropagation.jl).

Run once after installing the "julia" extra:

    pip install -e ".[julia]"
    python link-julia-python.py

Note that this mutates the active Julia environment by adding a package to it.
"""

from juliacall import Main as jl

jl.seval('using Pkg; Pkg.add("PauliPropagation")')
jl.seval("using PauliPropagation")
print("PauliPropagation.jl installed and loaded — Julia bridge working.")
