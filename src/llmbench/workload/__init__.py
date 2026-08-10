"""Workload definition: when requests arrive, and how big they are.

Everything here is deterministic given a seed and computed *before* a run
starts, so that every configuration under test faces a byte-identical offered
load.
"""
