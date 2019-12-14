"""Empty init file to ensure documentation for traffic lights is created."""

from flow.core.kernel.induction_loops.base import KernelLaneAreaDetector
from flow.core.kernel.induction_loops.traci import TraCILaneAreaDetector


__all__ = ["KernelLaneAreaDetector", "TraCILaneAreaDetector"]