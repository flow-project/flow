"""Empty init file to ensure documentation for lane area detectors is created."""

from flow.core.kernel.lane_area_detectors.base import KernelLaneAreaDetector
from flow.core.kernel.lane_area_detectors.traci import TraCILaneAreaDetector


__all__ = ["KernelLaneAreaDetector", "TraCILaneAreaDetector"]
