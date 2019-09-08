"""Empty init file to ensure documentation for traffic lights is created."""

from flow.core.kernel.traffic_light.base import KernelTrafficLight
from flow.core.kernel.traffic_light.sumo import SumoTrafficLight
from flow.core.kernel.traffic_light.aimsun import AimsunKernelTrafficLight


__all__ = ["KernelTrafficLight", "SumoTrafficLight",
           "AimsunKernelTrafficLight"]
