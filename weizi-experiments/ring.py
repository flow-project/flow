import flow.scenarios as scenarios

scenario_name = "LoopScenario"

from flow.core.params import NetParams, InitialConfig

name = "weizi_ring_example"

from flow.scenarios.loop import ADDITIONAL_NET_PARAMS

net_params = NetParams(additional_params = ADDITIONAL_NET_PARAMS)

initial_config = InitialConfig(spacing="uniform", perturbation=1)


