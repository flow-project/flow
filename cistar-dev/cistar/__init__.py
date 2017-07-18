import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)


register(
    id='SumoEnv-v0',
    entry_point='cistar.core:SumoEnvironment',
)
register(
    id='SimpleIntersectionEnv-v0',
    entry_point='cistar.envs:SimpleIntersectionEnvironment',
)
register(
    id='LoopEnv-v0',
    entry_point='cistar.envs:LoopEnvironment',
)
register(
    id='TwoIntersectionEnv-v0',
    entry_point='cistar.envs:TwoIntersectionEnvironment',
)