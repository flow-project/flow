"""Empty init file to handle deprecations."""

import warnings
from flow.networks import *  # noqa: F401,F403

warnings.simplefilter('always', PendingDeprecationWarning)
warnings.warn(
    "flow.scenarios will be deprecated in a future release. Please use "
    "flow.networks instead.",
    PendingDeprecationWarning
)
