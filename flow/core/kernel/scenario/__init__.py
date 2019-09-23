"""Empty init file to handle deprecations."""

import warnings
from flow.core.kernel.network import *  # noqa: F401,F403

warnings.simplefilter('always', PendingDeprecationWarning)
warnings.warn(
    "flow.core.kernel.scenario will be deprecated in a future release. Please "
    "use flow.core.kernel.network instead.",
    PendingDeprecationWarning
)
