"""Default config variables, which may be overridden by a user config."""
import os.path as osp
import os

PYTHON_COMMAND = "python"

SUMO_SLEEP = 1.0  # Delay between initializing SUMO and connecting with TraCI

PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))

LOG_DIR = PROJECT_PATH + "/data"

# users set both of these in their bash_rc or bash_profile
# and also should run aws configure after installing awscli
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", None)

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", None)

AWS_S3_PATH = "s3://bucket_name"

# path to the Aimsun_Next main directory (required for Aimsun simulations)
AIMSUN_NEXT_PATH = "/home/user/Aimsun_Next_XXX"

# path to the aimsun_flow environment's main directory (required for Aimsun
# simulations)
AIMSUN_SITEPACKAGES = "/path/to/envs/aimsun_flow"
