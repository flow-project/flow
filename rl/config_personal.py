
import os.path as osp
import os

USE_GPU = False

USE_TF = True

AWS_REGION_NAME = "us-west-1"

if USE_GPU:
    DOCKER_IMAGE = "dementrock/rllab3-shared-gpu"
else:
    DOCKER_IMAGE = "evinitsky/cistar-rllab" # "dementrock/rllab3-shared"

DOCKER_LOG_DIR = "/tmp/expt"

AWS_S3_PATH = "s3://aboudy.traffic/rllab/experiments"

AWS_CODE_SYNC_S3_PATH = "s3://aboudy.traffic/rllab/code"

ALL_REGION_AWS_IMAGE_IDS = {
    "us-west-1": "ami-ad81c8cd",
    "us-west-2": "ami-7ea27a1e",
    "us-east-1": "ami-6b99d57c"
}

AWS_IMAGE_ID = ALL_REGION_AWS_IMAGE_IDS[AWS_REGION_NAME]

if USE_GPU:
    AWS_INSTANCE_TYPE = "g2.2xlarge"
else:
    AWS_INSTANCE_TYPE = "c4.2xlarge"

ALL_REGION_AWS_KEY_NAMES = {
    "us-west-1": "rllab-us-west-1",
    "us-west-2": "rllab-us-west-2",
    "us-east-1": "rllab-us-east-1",
    # "us-west-1": "leah-west-1"
}

AWS_KEY_NAME = ALL_REGION_AWS_KEY_NAMES[AWS_REGION_NAME]

AWS_SPOT = True

AWS_SPOT_PRICE = '0.5	'

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", None)

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", None)

AWS_IAM_INSTANCE_PROFILE_NAME = "rllab"

AWS_SECURITY_GROUPS = ["rllab-sg"]

ALL_REGION_AWS_SECURITY_GROUP_IDS = {
    "us-west-1": [
        "sg-a7fbf4c3"
    ],
    "us-west-2": [
        "sg-07731d7e"
    ],
    "us-east-1": [
        "sg-471eed3a"
    ]
}

AWS_SECURITY_GROUP_IDS = ALL_REGION_AWS_SECURITY_GROUP_IDS[AWS_REGION_NAME]

FAST_CODE_SYNC_IGNORES = [
    ".git",
    "data/local",
    "data/s3",
    "data/video",
    "src",
    ".idea",
    ".pods",
    "tests",
    "examples",
    "docs",
    ".idea",
    ".DS_Store",
    ".ipynb_checkpoints",
    "blackbox",
    "blackbox.zip",
    "*.pyc",
    "*.ipynb",
    "scratch-notebooks",
    "conopt_root",
    "private/key_pairs",
    "BDD_new/"
]

FAST_CODE_SYNC = True


