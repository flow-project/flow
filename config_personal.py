import os.path as osp
import os

USE_GPU = False

USE_TF = True

AWS_REGION_NAME = "us-west-1"

if USE_GPU:
    DOCKER_IMAGE = "dementrock/rllab3-shared-gpu"
else:
    DOCKER_IMAGE = "dementrock/rllab3-shared"

DOCKER_LOG_DIR = "/tmp/expt"

AWS_S3_PATH = "s3://leah.traffic/rllab/experiments"

AWS_CODE_SYNC_S3_PATH = "s3://leah.traffic/rllab/code"

ALL_REGION_AWS_IMAGE_IDS = {
    "us-west-1": "ami-bbd19fdb",
    "us-west-2": "ami-92eb35f2",
    "us-east-1": "ami-472b6a50"
}

AWS_IMAGE_ID = ALL_REGION_AWS_IMAGE_IDS[AWS_REGION_NAME]

if USE_GPU:
    AWS_INSTANCE_TYPE = "g2.2xlarge"
else:
    AWS_INSTANCE_TYPE = "c4.2xlarge"

ALL_REGION_AWS_KEY_NAMES = {
    # "us-east-1": "rllab-us-east-1",
    # "us-west-2": "djf-us-west-2",
    "us-west-1": "leah-west-1"
}

# ALL_REGION_AWS_KEY_NAMES = $all_region_aws_key_names

AWS_KEY_NAME = ALL_REGION_AWS_KEY_NAMES[AWS_REGION_NAME]

AWS_SPOT = True

AWS_SPOT_PRICE = '0.5'

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", "AKIAIDGQA2DBXZQ7DQ5Q")

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", "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")

AWS_IAM_INSTANCE_PROFILE_NAME = "rllab"

AWS_SECURITY_GROUPS = ["rllab-sg"]

ALL_REGION_AWS_SECURITY_GROUP_IDS = {
    "us-east-1": [
        "sg-d9b28da3"
    ],
    "us-west-2": [
        "sg-dc38b8a5"
    ],
    "us-west-1": [
        "sg-3e94ab5a"
    ]
}

# ALL_REGION_AWS_SECURITY_GROUP_IDS = $all_region_aws_security_group_ids

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
]

FAST_CODE_SYNC = True

# DOCKER_IMAGE = "rein/rllab-exp-new"

# KUBE_PREFIX = "template_"

# AWS_IMAGE_ID = "ami-67c5d00d"

# AWS_KEY_NAME = "research_virginia"

# # LOCAL_CODE_DIR = "<insert local code dir>"

# LABEL = "template"

# DOCKER_CODE_DIR = "/root/code/rllab"
