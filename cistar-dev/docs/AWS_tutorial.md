# Setting up AWS


  - Followed instructions at http://rllab.readthedocs.io/en/latest/user/cluster.html
  - If prompted, region = us-west-1
  - Before running `scripts/setup_ec2_for_rllab.py` , modify by adding such that:
    try:
            s3_client.create_bucket(
                ACL='private',
                Bucket=S3_BUCKET_NAME,
                CreateBucketConfiguration={'LocationConstraint': 'us-west-1'}
            )
  - In `config_personal.py` change DOCKER_IMAGE to `DOCKER_IMAGE = "lahaela/cistar-rllab"` 
  - Also in `config_personal.py` ,if your rllab directory isn’t clean you may want to add certain folders to `FAST_CODE_SYNC_IGNORES` such as `"BDD_new/"` 
  - Go to the `Makefile` in `learning-traffic/cistar-dev` and update the path to your rllab root directory (no trailing slash)
  - (See note below); Run `make prepare` 
  - Try an example! Run any experiment from `cistar/examples` , change mode to “ec2”
  - Log into AWS via: https://cathywu.signin.aws.amazon.com/console
  - If you don’t see the instance you just launched (give it a few minutes) running on AWS, then make sure your region is correct (go to top right and change to `US West (N. California)` 


- When we run experiments on AWS, we create a new instance for each seed and use the Docker image I created as the VM. Built into the rllab script for running experiments in EC2 mode, we upload the rllab root directory to AWS. This way, the AWS instance has access to all files it might need to successfully run the experiment. Editing that code is pretty complicated, so Cathy and I have decided on the following workflow:
  - All code modification will happen in the learning-traffic directory
  - Before each experiment, run the command `make prepare` , which will remove the cistar directory in rllab root and copy `learning-traffic/cistar-dev/cistar` into your rllab root directory
    - This means if you make modifications to cistar in the rllab directory, they may be lost
  - Before each experiment, always make sure you have a commit to that exact snapshot of the `cistar` directory. This is because you may modify cistar later. When you want to run `visualizer.py` , which is our modified version of `sim_policy.py` , AKA when you want to create rollout plots, you need the files in `rllab/cistar` to match the files that were there when you originally ran the experiment. So when you want to create rollout plots, you will checkout the commit that matches when you ran the experiment and run make prepare, then you can create rollout plots in the rllab directory.
  - Ping me if there are issues!
- I recommend cleaning up your rllab directory, especially if you notice you’re uploading a large size to AWS (it will tell you how many MB you are uploading, should be < 5 MB). The command `make clean` removes the debug directory (since so far we hardcoded that our SUMO files go into that directory, this should be changed in the future) and also all XML files in rllab root directory.