# Docker documentation
### Leah Dickstein
--


- Follow the instructions at https://rllab.readthedocs.io/en/latest/user/cluster.html
- You will need an access key and secret key
- After you run the script, it's supposed to update rllab/config_personal.py
- You must make a few more changes:
	- `DOCKER_IMAGE = "lahaela/rllab-sumo"`
	- leah.traffic was my new RLLAB S3 Bucket I configured in my bash_profile
	- `AWS_S3_PATH = "s3://leah.traffic/rllab/experiments"`
	- `AWS_CODE_SYNC_S3_PATH = "s3://leah.traffic/rllab/code"`
	- If you have code you've added to your rllab dir, make sure to add the directory to `FAST_CODE_SYNC_IGNORES` ! The training script will launch all relevant files to AWS, and we don't want to upload stuff we don't need (takes longer + memory)
- Discovered bug in `setup_s3()` in `setup_ec2_for_rllab.py`. Bug was solved by adding a Location Constraint to the s3\_client.create\_bucket() call.

```
        s3_client.create_bucket(
            ACL='private',
            Bucket=S3_BUCKET_NAME,
            CreateBucketConfiguration={'LocationConstraint': 'us-west-1'}
        )
```

- I'm a little worried this bug implies we always have to use the West 1 region, so this is something I have to look into.

## Setting up a new Docker

- Take your time reading the new Docker documentation. Follow the getting started guide. Important notes:
- You can only have one ENTRYPOINT and CMD per Dockerfile. This indicates what is run immediately, and this should be rllab. Rllab will call SUMO and set up the I/O connection, so you don't have to instantiate SUMO upon creating the AWS bucket.