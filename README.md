# learning-traffic

Testing
====
To run the tests:

    pushd cistar-dev; python -m unittest discover; popd

To run only the fast (eg. unit) tests:

    pushd cistar-dev; python -m unittest discover tests/fast; popd

To run only the slow (eg. integration) tests:

    pushd cistar-dev; python -m unittest discover tests/slow; popd

Development
====
* For this project, and any derived from it, please run the following command
  from the project root directory:

      ln -s ../../pre-commit.sh .git/hooks/pre-commit