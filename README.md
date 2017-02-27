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
      
Warnings:
====
All car controllers come equipped with a fail-safe rule wherein cars are not allowed to 
move at a speed that would cause them to crash if the car in front of them suddenly started 
breaking with max acceleration. If they attempt to do so, they will be reset to move at $v_safe$ 
where $v_safe$ is the speed such that the cars will come to rest at the same point. 
