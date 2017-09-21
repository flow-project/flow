#!/usr/bin/env bash

# pre-commit.sh

git stash -q --keep-index

# Test prospective commit
pushd cistar_dev
python -m unittest discover tests/fast
RESULT=$?
popd

git stash pop -q

[ $RESULT -ne 0 ] && exit 1
exit 0
