#!/usr/bin/env python3
# flake8: noqa
"""Setup script for installing Flow."""
from os.path import dirname, realpath
from setuptools import find_packages, setup, Distribution
import setuptools.command.build_ext as _build_ext
import subprocess
from flow.version import __version__


def _read_requirements_file():
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


class build_ext(_build_ext.build_ext):
    """See parent class."""

    def run(self):
        """Install dependencies that are not covered by the conda env."""
        try:
            import traci
        except ImportError:
            subprocess.check_call(
                ['pip', 'install',
                 'https://akreidieh.s3.amazonaws.com/sumo/flow-0.2.0/'
                 'sumotools-0.1.0-py3-none-any.whl'])


class BinaryDistribution(Distribution):
    """See parent class."""

    def has_ext_modules(self):
        """See parent class."""
        return True


setup(
    name='flow',
    version=__version__,
    distclass=BinaryDistribution,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(),
    install_requires=_read_requirements_file(),
    zip_safe=False,
)
