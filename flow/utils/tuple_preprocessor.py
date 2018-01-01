"""
Custom preprocessor needed for Tuple observation spaces within rllib.
"""

import numpy as np

from ray.rllib.models.preprocessors import Preprocessor

class TuplePreprocessor(Preprocessor):

    def _init(self):
        self.shape = self._obs_space.shape

    def transform(self, observation):
        return np.concatenate(observation)

