import sys
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.optimizers.replay_buffer import PrioritizedReplayBuffer
@DeveloperAPI
class PrioritizedReplayBufferWithExperts(PrioritizedReplayBuffer):
    @DeveloperAPI
    def __init__(self, size, alpha, reserved_frac):
        """The reserved_frac sets which portion of the buffer stores reserved data.
           This data is never removed or modified after we are done populating it with expert data.
        """
        super(PrioritizedReplayBufferWithExperts, self).__init__(size, alpha)
        # we only add data to the buffer above this index
        self.reserved_idx = int(size * reserved_frac)
        self._next_reserved_idx = 0
        # next_reserved_idx goes from 0 to reserved_idx, next_idx goes from reserved_idx to max_size
        self._next_idx = self.reserved_idx

    @DeveloperAPI
    def add(self, obs_t, action, reward, obs_tp1, done, weight):
        """This adds to the """

        idx = self._next_idx
        data = (obs_t, action, reward, obs_tp1, done)
        self._num_added += 1

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
            self._est_size_bytes += sum(sys.getsizeof(d) for d in data)
        else:
            self._storage[self._next_idx] = data
        if self._next_idx + 1 >= self._maxsize:
            self._eviction_started = True
        self._next_idx = (self._next_idx + 1) % self._maxsize
        if self._next_idx == 0:
            self._next_idx = self.reserved_idx
        if self._eviction_started:
            self._evicted_hit_stats.push(self._hit_count[self._next_idx])
            self._hit_count[self._next_idx] = 0

        if weight is None:
            weight = self._max_priority
        self._it_sum[idx] = weight**self._alpha
        self._it_min[idx] = weight**self._alpha

    @DeveloperAPI
    def add_to_reserved(self, obs_t, action, reward, obs_tp1, done, weight):
        """See ReplayBuffer.store_effect"""

        idx = self._next_reserved_idx
        data = (obs_t, action, reward, obs_tp1, done)
        self._num_added += 1

        if self._next_reserved_idx >= len(self._storage):
            self._storage.append(data)
            self._est_size_bytes += sum(sys.getsizeof(d) for d in data)
        else:
            self._storage[self._next_reserved_idx] = data
        self._next_reserved_idx = (self._next_reserved_idx + 1) % self.reserved_idx

        if weight is None:
            weight = self._max_priority
        self._it_sum[idx] = weight**self._alpha
        self._it_min[idx] = weight**self._alpha