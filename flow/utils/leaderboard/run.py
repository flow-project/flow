"""Runner for flow/utils/leaderboard/evaluate.py/evaluate_policy."""

from solution import BENCHMARK, get_actions, get_states
from evaluate import evaluate_policy

# Evaluate the solution
mean, stdev = evaluate_policy(
    benchmark=BENCHMARK, _get_actions=get_actions, _get_states=get_states)
# Print results
print(mean, stdev)
