import sys
from evaluate import evaluate_policy
PATH = sys.argv[1]
sys.path.append(PATH)
from solution import BENCHMARK, get_actions, get_states

# Evaluate the solution
mean, stdev = evaluate_policy(benchmark=BENCHMARK,
                              _get_actions=get_actions,
                              _get_states=get_states)
# Print results
print(mean, stdev)

