from bayes_opt import BayesianOptimization
from SPM import SPM
import numpy as np

def charging_time_compute(current1, charging_number, current2):
    env = SPM(3.0, 298)
    done = False
    i = 0
    while not done:
        if i < int(charging_number):
            current = current1
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        else:
            current = current2
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))

        _, done, _ = env.step(current)
        i += 1
        if env.voltage > env.sett['constraints voltage max'] or env.temp > env.sett['constraints temperature max']:
            i += 1

        if done:
            return -i

pbounds = {"current1": (3, 6), "charging_number": (5, 25), "current2": (1, 4)}

env = SPM()
# print(env.param)
# optimizer = BayesianOptimization(
#     f=charging_time_compute,
#     pbounds=pbounds,
#     random_state=1,
# )

# optimizer.maximize(
#     init_points=5,
#     n_iter=30,
# )
print(charging_time_compute(5, 10, 3))