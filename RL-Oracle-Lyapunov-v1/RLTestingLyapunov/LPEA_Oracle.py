import Util
import LPEA_Env
import numpy as np
from gymnasium.wrappers import TimeLimit 
import stable_baselines3 as sb3
import Lyaponov_oracle_util as LO
import os
import sys

bug_no = int(sys.argv[1])
algorithm = sys.argv[2]
n = int(sys.argv[3])
m = int(sys.argv[4])
I = int(sys.argv[5])
J = int(sys.argv[6])

#Generate A, B matrix
file_path = f'./saved_array/{n}by{n}'.format(n=n, m=m)
if not os.path.exists(file_path):
    os.makedirs(file_path)

for i in range(I):
    A, B = Util.generate_state_transition_matix(n, m)
    np.save(f'./saved_array/{n}by{n}/array_A_{i}.npy'.format(n=n, m=m), A)
    np.save(f'./saved_array/{n}by{m}/array_B_{i}.npy'.format(n=n, m=m), B)

file_path_log = './trained_models/oracle_{alg}/bug_{bug}/{n}by{m}/'.format(n=n, m=m, bug = bug_no, alg=algorithm)
random_seed = 1

for i in range(I):
    print("Training: ", i, "th agent")
    file_path_A = 'saved_array/{n}by{m}/array_A_{i}.npy'.format(n=n, m=m, i=i)
    file_path_B = 'saved_array/{n}by{m}/array_B_{i}.npy'.format(n=n, m=m, i=i)
    loaded_A = np.load(file_path_A)
    loaded_B = np.load(file_path_B)
    env = TimeLimit(LPEA_Env.CustomEnv(loaded_A, loaded_B, n, m), max_episode_steps=50)
    if algorithm == 'ppo':
        model = sb3.PPO("MlpPolicy", env, verbose=0, seed=random_seed, learning_rate=0.0012)
        model.learn(total_timesteps=120000)
    elif algorithm == 'a2c':
        model = sb3.A2C("MlpPolicy", env, verbose=0, seed=random_seed, learning_rate=0.0004)
        model.learn(total_timesteps=90000)
    elif algorithm == 'td3':
        model = sb3.TD3("MlpPolicy", env, verbose=0, seed=random_seed, )
        model.learn(total_timesteps=90000)
    model.save('./trained_models/oracle_{alg}/bug_{bug}/{n}by{m}/{i}_model'.format(n=n, m=m, i=i, bug=bug_no, alg=algorithm))

buggy_metrics = LO.buggy_trained_model_metrics_calculation(algorithm, n, m, I, J, bug_no)

for vartheta in range(100, 40, -10):
    for theta in range(100, 45, -25):
        Oracle_result = LO.LPEA_Oracle(buggy_metrics, I, J, vartheta * 0.01, theta * 0.01)
        if Oracle_result:
            print("vartheta={vartheta}%, theta={theta}%, the software is bug-less based on LPEA Oracle".format(vartheta=vartheta, theta=theta))
        else:
            print("vartheta={vartheta}%, theta={theta}%, the software is buggy based on LPEA Oracle".format(vartheta=vartheta, theta=theta))

