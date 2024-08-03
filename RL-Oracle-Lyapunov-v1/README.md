# A Test Oracle for Reinforcement Learning Software based on Lyapunov Stability Control Theory

## Abstract

Reinforcement Learning (RL) has gained significant attention in recent years. As RL software becomes more complex and infiltrates critical application domains, ensuring its quality and correctness becomes increasingly important. An indispensable aspect of software quality/correctness assurance is testing. However, testing RL software faces unique challenges compared to testing traditional software due to the difficulty in defining the outputs’ correctness. This leads to the RL test oracle problem.

Current approaches to testing RL software often rely on human oracles, i.e., convening human experts to judge the correctness of RL software outputs. This heavily depends on the availability and quality (including the experiences, subjective states, etc.) of the human experts and cannot be fully automated. 

In this project, we propose a novel approach to design test oracles for RL software by leveraging the Lyapunov stability control theory. By constructing Lyapunov functions to guide RL training, we hypothesize that correctly implemented RL software shall output an agent that respects Lyapunov stability control theories. Based on this heuristic, we propose a Lyapunov stability control theory-based oracle, the LPEA oracle, for testing RL software. We conduct extensive experiments to evaluate the effectiveness of the LPEA oracle. 

## Getting Started

A demo video without any voice (to avoid revealing any personal information), named demo.mp4, is included in the zip file. You can watch it and follow the order.

### Prerequisites
Make sure you have the following libraries installed:
- Python 3.x
- [Stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [NumPy](https://numpy.org/)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [scipy](https://scipy.org/)
- [cvxpy](https://www.cvxpy.org/)
- [control](https://github.com/python-control/python-control/tree/0.10.0)
- [matplotlib](https://matplotlib.org/)
- [imageio](https://github.com/imageio/imageio/tree/master)
- [openpyxl](https://openpyxl.readthedocs.io/en/stable/)

## Usage

We provide two methods for using our software: running command line scripts directly or using our Jupyter Notebook tutorial for a step-by-step interactive experience.

### Command Line
Make sure you are in the **\RL-Oracle-Lyapunov-v1\RLTestingLyapunov** directory before running the following commands.


1. **Inject a bug into the Stable-baselines library:**
    ```sh
    python bug_injection.py <bug_id>
    ```

    Replace `<bug_id>` with the specific bug id you want to inject.
    - `<bug_id>`: The id of the specific bug you want to inject. There are totally 44 bugs(0-43). Specially, bug_id = -1 means that there is no bug injected into the SB3 library. You can find detailed algorithms' bug document in **\RLTestingLyapunov\Bug library** file.

    Example command(bug-less):
    ```sh
    python bug_injection.py -1
    ```


2. **Run the LPEA Oracle:**

    ```sh
    python LPEA_Oracle.py <bug_id> <algorithm> <n> <m> <I> <J>
    ```

    - `<bug_id>`: The id of the specific bug you want to inject. There are totally 44 bugs(0-43). Specially, bug_id = -1 means that there is no bug injected into the SB3 library. You can find detailed algorithms' bug document in **\RLTestingLyapunov\Bug library** file.
    
    - `<algorithm>`: The algorithm to use (`a2c`, `ppo`, or `td3`). You can find detailed algorithms' bug document in **\RLTestingLyapunov\Bug library** file.
    - `<n>`: The dimension of the state vector (default is 2). See Section II-B Eq. (1) of the paper.
    - `<m>`: The dimension of qcontrol vector (default is 2). See Section II-B Eq. (1) of the paper.
    - `<I>`: The number of agents (also the enviroments) to train (default is 20). See Section III Step 1 of the paper.
    - `<J>`: The number of initial state samples for evaluation (default is 800). See Section III Step 3 - Conjecture 1 of the paper.
    In the demo, it is highly recommanded to set a small I(like 1) to shorten the running time. In our experiment, we set I = 20 to get more stable results.

    Example command(bug-less):
    ```sh
    python LPEA_Oracle.py -1 ppo 2 2 1 800
    ```

### Jupyter Notebook

For a more interactive and detailed walkthrough, you can use our Jupyter Notebook tutorial. This allows you to play with our software step-by-step and understand the process more deeply. To get started:

1. Launch Jupyter Notebook:
    ```sh
    jupyter notebook
    ```

2. Open the provided tutorial notebook and follow the instructions inside.

This approach is ideal for those who prefer a guided, hands-on experience.

### Jupyter Notebook:
Open the Jupyter Notebook file `LPEA_Oracle.ipynb` in the `RLTestingLyapunov` folder.

You should run the code block in the following order: 
**Step 1 -> Step 2 -> Restart kernel -> Step 1 -> Step 3 ->Step 4 -> Step 5 -> Step 6**
1. Choose and inject the bug:

    The algorithms and bugs you choose must correspond to each other, if you need more information, please check bug document in **\RLTestingLyapunov\Bug library** file. Specially, if bug_no = -1, it means inject no bugs into SB3 library.
    ```python
    bug_no = -1
    algorithm = 'ppo'
    n = 2
    m = 2
    I = 1
    J = 800
    ```

2. Inject bug into Stable-baselines3 library

    **Important**: Restart the kernal after injecting bug to make sure the running stable-baselines3 is injected successfully. After kernal restarts, you don't have to run this block again.

    ```python
    import bug_lib
    if bug_no == -1:
        bug_lib.cover_then_inject_bugs([])
    else:
        bug_lib.cover_then_inject_bugs([bug_no])
    ```

3. Import relevant libraries and buggy SB3

    **Important:** Run the Step 1 code block to redefine bug_no, algorithm, I, J, n and m.
    ```python
    import Util
    import LPEA_Env
    import numpy as np
    from gymnasium.wrappers import TimeLimit 
    import stable_baselines3 as sb3
    import Lyaponov_oracle_util as LO
    import os
    ```

4. Generate state transition matrices. 

    In our experiment, we set I = 1, n = 2, m = 2. This implies that we randomly generate 1 pair of 2 by 2 matrices A and B. This pair of A and B will be used to create 1 environment. See **Section II-B Lyapunov Stability Control Theory** for more information.
    ```python
    file_path = f'./saved_array/{n}by{n}'.format(n=n, m=m)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    for i in range(I):
        A, B = Util.generate_state_transition_matix(n, m)
        np.save(f'./saved_array/{n}by{n}/array_A_{i}.npy'.format(n=n, m=m), A)
        np.save(f'./saved_array/{n}by{m}/array_B_{i}.npy'.format(n=n, m=m), B)
    ```

5. Train **I** agents and save all the trained agents into a file for further useage  

    This step may take some time, depending on your device and the algorithm you choose. Noramlly, a2c and ppo can run quickly but td3 is slow. If I = 20, a2c and ppo need 45 min, td3 needs 2 hour to run.

    ```python
    file_path_log = './trained_models/oracle_{alg}/bug_{bug}/{n}by{m}/'.format(n=n, m=m, bug = bug_no, alg=algorithm)
    random_seed = 1

    for i in range(I):
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
    ```

6. Evaluation the trained agents.
    ```python
    buggy_metrics = LO.buggy_trained_model_metrics_calculation(algorithm, n, m, I, J, bug_no)
    for vartheta in range(100, 40, -10):
        for theta in range(100, 45, -25):
            Oracle_result = LO.LPEA_Oracle(buggy_metrics, I, J, vartheta * 0.01, theta * 0.01)
            if Oracle_result:
                print("vartheta={vartheta}%, theta={theta}%, the software is bug-less based on LPEA Oracle".format(vartheta=vartheta, theta=theta))
            else:
                print("vartheta={vartheta}%, theta={theta}%, the software is buggy based on LPEA Oracle".format(vartheta=vartheta, theta=theta))
    ```


## Results
This is an example of the output. You could check Section III for more details or vartheta and theta.

    vartheta=100%, theta=100%, the software is bug-less based on LPEA Oracle
    vartheta=100%, theta=75%, the software is bug-less based on LPEA Oracle
    vartheta=100%, theta=50%, the software is bug-less based on LPEA Oracle
    vartheta=90%, theta=100%, the software is bug-less based on LPEA Oracle
    vartheta=90%, theta=75%, the software is bug-less based on LPEA Oracle
    vartheta=90%, theta=50%, the software is bug-less based on LPEA Oracle
    vartheta=80%, theta=100%, the software is bug-less based on LPEA Oracle
    vartheta=80%, theta=75%, the software is bug-less based on LPEA Oracle
    vartheta=80%, theta=50%, the software is bug-less based on LPEA Oracle
    vartheta=70%, theta=100%, the software is bug-less based on LPEA Oracle
    vartheta=70%, theta=75%, the software is bug-less based on LPEA Oracle
    vartheta=70%, theta=50%, the software is bug-less based on LPEA Oracle
    vartheta=60%, theta=100%, the software is bug-less based on LPEA Oracle
    vartheta=60%, theta=75%, the software is bug-less based on LPEA Oracle
    vartheta=60%, theta=50%, the software is bug-less based on LPEA Oracle
    vartheta=50%, theta=100%, the software is bug-less based on LPEA Oracle
    vartheta=50%, theta=75%, the software is bug-less based on LPEA Oracle
    vartheta=50%, theta=50%, the software is bug-less based on LPEA Oracle
