import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy import linalg as la, optimize
import cvxpy as cp
import control

class CustomEnv(gym.Env):
    def __init__(self, A, B, n, m, random_seed=1):
        super(CustomEnv, self).__init__()
        np.random.seed(random_seed)
        self.n = n
        self.m = m
        self.Q = np.eye(n)
        self.R = np.eye(m)
        self.X_desired = np.zeros(n)

        self.stable_counter = 0
        self.stable_counter_threshold = 7
        self.step_counter = 0

        self.state_distance_boundary = 2
        self.out_of_boundary_punish = 100

        self.A = A
        self.B = B
        self.K, self.P = self.findPK(self.A, self.B, self.Q, self.R)
        self.Acl = self.A - self.B @ self.K
        self.action_inf = abs(self.K).sum(axis=1).max()
        
        self.action_space = spaces.Box(low=-20, high=20, shape=(m,1), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(n,1), dtype=np.float32)

        self.reward_distance_threshold = 0.2

    def set_model(self, model):
        self.model = model

    def findPK(self, A, B, Q, R):
        K = control.dlqr(A, B, Q, R)[0]
        Acl = A - B @ K
        P = la.solve_discrete_lyapunov(Acl.T, Q)
        is_positive_definite = np.all(np.linalg.eigvals(P) > 0)
        if is_positive_definite:
            # print("System can be stable")
            return K, P
        else:
            print("System can't be stable")
            return None, None
    
    def step(self, action):
        self.step_counter += 1
        X_k_plus_1 = (self.A @ self.state + self.B @ action).astype(np.float32)

        terminated = False

        reward = self.calculateRewardPotentialMultiStep()
        self.state = X_k_plus_1

        distance = np.linalg.norm(self.state - self.X_desired)

        if distance >= self.state_distance_boundary:
                terminated = True
                reward -= self.out_of_boundary_punish

        info = {}
        return self.state, float(reward), terminated, False, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.stable_counter = 0
        self.step_counter = 0
        self.state = np.random.uniform(-1, 1, size=(self.m, 1)).astype(np.float32)
        return self.state, {}
    
    def calculateRewardPotentialMultiStep(self):
        distance = np.linalg.norm(self.state - self.X_desired)
        reward = 0

        X_k_potential = self.state.T @ self.P @ self.state
        potential_reward = np.exp(-1 * X_k_potential)
        
        convergence_reward = np.exp(-1 * distance)
        reward = potential_reward + convergence_reward * 0.7
        
        return reward
