import stable_baselines3 as sb3
import Util
import numpy as np
import LPEA_Env
from gymnasium.wrappers import TimeLimit 

def calculate_cosine_similarity(vector1, vector2):
    vector1 = vector1.flatten()
    vector2 = vector2.flatten()
    dot_product = np.dot(vector1, vector2) 
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    similarity = dot_product / (norm1 * norm2)
    return similarity

def generate_point_with_min_distance(n):
    while True:
        center = np.random.rand(n)
        if np.all(center >= 0.01):
            break
    return center

def calculate_action_similarity_and_distance_metrics_for_model(env, model, n, d, num_cases):
    min_cos = []
    max_dis = []
    max_mag = []
    directions = []
    
    for _ in range(num_cases):
        cos_similarities = []
        vector_distances = []
        magnitudes = []
        center = generate_point_with_min_distance(d, n).reshape((n, 1))
        around_points = Util.generate_points_around(center, d)

        center_action = model.predict(center, deterministic=True)[0]

        #potential calculation
        X_plus_one = (env.A @ center + env.B @ center_action)
        X_plus_one_potential = X_plus_one.T @ env.P @ X_plus_one
        X_potential = center.T @ env.P @ center
        potential_diff = X_plus_one_potential.flatten() - X_potential.flatten()
        directions.append(np.all(potential_diff < 0))

        for around_point in around_points:
            around_point = around_point.reshape((n, 1))
            #action-related calculation
            action = model.predict(around_point, deterministic=True)[0]
            cosine_similarity = calculate_cosine_similarity(action, center_action)
            vector_distance = np.linalg.norm(center_action - action)
            magnitude = np.abs(np.linalg.norm(center_action) - np.linalg.norm(action))

            cos_similarities.append(cosine_similarity)
            vector_distances.append(vector_distance)
            magnitudes.append(magnitude)

        min_cos_similarity = np.min(cos_similarities)
        max_vector_distance = np.max(vector_distances)
        max_magnitude = np.max(magnitudes)

        min_cos.append(min_cos_similarity)
        max_dis.append(max_vector_distance)
        max_mag.append(max_magnitude)
    
    return min_cos, max_dis, directions, max_mag


def calculate_action_similarity_and_distance_metrics_for_Acl(env, n, d, num_cases):
    min_cos = []
    max_dis = []
    directions = []
    max_mag = []
    
    for _ in range(num_cases):
        cos_similarities = []
        vector_distances = []
        magnitudes = []
        center = generate_point_with_min_distance(d, n).reshape((n, 1))
        around_points = Util.generate_points_around(center, d)

        center_action = -env.K @ center

        #potential calculation
        X_plus_one = env.Acl @ center
        X_plus_one_potential = X_plus_one.T @ env.P @ X_plus_one
        X_potential = center.T @ env.P @ center
        potential_diff = X_plus_one_potential.flatten() - X_potential.flatten()
        directions.append(np.all(potential_diff < 0))

        for around_point in around_points:
            around_point = around_point.reshape((n, 1))
            #action-related calculation
            action = -env.K @ around_point
            cosine_similarity = calculate_cosine_similarity(action, center_action)
            vector_distance = np.linalg.norm(center_action - action)
            magnitude = np.abs(np.linalg.norm(center_action) - np.linalg.norm(action))

            cos_similarities.append(cosine_similarity)
            vector_distances.append(vector_distance)
            magnitudes.append(magnitude)

        min_cos_similarity = np.min(cos_similarities)
        max_vector_distance = np.max(vector_distances)
        max_magnitude = np.max(magnitudes)

        min_cos.append(min_cos_similarity)
        max_dis.append(max_vector_distance)
        max_mag.append(max_magnitude)
    
    return min_cos, max_dis, directions, max_mag


def calculate_state_similarity_and_distance_metrics_for_Acl(env, n, d, num_cases):
    min_cos = []
    max_norm_dis = []
    directions = []
    max_norm_mag = []
    
    for _ in range(num_cases):
        cos_similarities = []
        norm_vector_distances = []
        norm_magnitudes = []
        center = generate_point_with_min_distance(d, n).reshape((n, 1))
        around_points = Util.generate_points_around(center, d)

        center_next_state = env.Acl @ center
        center_drift = center_next_state - center
        center_drift_norm = np.linalg.norm(center_drift)

        #potential calculation
        X_plus_one = env.Acl @ center
        X_plus_one_potential = X_plus_one.T @ env.P @ X_plus_one
        X_potential = center.T @ env.P @ center
        potential_diff = X_plus_one_potential.flatten() - X_potential.flatten()
        directions.append(np.all(potential_diff < 0))

        for around_point in around_points:
            around_point = around_point.reshape((n, 1))
            #action-related calculation
            around_point_next_state = env.Acl @ around_point
            around_point_drift = around_point_next_state - around_point

            cosine_similarity = calculate_cosine_similarity(around_point_drift, center_drift)
            vector_distance = np.linalg.norm(around_point_drift - center_drift)
            norm_vector_distance = vector_distance/center_drift_norm
            magnitude = np.abs(np.linalg.norm(around_point_drift) - np.linalg.norm(center_drift))
            norm_magnitude = magnitude/center_drift_norm

            cos_similarities.append(cosine_similarity)
            norm_vector_distances.append(norm_vector_distance)
            norm_magnitudes.append(norm_magnitude)

        min_cos_similarity = np.min(cos_similarities)
        max_norm_vector_distance = np.max(norm_vector_distances)
        max_norm_magnitude = np.max(max_norm_vector_distance)

        min_cos.append(min_cos_similarity)
        max_norm_dis.append(max_norm_vector_distance)
        max_norm_mag.append(max_norm_magnitude)
    
    return min_cos, max_norm_dis, directions, max_norm_mag

def calculate_state_similarity_and_distance_metrics_for_model(env, model, n, num_cases):
    directions = []
    
    for _ in range(num_cases):
        center = generate_point_with_min_distance(n).reshape((n, 1))
        center_action = model.predict(center, deterministic=True)[0]

        #potential calculation
        X_plus_one = (env.A @ center + env.B @ center_action)
        X_plus_one_potential = X_plus_one.T @ env.P @ X_plus_one
        X_potential = center.T @ env.P @ center
        potential_diff = X_plus_one_potential.flatten() - X_potential.flatten()
        directions.append(np.all(potential_diff < 0))

    return directions

def buggy_trained_model_metrics_calculation(algorithm, n, m, I, num_points, bug_no, random_seed=1):
    buggy_similarity_distance_metrics = []
    np.random.seed(random_seed)

    for i in range(I):
        file_path_A = './saved_array/{n}by{m}/array_A_{i}.npy'.format(n=n, m=m, i=i)
        file_path_B = './saved_array/{n}by{m}/array_B_{i}.npy'.format(n=n, m=m, i=i)
        loaded_A = np.load(file_path_A)
        loaded_B = np.load(file_path_B)
        env = TimeLimit(LPEA_Env.CustomEnv(loaded_A, loaded_B, n, m), max_episode_steps=1)

        #buggy model
        model_path = './trained_models/oracle_{algorithm}/bug_{bug}/{n}by{m}/{i}_model'.format(n=n, m=m, i=i, bug=bug_no, algorithm=algorithm)
        if algorithm == 'a2c':
            buggy_model = sb3.A2C.load(model_path, env)
        elif algorithm == 'ppo':
            buggy_model = sb3.PPO.load(model_path, env)
        elif algorithm == 'td3':
            buggy_model = sb3.TD3.load(model_path, env)
        buggy_similarity_distance_metrics.append(calculate_state_similarity_and_distance_metrics_for_model(env, buggy_model, n, num_points))
    return buggy_similarity_distance_metrics

def LPEA_Oracle(buggy_metrics, I, J, vartheta, theta):
    counter = 0
    for i in range(I):
        count = np.sum(buggy_metrics[i])
        if count/J >= vartheta:
            counter += 1    
    return counter/I >= theta