from datetime import datetime
import numpy as np
import pandas as pd
import pykalman
from joblib import Parallel, delayed
from itertools import combinations

__all__ = ["anymazeResults", "dlcResults", "BehavioralDataManager"]


class anymazeResults:
    def __init__(self, filepath: str):
        self.anymaze_df = pd.read_csv(filepath)
        self.freeze_vector = None

    def create_freeze_vector(self, timestamps, time_format='%H:%M:%S.%f', time_col='Time', behavior_col='Freezing'):
        binary_vector = np.zeros(len(timestamps), dtype=int)
        for i, ts in enumerate(timestamps):
            state = self.anymaze_df.loc[pd.to_datetime(self.anymaze_df[time_col], format=time_format) <= ts, behavior_col].iloc[-1]
            binary_vector[i] = state
        self.freeze_vector = binary_vector
        return pd.DataFrame(binary_vector)


class dlcResults:
    def __init__(self, dlc_file):
        self.dlc_df = pd.read_csv(dlc_file, header=[1, 2], index_col=[0])
        self.filtered_df = None

    def kalman_filter(self, x_data, y_data, dt=1):
        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        B = np.array([
            [0.5 * (dt ** 2), 0],
            [0, 0.5 * (dt ** 2)],
            [dt, 0],
            [0, dt]
        ])
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        R = np.eye(2) * [[np.var(x_data), np.var(y_data)]]
        Q = np.array([
            [(dt ** 4) / 4, 0, (dt ** 3) / 2, 0],
            [0, (dt ** 4) / 4, 0, (dt ** 3) / 2],
            [(dt ** 3) / 2, 0, dt ** 2, 0],
            [0, (dt ** 3) / 2, 0, dt ** 2]
        ])

        filter = pykalman.KalmanFilter(transition_matrices=A, observation_matrices=H, transition_covariance=Q,
                                       observation_covariance=R, initial_state_covariance=np.eye(4) * 0.1,
                                       initial_state_mean=(x_data[0], y_data[0], 0, 0))
        kalman_means = np.zeros((A.shape[0], len(x_data))).T
        kalman_covs = np.zeros((A.shape[0], A.shape[0], len(x_data))).T
        kalman_accel = np.zeros((2, len(x_data))).T
        kalman_means[0, :] = (x_data[0], y_data[0], 0, 0)
        kalman_covs[0, :, :] = 0.1
        kalman_accel[0, :] = 0

        for t in range(1, len(x_data)):
            kalman_means[t, :], kalman_covs[t, :, :] = filter.filter_update(filtered_state_mean=kalman_means[t - 1, :],
                                                                            observation=[x_data[t], y_data[t]],
                                                                            filtered_state_covariance=kalman_covs[t - 1,
                                                                                                      :, :],
                                                                            transition_offset=B @ kalman_accel[t - 1,
                                                                                                  :].T)
            kalman_accel[t, 0] = (kalman_means[t, 2] - kalman_means[t - 1, 2])
            kalman_accel[t, 1] = (kalman_means[t, 3] - kalman_means[t - 1, 3])

        return kalman_means, kalman_covs, kalman_accel

    def filter_predictions(self, bparts=None, fps=None):
        if bparts is None:
            bparts = ['snout', 'l_ear', 'r_ear', 'front_l_paw', 'front_r_paw', 'back_l_paw', 'back_r_paw',
                      'base_of_tail', 'tip_of_tail']
        if fps is None:
            fps = 30
        dt = 1 / fps

        kalman_dict = {}
        for bpart in bparts:
            k_means, _, k_accel = self.kalman_filter(self.dlc_df.loc[:, (bpart, 'x')], self.dlc_df.loc[:, (bpart, 'y')],
                                                     dt=dt)
            kalman_dict[bpart] = {
                'x': k_means[:, 0],
                'y': k_means[:, 1],
                'velocity_x': k_means[:, 2],
                'velocity_y': k_means[:, 3],
                'acceleration_x': k_accel[:, 0],
                'acceleration_y': k_accel[:, 1]
            }

        reformed_dict = {}
        for outerKey, innerDict in kalman_dict.items():
            for innerKey, values in innerDict.items():
                reformed_dict[(outerKey, innerKey)] = values

        df = pd.DataFrame.from_dict(reformed_dict)
        self.filtered_df = df
        return df


class BehavioralDataManager:
    def __init__(self, anymaze_filepaths: list, dlc_filepaths: list):
        if len(anymaze_filepaths) != len(dlc_filepaths):
            raise ValueError("The number of anymaze files must match the number of DLC files.")

        self.anymaze_filepaths = anymaze_filepaths
        self.dlc_filepaths = dlc_filepaths
        self.features = None
        self.labels = None

    def _process_single_data_pair(self, anymaze_filepath, dlc_filepath):
        anymaze = anymazeResults(anymaze_filepath)
        dlc = dlcResults(dlc_filepath)
        df = dlc.filter_predictions()

        times = pd.to_datetime(np.arange(dlc.dlc_df.shape[0]) * (1 / 30), unit="s")

        labels = anymaze.create_freeze_vector(times)

        return df, labels

    def process_data(self):
        num_cores = -1  # Use all available cores
        results = Parallel(n_jobs=num_cores)(delayed(self._process_single_data_pair)(am_fp, dlc_fp)
                                             for am_fp, dlc_fp in zip(self.anymaze_filepaths, self.dlc_filepaths))

        # Unpack results
        features_list, labels_list = zip(*results)

        # Combine results
        self.features = pd.concat(features_list, ignore_index=True)
        self.labels = pd.concat(labels_list, ignore_index=True)

        return self.features, self.labels

    def calculate_movement(self, df, bpart):
        df[(bpart, 'movement')] = np.sqrt((df[(bpart, 'x')].diff())**2 + (df[(bpart, 'y')].diff())**2)
        return df

    def calculate_distance(self, df, bpart1, bpart2):
        df[(f'{bpart1}_{bpart2}', 'distance')] = np.sqrt((df[(bpart1, 'x')] - df[(bpart2, 'x')])**2 + 
                                                         (df[(bpart1, 'y')] - df[(bpart2, 'y')])**2)
        return df

    def calculate_angle(self, bpart1, bpart2, bpart3):
        df = self.features
        vector_1 = np.array([df[(bpart1, 'x')] - df[(bpart2, 'x')], df[(bpart1, 'y')] - df[(bpart2, 'y')]])
        vector_2 = np.array([df[(bpart3, 'x')] - df[(bpart2, 'x')], df[(bpart3, 'y')] - df[(bpart2, 'y')]])

        # normalize the vectors
        vector_1 /= np.linalg.norm(vector_1, axis=0)
        vector_2 /= np.linalg.norm(vector_2, axis=0)
    
        # compute the dot product element-wise
        dot_product = np.sum(vector_1 * vector_2, axis=0)
    
        # clip the values to avoid getting NaN from arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)

        angle = np.arccos(dot_product)

        df[(f'angle_{bpart1}_{bpart2}_{bpart3}', 'angle')] = np.degrees(angle)
        return df

    def feature_engineering(self):
        bparts = ['snout', 'l_ear', 'r_ear', 'front_l_paw', 'front_r_paw', 'back_l_paw', 'back_r_paw', 
                  'base_of_tail', 'tip_of_tail']

        for bpart in bparts:
            self.features = self.calculate_movement(self.features, bpart)

        for pair in combinations(bparts, 2):
            self.features = self.calculate_distance(self.features, *pair)

        for triplet in combinations(bparts, 3):
            self.features = self.calculate_angle(*triplet)

        return self.features
