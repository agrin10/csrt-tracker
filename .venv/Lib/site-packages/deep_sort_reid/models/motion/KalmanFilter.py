

from typing import Tuple

from torch import Tensor
import torch
from deep_sort_reid.constants.tracker import AR_POS_STD_DEFAULT, AR_PROJ_STD_DEFAULT, AR_VEL_STD_DEFAULT, MEASUREMENT_SPACE_DIM, POS_WEIGHT_DEFAULT, POS_WEIGHT_FACTOR_INIT, STATE_MEAN_HEIGHT_IDX, VEL_WEIGHT_DEFAULT, VEL_WEIGHT_FACTOR_INIT
from deep_sort_reid.types.coords import CoordinatesXYAH

from deep_sort_reid.utils.box_methods import from_xyah_to_tensor


class KalmanFilter():

    def __init__(self):
        dim = MEASUREMENT_SPACE_DIM
        dt = 1

        # Projects mean and cov to measurement space, discarding velocity information
        self.projection_matrix = torch.eye(dim, 2*dim)

        # Transforms mean and cov using velocity information
        self.transformation_matrix = torch.eye(2 * dim, 2 * dim)
        for i in range(dim):
            self.transformation_matrix[i, dim + i] = dt

    def initiate(self, detection_coords: CoordinatesXYAH,
                 pos_weight=POS_WEIGHT_DEFAULT,
                 vel_weight=VEL_WEIGHT_DEFAULT) -> Tuple[Tensor, Tensor]:

        assert isinstance(detection_coords, CoordinatesXYAH), Exception(
            "detection_coords invalid type")

        measurement = from_xyah_to_tensor(detection_coords)

        state_mean_pos = measurement
        state_mean_vel = torch.zeros_like(state_mean_pos)
        state_mean = torch.cat((state_mean_pos, state_mean_vel))

        pos_weight *= POS_WEIGHT_FACTOR_INIT
        vel_weight *= VEL_WEIGHT_FACTOR_INIT
        state_std_pos = Tensor([
            pos_weight * detection_coords.height,
            pos_weight * detection_coords.height,
            AR_POS_STD_DEFAULT,
            pos_weight * detection_coords.height
        ])
        state_std_vel = Tensor([
            vel_weight * detection_coords.height,
            vel_weight * detection_coords.height,
            AR_VEL_STD_DEFAULT,
            vel_weight * detection_coords.height
        ])

        state_std = torch.cat((state_std_pos, state_std_vel))
        state_covariance = torch.diag(torch.square(state_std))

        return (state_mean, state_covariance)

    def predict(self, state_mean: Tensor, state_covariance: Tensor,
                pos_weight=POS_WEIGHT_DEFAULT,
                vel_weight=VEL_WEIGHT_DEFAULT):

        dim = MEASUREMENT_SPACE_DIM
        assert state_mean.shape[0] == (
            2*dim), "Incorrect state_mean shape"
        assert state_covariance.shape == (
            2*dim, 2*dim), "Incorrect state_covariance shape"

        """
        We introduce noise proportional to the height of the `state_mean`, 
        similarly to how the filter was initialized. 

        This noise accounts for increased uncertainty in the position and velocity 
        based on object size (height), and it helps adjust the predicted state covariance matrix.

        Higher uncertainty is modeled for larger objects, which affects the filter update step.
        """
        state_mean_height = state_mean[STATE_MEAN_HEIGHT_IDX]
        state_std_pos_noise = Tensor([
            pos_weight * state_mean_height,
            pos_weight * state_mean_height,
            AR_POS_STD_DEFAULT,
            pos_weight * state_mean_height
        ])
        state_std_vel_noise = Tensor([
            vel_weight * state_mean_height,
            vel_weight * state_mean_height,
            AR_VEL_STD_DEFAULT,
            vel_weight * state_mean_height
        ])

        state_std_noise = torch.cat(
            (state_std_pos_noise, state_std_vel_noise))
        state_covariance_noise = torch.diag(torch.square(state_std_noise))

        predicted_state_mean = self.transformation_matrix @ state_mean

        predicted_state_covariance = (self.transformation_matrix
                                      @ state_covariance
                                      @ self.transformation_matrix.T) + state_covariance_noise

        return (predicted_state_mean, predicted_state_covariance)

    def __project(self, state_mean, state_covariance, pos_weight=POS_WEIGHT_DEFAULT):

        state_mean_height = state_mean[STATE_MEAN_HEIGHT_IDX]
        measurement_std_noise = Tensor([
            pos_weight * state_mean_height,
            pos_weight * state_mean_height,
            AR_PROJ_STD_DEFAULT,
            pos_weight * state_mean_height
        ])

        measurement_covariance_noise = torch.diag(
            torch.square(measurement_std_noise))

        # Project the state mean to the measurement space
        measurement_mean = self.projection_matrix @ state_mean

        # Project the state covariance to the measurement space, adding further noise
        measurement_covariance = (self.projection_matrix
                                  @ state_covariance
                                  @ self.projection_matrix.T) + measurement_covariance_noise

        return measurement_mean, measurement_covariance

    def update(self, state_mean: Tensor, state_covariance: Tensor,
               detection_coords: CoordinatesXYAH,
               pos_weight=POS_WEIGHT_DEFAULT,
               vel_weight=VEL_WEIGHT_DEFAULT):

        measurement_mean, measurement_covariance = self.__project(
            state_mean, state_covariance, pos_weight)

        """
        Calculation of Kalman Gain

        Let 
        H = projection_matrix
        P = state_covariance
        R = measurement_covariance_noise
        K = kalman_gain
        A = L L.T = (H P H.T + R)
        B = P H.T
    
        Then
        A K = B

        And 
        K = A^-1 B = (L.T)^-1 (L)^-1 B
        """
        L = torch.linalg.cholesky(measurement_covariance)
        B = state_covariance @ self.projection_matrix.T
        K = torch.cholesky_solve(B.T, L).T

        measurement = from_xyah_to_tensor(detection_coords)
        residuals = measurement - measurement_mean

        """
            Updating the predictions and uncertainties using the Kalman Gain. 
            To simplify, a low K would mean higher trust in the prediction,
            while a high K would mean higher trust in the measurement.
        """

        new_mean = state_mean + (residuals@K.T)
        new_covariance = state_covariance - (K@measurement_covariance@K.T)

        return new_mean, new_covariance

    def gating_distance(self, state_mean, state_covariance, measurements_tensor: Tensor):
        """
        Calculates the Mahalanobis distance, 
        which can be thought off as residuals normalized 
        using the state uncertainty (covariance), in turn yielding
        a better measurement as a distance between the state mean and the measurements
        """
        measurement_mean, measurement_covariance = self.__project(
            state_mean, state_covariance)

        """
        Solves for K in
        LK = B
        """
        B = measurements_tensor - measurement_mean
        L = torch.linalg.cholesky(measurement_covariance)
        K = torch.linalg.solve_triangular(L, B.T, upper=False)

        squared_maha = torch.sum(K * K, dim=0)
        return squared_maha
