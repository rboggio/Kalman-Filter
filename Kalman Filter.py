import numpy as np

class KalmanFilter:

    """

    Kalman Filter estimate the state of a linear dynamical system.

    Methods:
        predict:
            Predicts the next state of the system using the current state estimate, 
            state-transition model, control input, and process noise covariance.

        update:
            Updates the state estimate with a new measurement using the observation model, 
            measurement noise covariance, and Kalman Gain.

    :param F: Define the transition matrix
    :param H: Define the observation matrix
    :param B: Specify the control matrix
    :param Q: the covariance of the process noise
    :param R: Set the covariance of the observation noise
    :param P: Set the initial value of the covariance matrix
    :param theta_0: Set the initial state of the filter
    
    """

    def __init__(self, F: np.array, H: np.array, B: np.array = None, Q: np.array = None, R: np.array = None, P: np.array = None, theta_0: np.array = None):

        self.n = F.shape[1]
        self.m = H.shape[1]
        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.theta = np.zeros((self.n, 1)) if theta_0 is None else theta_0

    def predict(self, u=0):
        """
        The predict function is used to predict the next state of the system.
        It takes in an optional control input u, which defaults to 0 if not specified.
        The function returns a predicted state x.

        :param u: Add control to the system
        :return: The predicted state
        """
        # Predicted state estimate
        self.theta = (self.F @ self.theta) + (self.B * u)
        # Predicted estimate covariance
        self.P = (self.F @ self.P) @ self.F.T + self.Q

        return self.theta

    def update(self, z):
        """
        The update function takes in a measurement and updates the state estimate.
        The update function is called after the predict function, when a new measurement (z) has been received.
        :param z: add the measurements
        """
        innovation = z - (self.H @ self.theta)
        # Kalman Gain
        K = (self.P @ self.H.T) @ np.linalg.inv(((self.H @ (self.P @ self.H.T)) + self.R))
        # Updated state estimate
        self.theta = self.theta + (K @ innovation)
        I = np.eye(self.n)
        # Updated estimate covariance
        self.P = (I - (K @ self.H)) @ self.P