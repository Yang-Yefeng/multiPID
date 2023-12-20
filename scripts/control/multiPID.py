import numpy as np
from typing import Union


class multiPID:
    def __init__(self):
        self.kp_pos = np.array([1.0, 1.0, 1.0])
        self.ki_pos = np.array([1.0, 1.0, 1.0])
        self.kd_pos = np.array([1.0, 1.0, 1.0])

        self.kp_vel = np.array([1.0, 1.0, 1.0])
        self.ki_vel = np.array([1.0, 1.0, 1.0])
        self.kd_vel = np.array([1.0, 1.0, 1.0])

        self.e_pos = np.zeros(3)
        self.de_pos = np.zeros(3)
        self.ie_pos = np.zeros(3)

        self.e_vel = np.zeros(3)
        self.de_vel = np.zeros(3)
        self.ie_vel = np.zeros(3)

        self.control = np.zeros(3)
        self.n = 0

    def control_update(self,
                       e_pos: Union[np.ndarray, list],
                       e_vel: Union[np.ndarray, list]):
        if self.n == 0:

        else:
        self.de_pos = e_pos -

        self.control =
        self.adjust_pid_param(e_pos, e_vel)

    def adjust_pid_param(self,
                         e_pos: Union[np.ndarray, list],
                         e_vel: Union[np.ndarray, list]):
        pass
