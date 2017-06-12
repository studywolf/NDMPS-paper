# Config file for Poppy robot in VREP
# just enough to interface with it and make legs move

from abr_control.arms.base_config import BaseConfig


class Config(BaseConfig):
    """ Robot config file for the Poppy"""

    def __init__(self, **kwargs):

        super(Config, self).__init__(
            N_JOINTS=3, N_LINKS=3, ROBOT_NAME='poppy', **kwargs)
        self.JOINT_NAMES = ['l_ankle_y', 'r_ankle_y',
                            'l_knee_y', 'r_knee_y',
                            'l_hip_y', 'r_hip_y']
