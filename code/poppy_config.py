# Config file for Poppy robot in VREP
# just enough to interface with it and make legs move

from abr_control.arms.robot_config import robot_config


class robot_config(robot_config):
    """ Robot config file for the Poppy"""

    def __init__(self, hand_attached=False, **kwargs):

        num_links = 3
        super(robot_config, self).__init__(num_joints=3, num_links=num_links,
                                           robot_name='poppy', **kwargs)
        print(self.num_joints)

        self.joint_names = ['l_ankle_y', 'r_ankle_y',
                            'l_knee_y', 'r_knee_y',
                            'l_hip_y', 'r_hip_y']
