from __future__ import print_function, division
from env.py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
import numpy as np

class PositionController:
    """A simple position controller, with acceleration as output. 
    
    Makes position behave as a second order system. 
    """
    
    def __init__(self, natFreq, dampingRatio):
        self._natFreq = natFreq
        self._dampingRatio = dampingRatio
        
        
    def get_acceleration_command(self, desPos, curPos, curVel):

        desAcc = -2*self._dampingRatio*self._natFreq*curVel - self._natFreq**2*(curPos - desPos)
        return desAcc
    
    def get_thrust_command(self, desPos, curPos, curVel, curAtt, resCmd=0.0):
        
        desAcc = self.get_acceleration_command(desPos, curPos, curVel)
        desProperAcc = desAcc + Vec3(0, 0, 9.81)

        normProperAcc = desProperAcc.norm2() 
        directProperAcc = desProperAcc / normProperAcc

        thrustCmd = normProperAcc * (curAtt.to_rotation_matrix() * Vec3(0, 0, 1)).dot(directProperAcc)
        totalthrustCmd = thrustCmd + resCmd
        return desProperAcc, thrustCmd, totalthrustCmd
        
