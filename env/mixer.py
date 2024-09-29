from __future__ import division, print_function

from env.py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
import numpy as np

class QuadcopterMixer:
    def __init__(self, mass, inertiaMatrix, armLength, thrustToTorque, timeConst_xy, timeConst_z):
        #A simple, nested controller.
        self._mass          = mass
        self._inertiaMatrix = inertiaMatrix
        self._timeConst_xy  = timeConst_xy
        self._timeConst_z   = timeConst_z
        
        #compute mixer matrix:
        l = armLength * (2**0.5)
        k = thrustToTorque
        M = np.array([[ 1,  1,  1,  1],
                      [ -l,  -l,  l, l],
                      [-l,  l,  l,  -l],
                      [ -k, k,  -k, k],
                      ])
        
        self._mixerMat = np.linalg.inv(M)
        return

        
    def get_motor_force_cmd(self, desNormThrust, desAngAcc, estAngVel):
        ftot = self._mass * desNormThrust
        nonlinearCorrect = estAngVel.cross(self._inertiaMatrix*estAngVel)
        moments = self._inertiaMatrix*desAngAcc + nonlinearCorrect
        return self._mixerMat.dot(np.array([ftot, moments.x, moments.y, moments.z]))
    
    
    def get_motor_force_cmd_from_rates(self, desNormThrust, desAngVel, estAngVel):
        angVelError = desAngVel - estAngVel
        desAngAcc = Vec3(
            angVelError.x / self._timeConst_xy,
            angVelError.y / self._timeConst_xy,
            angVelError.z / self._timeConst_z
        )
        motorCmds = self.get_motor_force_cmd(desNormThrust, desAngAcc, estAngVel)
        return motorCmds
        

        
        
