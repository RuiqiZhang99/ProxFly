import numpy as np
import matplotlib.pyplot as plt



def downwashForce(relativePos, radiumSepBound=0.5, zSepBound=0.5, dockSep=0.1):

    centerForce = -6.8
    kForce = -20.8
    maxForce = 3.0

    extraForce = 0.0
    zForceNoise = np.clip(np.random.normal(0, 0.1), -0.5, 0.5)

    xForce = np.clip(np.random.normal(0, 0.2), -1, 1)
    yForce = np.clip(np.random.normal(0, 0.2), -1, 1)

    radiumSep = np.linalg.norm(relativePos[:2])

    if relativePos[2] >= dockSep and relativePos[2] <= zSepBound + dockSep and radiumSep <= radiumSepBound:
        
        extraForce = (1 - (relativePos[2] - dockSep) / zSepBound) * maxForce
        zForce = (extraForce + centerForce) * np.exp(kForce * radiumSep) + zForceNoise
        return np.array([xForce, yForce, min(zForce, maxForce)])

    else:
        return np.array([xForce, yForce, zForceNoise])


def downwashTorque(frac=0.0):
    
    rollTorque = np.random.normal(0, 1e-3)
    pitchTorque = np.random.normal(0, 1e-3)

    if frac <= 1:
        xMagnitude, yMagnitude = np.random.uniform(5e-3, 2e-2), np.random.uniform(5e-3, 2e-2)
        rollTorque = (-1) ** np.random.randint(1, 2) * xMagnitude * np.sin(2 * np.pi * frac) + np.random.normal(0, 1e-3)
        pitchTorque = (-1) ** np.random.randint(1, 2) * yMagnitude * np.sin(2 * np.pi * frac) + np.random.normal(0, 1e-3)

    yawTorque = np.random.normal(0, 5e-3)

    return np.array([rollTorque, pitchTorque, yawTorque])

def externalDisturbance(startTS, currentTS, approSpeed, randHeight, simFreq=500):
    
    radiumSepBound = 0.5
    zSepBound = 0.5
    dockSep = 0.1

    duration = 2 * radiumSepBound / approSpeed # Time: in second (s)
    frac = (currentTS - startTS) / (duration * simFreq)

    fakeRelativePos = np.array([(-radiumSepBound + 2 * radiumSepBound * frac), 0, randHeight])

    externalForce = downwashForce(fakeRelativePos, radiumSepBound, zSepBound, dockSep)
    externalTorque = downwashTorque(frac)


    return externalForce, externalTorque