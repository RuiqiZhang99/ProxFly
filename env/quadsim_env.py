from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from env.py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
from env.vehicle import Vehicle
from env.positioncontroller import PositionController
from env.attitudecontroller import QuadcopterAttitudeControllerNested
from env.mixer import QuadcopterMixer
from env.pyplot3d.utils import ypr_to_R
import gym
from gym import spaces
import pandas as pd
from scipy.spatial.transform import Rotation as R
import random
import yaml

np.random.seed(0)

class QuadSimEnv(gym.Env):
    """Base class for "drone aviary" Gym environments."""

    def __init__(self,
                 init_pos = np.array([0, 0, 0]),
                 hover_pos = np.array([0, 0, 1.2]),
                 land_pos = np.array([0, 0, 0]),
                 dt = 0.002,
                 norm_obs = True,
                 residual_rl = True,
                 mb_control = False,
                 output_folder='results',
                 end_time = 20.0,
                 takeoff_time = 5.0,
                 hover_time = 10.0,
                 land_time = 5.0,
                 thrust_scale = 10.0,
                 rates_scale = 1.0,
                 ):
        assert end_time == (takeoff_time + hover_time + land_time)
        assert not (residual_rl and mb_control)
        #========================= Constants ==========================
        self.g = 9.81
        self.Rad2Deg = 180 / np.pi
        self.Deg2Rad = np.pi / 180
        self.dt = dt
        self.sim_freq = 500
        self.ctrl_freq = 50
        self.max_timestep = int(end_time/self.dt)

        with open('./env/large_quad.yaml', 'r') as file:
            self.model_data = yaml.safe_load(file)

        #======================== Flags ==========================
        self.norm_obs = norm_obs
        self.residual_rl = residual_rl
        self.mb_control = mb_control
        
        #======================== Properties ==========================
        self.output_folder = output_folder
        self.init_pos = init_pos
        self.hover_pos = hover_pos
        self.land_pos = land_pos

        self.thrust_scale = thrust_scale
        self.rates_scale = rates_scale

        self.cmdPos = np.array([0, 0, 0])
        self.cmdVelo = np.array([0, 0, 0])

        #=========== Create action and observation spaces ===============
        self.action_space = self.actionSpace()
        self.observation_space = self.observationSpace()
        
        #========================= For Planner ==========================
        self.takeoff_time = takeoff_time
        self.hover_time = hover_time
        self.land_time = land_time

        #===================== Reset the environment =====================
        self.reset()
    
    #===================================================================================

    def reset(self):
        """Resets the environment."""

        self.quadcopter_initialize()
        
        initial_obs = self.computeObs()
        
        return initial_obs
    
    #===================================================================================
    def quadcopter_initialize(self):

        # --- Vehicle Parameters ---
        mass = self.model_data['Mass']

        # Inertia
        inertia = self.model_data['Inertia']
        Ixx = inertia['Ixx']
        Iyy = inertia['Iyy']
        Izz = inertia['Izz']
        inertiaMatrix = np.matrix([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])

        omegaSqrToDragTorque = np.matrix(np.diag([0, 0, 0.00014]))
        armLength = self.model_data['ArmLength']
        motorPos = armLength * (2**0.5)

        # --- Motors Parameters ---
        motSpeedSqrToThrust = self.model_data['MotSpeedSqrToThrust']
        motSpeedSqrToTorque = self.model_data['MotSpeedSqrToTorque']
        motInertia = self.model_data['MotInertia']
        motTimeConst = self.model_data['MotTimeConst']
        motMinSpeed = self.model_data['MotMinSpeed']
        motMaxSpeed = self.model_data['MotMaxSpeed']

        # --- Domain Randomization ---
        randMassFactor = np.random.uniform(self.model_data['MassRandom'][0], self.model_data['MassRandom'][1])
        randInertiaFactor_xy = randMassFactor * np.random.uniform(self.model_data['InertiaRandom'][0], self.model_data['InertiaRandom'][1])
        randInertiaFactor_z = randMassFactor * np.random.uniform(self.model_data['InertiaRandom'][0], self.model_data['InertiaRandom'][1])
        randMass = mass*randMassFactor
        randIxx, randIyy = Ixx*randInertiaFactor_xy, Iyy*randInertiaFactor_xy
        randIzz = Izz*randInertiaFactor_z
        randInertiaMatrix = np.matrix([[randIxx, 0, 0], [0, randIyy, 0], [0, 0, randIzz]])

        # --- Disturbance Parameters ---
        stdDevTorqueDisturbance = self.model_data['StdDevTorqueDisturbance']

        # ---High-level Controller Parameters ---
        timeConstRatesRP = self.model_data['TimeConstRatesRP']
        timeConstRatesY = self.model_data['TimeConstRatesY']
        timeConstAngleRP = self.model_data['TimeConstAngleRP']
        timeConstAngleY = self.model_data['TimeConstAngleY']
        posCtrlNatFreq = self.model_data['PosCtrlNatFreq']
        posCtrlDampingRatio = self.model_data['PosCtrlDampingRatio']

        self.quadcopter = Vehicle(randMass, randInertiaMatrix, omegaSqrToDragTorque, stdDevTorqueDisturbance)
        self.quadcopter.add_motor(Vec3( motorPos, -motorPos, 0), Vec3(0,0,1), motMinSpeed, motMaxSpeed, 
                                  motSpeedSqrToThrust*np.random.uniform(self.model_data['ThrustRandom'][0], self.model_data['ThrustRandom'][1]), motSpeedSqrToTorque, motTimeConst, motInertia)
        self.quadcopter.add_motor(Vec3(-motorPos, -motorPos, 0), Vec3(0,0,-1), motMinSpeed, motMaxSpeed, 
                                  motSpeedSqrToThrust*np.random.uniform(self.model_data['ThrustRandom'][0], self.model_data['ThrustRandom'][1]), motSpeedSqrToTorque, motTimeConst, motInertia)
        self.quadcopter.add_motor(Vec3(-motorPos,  motorPos, 0), Vec3(0,0,1), motMinSpeed, motMaxSpeed, 
                                  motSpeedSqrToThrust*np.random.uniform(self.model_data['ThrustRandom'][0], self.model_data['ThrustRandom'][1]), motSpeedSqrToTorque, motTimeConst, motInertia)
        self.quadcopter.add_motor(Vec3( motorPos,  motorPos, 0), Vec3(0,0,-1), motMinSpeed, motMaxSpeed, 
                                  motSpeedSqrToThrust*np.random.uniform(self.model_data['ThrustRandom'][0], self.model_data['ThrustRandom'][1]), motSpeedSqrToTorque, motTimeConst, motInertia)

        self.posControl = PositionController(posCtrlNatFreq, posCtrlDampingRatio)
        self.attController = QuadcopterAttitudeControllerNested(timeConstAngleRP, timeConstAngleY, timeConstRatesRP, timeConstRatesY)
        self.lowLevelController = QuadcopterMixer(mass, inertiaMatrix, armLength, motSpeedSqrToTorque/motSpeedSqrToThrust, timeConstRatesRP, timeConstRatesY)


        # --- Initialize randomly ---
        self.quadcopter.set_position(Vec3(np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0))
        self.quadcopter.set_velocity(Vec3(0, 0, 0))
        self.quadcopter.set_attitude(Rotation.identity())
        self.quadcopter._omega = Vec3(0,0,0)


        # --- Control Commands ---
        self.resNormThrustCmd = 0.0
        self.resRatesCmd = np.array([0, 0, 0])
        self.casNormThrustCmd = 0.0
        self.casRatesCmd = np.array([0, 0, 0])

        self.totalThrustCmd = self.resNormThrustCmd + self.casNormThrustCmd
        self.totalRatesCmd = self.resRatesCmd + self.casRatesCmd
        self.lastThrustCmd = 0.0
        self.lastRatesCmd = np.array([0, 0, 0])
    #===================================================================================

    def planner(self, ep_len):
        
        if ep_len == 0:
            self.init_pos = self.quadcopter._pos.to_array().flatten()
        
        if (ep_len * self.dt) <= self.takeoff_time:
            frac = (ep_len * self.dt) / self.takeoff_time
            cmdPos = frac*self.hover_pos + (1-frac)*self.init_pos
            cmdVelo = (self.hover_pos - self.init_pos) / self.takeoff_time
            # cmdPos = self.hover_pos
            # cmdVelo = np.array([0, 0, 0])
        
        elif (ep_len * self.dt) <= (self.takeoff_time + self.hover_time):
            frac = (ep_len * self.dt - self.takeoff_time) / self.hover_time
            cmdPos = self.hover_pos
            cmdVelo = np.array([0, 0, 0])

        else:
            frac = (ep_len * self.dt - self.takeoff_time - self.hover_time) / self.land_time
            cmdPos = frac*self.land_pos + (1-frac)*self.hover_pos
            cmdVelo = (self.land_pos - self.hover_pos) / self.land_time
        return cmdPos, cmdVelo
            

    #===================================================================================

    def step(self,
             nn_action,
             ep_len):
        
        if ep_len % int(self.sim_freq / self.ctrl_freq) == 0:

            self.cmdPos, self.cmdVelo = self.planner(ep_len)

            if self.residual_rl:
                self.resNormThrustCmd = nn_action[0] * self.thrust_scale
                self.resRatesCmd = nn_action[1:] * self.rates_scale
                desAcc, self.casNormThrustCmd, self.totalThrustCmd = self.posControl.get_thrust_command(Vec3(self.cmdPos), self.quadcopter._pos, self.quadcopter._vel, self.quadcopter._att, self.resNormThrustCmd)
                self.casRatesCmd = self.attController.get_angular_velocity(desAcc, self.quadcopter._att).to_array().flatten()
                self.totalRatesCmd = self.casRatesCmd + self.resRatesCmd
                self.motorCmds = self.lowLevelController.get_motor_force_cmd_from_rates(self.totalThrustCmd, Vec3(self.totalRatesCmd), self.quadcopter._omega)
            elif self.mb_control:
                mb_desAcc, _, mb_thrust = self.posControl.get_thrust_command(Vec3(self.cmdPos), self.quadcopter._pos, self.quadcopter._vel, self.quadcopter._att)
                mb_casRatesCmd = self.attController.get_angular_velocity(mb_desAcc, self.quadcopter._att)
                self.motorCmds = self.lowLevelController.get_motor_force_cmd_from_rates(mb_thrust, mb_casRatesCmd, self.quadcopter._omega)
            else:
                self.motorCmds = self.lowLevelController.get_motor_force_cmd_from_rates(nn_action[0]*self.thrust_scale, Vec3(nn_action[1:]*self.rates_scale))

        self.quadcopter.run(self.dt, self.motorCmds)

        
        # Prepare the return values
        obs = self.computeObs()
        reward = self.computeReward()
        truncated = self.computeTruncated()
        
        self.lastThrustCmd = self.totalThrustCmd
        self.lastRatesCmd = self.totalRatesCmd


        return obs, reward, truncated

    #===================================================================================
    
    def actionSpace(self):
        act_size = 4
        act_lower_bound = np.array(-1 * np.ones(act_size))
        act_upper_bound = np.array(+1 * np.ones(act_size))
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    #===================================================================================

    def observationSpace(self):
        
        obs_size = 20
        if self.norm_obs:
            obs_lower_bound = np.array(-1.0 * np.ones(obs_size))
            obs_upper_bound = np.array(+1.0 * np.ones(obs_size))
        else:
            obs_lower_bound = np.array(-100 * np.ones(obs_size))
            obs_upper_bound = np.array(+100 * np.ones(obs_size))
            
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
    
    
    def computeObs(self):
        
        if self.norm_obs:
            obsScaling = 10.0
            obs = np.clip(
                np.hstack((
                self.quadcopter._pos.to_array().flatten() / obsScaling,
                self.quadcopter._att.to_euler_YPR() / obsScaling,
                self.quadcopter._vel.to_array().flatten() / obsScaling,
                self.quadcopter._omega.to_array().flatten() / obsScaling,
                self.resNormThrustCmd / obsScaling,
                self.resRatesCmd / obsScaling,
                (self.casNormThrustCmd - self.g) / obsScaling,
                self.casRatesCmd / obsScaling,
                )), -1, +1)
            
        else:
            # use raw obs
            obs = np.hstack((
                self.cmdPos - self.quadcopter._pos.to_array().flatten(),
                np.array([0, 0, 0])- self.quadcopter._att.to_euler_YPR(),
                self.quadcopter._vel.to_array().flatten(),
                self.quadcopter._omega.to_array().flatten(),
                self.resNormThrustCmd,
                self.resRatesCmd,
                self.casNormThrustCmd,
                self.casRatesCmd,
                ))
        # print('OBS: {}'.format(obs))
        return obs


    #===================================================================================

    def computeReward(self, survival_reward=0.1):
        reward = 0

        posError = np.linalg.norm(self.cmdPos - self.quadcopter._pos.to_array().flatten())
        attError = 3 - np.trace(np.eye(3) @ self.quadcopter._att.to_rotation_matrix())

        thrustCmdOsc = np.abs(self.lastThrustCmd - self.totalThrustCmd)
        ratesCmdOsc = np.linalg.norm(self.lastRatesCmd - self.totalRatesCmd)

        thrustCmdPenalty = np.abs(self.totalThrustCmd) + 2 * thrustCmdOsc
        ratesCmdPenalty = np.linalg.norm(self.totalRatesCmd) + 2 * ratesCmdOsc

        reward += survival_reward\
            - posError\
            - attError\
            - 0.01 * thrustCmdPenalty\
            - 0.1 * ratesCmdPenalty\
            
        return reward
    #===================================================================================

    def computeTruncated(self):
        
        if abs(self.quadcopter._pos.x) > 3 or abs(self.quadcopter._pos.y) > 3 or abs(self.quadcopter._pos.z) > 3:
            print('QUAD RESET: Flying out the safe area!')
            return True
        if abs(self.quadcopter._att.to_euler_YPR()[1]) > 90 * self.Deg2Rad or abs(self.quadcopter._att.to_euler_YPR()[2]) > 90 * self.Deg2Rad:
            print('QUAD RESET: Upside Down!')
            return True
        else:
            return False
        # return False