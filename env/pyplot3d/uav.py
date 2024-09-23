from .basic import Sphere, Line, Arrow, Disk

import numpy as np


class Uav:
    '''
    Draws a quadrotor at a given position, with a given attitude.
    '''

    def __init__(self, ax, arm_length,tilt_angle=0.0):
        '''
        Initialize the quadrotr plotting parameters.

        Params:
            ax: (matplotlib axis) the axis where the sphere should be drawn
            arm_length: (float) length of the quadrotor arm

        Returns:
            None
        '''

        self.ax = ax
        self.arm_length = arm_length

        self.b1 = np.array([1.0, 0.0, 0.0]).T
        self.b2 = np.array([0.0, 1.0, 0.0]).T
        self.b3 = np.array([0.0, 0.0, 1.0]).T

        # Center of the quadrotor
        self.body = Sphere(self.ax, 0.08, 'y')

        # Each motor
        self.motorTiltR = np.array([
        [np.cos(tilt_angle), 0, np.sin(tilt_angle)],
        [0, 1, 0],
        [-np.sin(tilt_angle), 0, np.cos(tilt_angle)]])
        self.motor1 = Disk(self.ax, r=0.1, h=0, c=(1, 0, 0, 0.7), R=self.motorTiltR)
        self.motor2 = Disk(self.ax, r=0.1, h=0, c=(0, 1, 0, 0.7), R=self.motorTiltR)
        self.motor3 = Disk(self.ax, r=0.1, h=0, c=(1, 0, 0, 0.7), R=self.motorTiltR)
        self.motor4 = Disk(self.ax, r=0.1, h=0, c=(0, 1, 0, 0.7), R=self.motorTiltR)

        # Arrows for the each body axis
        self.arrow_b1 = Arrow(ax, self.b1,  (1, 0, 0, 0.5))
        self.arrow_b2 = Arrow(ax, self.b2,  (0, 1, 0, 0.5))
        self.arrow_b3 = Arrow(ax, self.b3,  (0, 0, 1, 0.5))

        # Quadrotor arms
        self.arm_b1 = Line(ax)
        self.arm_b2 = Line(ax)
    

    def draw_at(self, x=np.array([0.0, 0.0, 0.0]).T, R=np.eye(3)):
        '''
        Draw the quadrotor at a given position, with a given direction

        Args:
            x: (3x1 numpy.ndarray) position of the center of the quadrotor, 
                default = [0.0, 0.0, 0.0]
            R: (3x3 numpy.ndarray) attitude of the quadrotor in SO(3)
                default = eye(3)
        
        Returns:
            None
        '''

        # First, clear the axis of all the previous plots
        self.ax.clear()

        # Center of the quadrotor
        # self.body.draw_at(x)


        # Each motor
        motor_pos = self.arm_length*(2**0.5)
        self.motor1.draw_at(x + R.dot(np.array([motor_pos, -motor_pos, 0]).T) * self.arm_length, R=R)
        self.motor2.draw_at(x + R.dot(np.array([-motor_pos, -motor_pos, 0]).T) * self.arm_length, R=R)
        self.motor3.draw_at(x + R.dot(np.array([-motor_pos, +motor_pos, 0]).T) * self.arm_length, R=R)
        self.motor4.draw_at(x + R.dot(np.array([motor_pos,motor_pos, 0]).T) * self.arm_length, R=R)
   

        # Arrows for the each body axis
        self.arrow_b1.draw_from_to(x, (self.b1).dot(R.T) * self.arm_length )
        self.arrow_b2.draw_from_to(x, (self.b2).dot(R.T) * self.arm_length )
        self.arrow_b3.draw_from_to(x, (self.b3).dot(R.T) * self.arm_length )

        # Quadrotor arms
        self.arm_b1.draw_from_to(x + R.dot(np.array([motor_pos, -motor_pos, 0]).T) * self.arm_length, x + R.dot(np.array([-motor_pos, +motor_pos, 0]).T) * self.arm_length)
        self.arm_b2.draw_from_to(x + R.dot(np.array([-motor_pos, -motor_pos, 0]).T) * self.arm_length, x + R.dot(np.array([motor_pos,motor_pos, 0]).T) * self.arm_length)



if __name__ == '__main__':
    from utils import ypr_to_R

    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D

    import matplotlib.pyplot as plt

    
    def update_plot(i, x, R):
        uav_plot.draw_at(x[:, i], R[:, :, i])
        
        # These limits must be set manually since we use
        # a different axis frame configuration than the
        # one matplotlib uses.
        xmin, xmax = -2, 2
        ymin, ymax = -2, 2
        zmin, zmax = -2, 2

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymax, ymin])
        ax.set_zlim([zmax, zmin])

    # Initiate the plot
    plt.style.use('seaborn')

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    arm_length = 0.24  # in meters
    uav_plot = Uav(ax, arm_length)


    # Create some fake simulation data
    steps = 60
    t_end = 1

    x = np.zeros((3, steps))
    x[0, :] = np.arange(0, t_end, t_end / steps)
    x[1, :] = np.arange(0, t_end, t_end / steps) * 2

    R = np.zeros((3, 3, steps))
    for i in range(steps):
        ypr = np.array([i, 0.1 * i, 0.0])
        R[:, :, i] = ypr_to_R(ypr, degrees=True)


    # Run the simulation
    ani = animation.FuncAnimation(fig, update_plot, frames=steps, \
        fargs=(x, R,))
    
    plt.show()