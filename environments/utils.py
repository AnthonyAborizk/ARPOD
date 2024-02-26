# This script contains "helper" functions for the docking environments

import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

def skew(vector):

    """
    this function returns a numpy array with the skew symmetric cross product matrix for vector.
    the skew symmetric cross product matrix is defined such that
    np.cross(a, b) = np.dot(skew(a), b)

    :param vector: An array like vector to create the skew symmetric cross product matrix
    :return: A numpy array of the skew symmetric cross product vector
    """

    return np.array([[0, -vector[2,0], vector[1,0]], 
                    [vector[2,0], 0, -vector[0,0]], 
                    [-vector[1,0], vector[0,0], 0]])


def Yaw(psi):

    '''
        This yaw function creates a passive rotation matrix in SO(3) and a passive quaternion rotation
        given an input angle psi 
    ''' 
    psi = psi.item()
    q_yaw = np.array([[cos(psi/2)], [0], [0], [sin(psi/2)]])

    return q_yaw


def Pitch(theta):
    '''
    # This pitch function creates a passive rotation matrix in SO(3) and a passive quaternion rotation
    # given an input angle theta 
    '''
    theta = theta.item()
    q_pitch = np.array([[cos(theta/2)], [0], [sin(theta/2)], [0]])

    return  q_pitch


def Roll(phi):
    '''
    # This roll function creates a passive rotation matrix in SO(3) and a passive quaternion rotation
    # given an input angle phi 
    '''
    phi = phi.item()
    q_roll = np.array([[cos(phi/2)], [sin(phi/2)], [0], [0]])

    return q_roll


def QuatMultiply(q_1, q_2):
        # This function performs Quaternion multiplication between two unit quaternions
        
        if len(q_1) == 3:
            q_1 = np.concatenate(([0], q_1))
        elif len(q_2) == 3:
            q_2 = np.concatenate(([0], q_2))
        
        if len(q_1) != 4 or len(q_2) != 4:
            raise ValueError('Quaternions must contain 4 elements.')
        
        eta_1 = q_1[0].item()    # scalar 1
        rho_1 = q_1[1:4]  # vector 1
        eta_2 = q_2[0].item()    # scalar 1
        rho_2 = q_2[1:4]  # vector 1
        
        skew_mat = skew(rho_1)
        
        q_3 = np.concatenate((eta_1 * eta_2 - np.dot(rho_1.T, rho_2), eta_1 * rho_2 + eta_2 * rho_1 + np.dot(skew_mat, rho_2)))
        
        return q_3


def EulerConvert(Gamma): 
    psi   = Gamma[0]
    theta = Gamma[1]
    phi   = Gamma[2]

    while psi>pi:
        psi = psi - 2*pi

    while psi < 0:
        psi = psi + 2 * np.pi
    
    # Pitch constraint
    while theta > 2 * np.pi:
        theta = theta - 2 * np.pi
    
    while theta < 0:
        theta = theta + 2 * np.pi
    
    # Roll constraint
    while phi > 2 * np.pi:
        phi = phi - 2 * np.pi
    
    while phi < 0:
        phi = phi + 2 * np.pi
    
    q_yaw   = Yaw(psi)
    q_pitch = Pitch(theta)
    q_roll  = Roll(phi)
    
    # Quaternion (remember order is reverse due to passive convention)
    q = QuatMultiply(q_yaw, QuatMultiply(q_pitch, q_roll))
    
    eta = q[0].item()    # scalar 1
    rho = q[1:4].squeeze()  # vector 1
    
    # Rodrigues Parameters
    
    # Modified Rodrigues Parameters
    zeta = rho / (1 + eta)
    
    # Shadow Rodrigues Parameters
    sigma = rho / (eta - 1)
    
    # I concatenate both of these because they are duals of each other
    MRP = np.concatenate((zeta, sigma))
    
    return np.array(MRP, dtype=np.float32)

def EulerToQuaternion(Gamma): 
    psi, theta, phi = Gamma[0], Gamma[1], Gamma[2]

    # Yaw Constraint
    while psi>pi:
        psi = psi - 2*pi

    while psi < 0:
        psi = psi + 2 * np.pi
    
    # Pitch constraint
    while theta > 2 * np.pi:
        theta = theta - 2 * np.pi
    
    while theta < 0:
        theta = theta + 2 * np.pi
    
    # Roll constraint
    while phi > 2 * np.pi:
        phi = phi - 2 * np.pi
    
    while phi < 0:
        phi = phi + 2 * np.pi
    
    q_yaw, q_pitch, q_roll = Yaw(psi), Pitch(theta), Roll(phi)
    
    # Quaternion (remember order is reverse due to passive convention)
    q = QuatMultiply(q_yaw, QuatMultiply(q_pitch, q_roll))
    q = q.squeeze()
    
    # Quaternion has [w,x,y,z] form
    # w = scalar part
    # x,y,z = vector part

    return np.array(q, dtype=np.float32)

def QuaternionToEuler(q):
    """
    Convert quaternion to Euler angles (yaw, pitch, roll)
    :param q: Quaternion(s) in the form of [w, x, y, z], or an array of shape (N, 4)
    :return: Euler angles in degrees
    """
    if q.ndim == 1:
        w, x, y, z = q
        q = np.array([w, x, y, z])
        q = q / np.linalg.norm(q)  # Normalize quaternion
        # Roll (x-axis rotation)
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        # Pitch (y-axis rotation)
        pitch = np.arctan2(2 * (w * y + z * x), 1 - 2 * (y**2 + x**2))
        # Yaw (z-axis rotation)
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (z**2 + y**2))
    else:
        roll = np.arctan2(2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]),
                          1 - 2 * (q[:, 1]**2 + q[:, 2]**2))
        pitch = np.arctan2(2 * (q[:, 0] * q[:, 2] + q[:, 3] * q[:, 1]),
                           1 - 2 * (q[:, 2]**2 + q[:, 1]**2))
        yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                         1 - 2 * (q[:, 3]**2 + q[:, 2]**2))

    # Convert radians to degrees
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    
    return np.column_stack((yaw, pitch, roll))

def plots(states, actions, env_name):
    '''
    Plots the states and actions of a given trajectory
    '''
    x  = states[:,:,0].squeeze()
    y  = states[:,:,1].squeeze()
    time = np.linspace(0, len(x)/2, len(x))
    
    if env_name == 'ActuatedDocking':
        theta = states[:, : ,2].squeeze()
        vx    = states[:,:,3]
        vy    = states[:,:,4]
        omega = states[:,:,5]

        fx  = actions[:, 0]
        fy  = actions[:, 1]
        psi = actions[:, 2]
    
        #************* POSITION *******************
        plt.figure()
        plt.plot(time, x, label='x')
        plt.plot(time, y, label='y')
        plt.ylabel('Position [m]')
        plt.xlabel('Times [s]')
        plt.legend()

        #************* ANGLE *******************
        plt.figure()
        plt.plot(time, theta, label='theta')
        plt.ylabel('Angle [rad]')
        plt.xlabel('Time [s]')

        #************ Control Input ****************
        plt.figure()
        plt.plot(time, fx, alpha = 0.3, label='Fx [N]')
        plt.plot(time, fy, label='Fy [N]')
        plt.plot(time, psi, label='fluwheel input [rad/s^2]')
        plt.ylabel('Force')
        plt.xlabel('Time [s]')
        plt.legend()

        #************* TRAJECTORY *******************
        plt.figure()  
        plt.scatter(y, x, s=2)
        plt.plot(y[0], x[0], 'o', label='start')
        plt.plot(y[-1], x[-1], 'o', label= 'end')
        plt.gca().invert_xaxis()
        plt.xlabel('Local Horizontal [m]')
        plt.ylabel('Local Vertical [m]')
        plt.title('Trajectory')
        plt.legend() 

        #************* VELOCITY *******************
        plt.figure()
        plt.plot(time, vx, label='xvelocity')
        plt.plot(time, vy, label='yvelocity')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.legend()

        #************* ANGULAR VELOCITY ****************
        plt.figure()
        plt.plot(time, omega)
        plt.ylabel('Angular Velocity [rad/s]')
        plt.xlabel('Time [s]')
        plt.legend() 

        #************* ORIENTATION *******************
        plt.figure()
        i = 32
        ax = plt.gca()
        ax.quiver(y[::i], x[::i], -np.sin(theta[::i]), np.cos(theta[::i]),scale=30, color='r', label='Orientation' )
        ax.quiver(y[::i], x[::i], -fx[::i]*np.sin(theta[::i]), fx[::i]*np.cos(theta[::i]), scale=100, label='Force')
        ax.invert_xaxis()
        ax.set_aspect('equal')
        ax.set_xlabel('Local Horizontal [m]')
        ax.set_ylabel('Local Vertical [m]')
        ax.legend()

    elif env_name == 'ActuatedDocking3DOF':
        z  = states[:,:,2].squeeze()
        vx = states[:,:,3].squeeze()
        vy = states[:,:,4].squeeze()
        vz = states[:,:,5].squeeze()

        fx = actions[:, 0]
        fy = actions[:, 1]
        fz = actions[:, 2]

        #************* POSITION *******************
        plt.figure()
        plt.plot(time, x, label='x')
        plt.plot(time, y, label='y')
        plt.plot(time, z, label='z')
        plt.ylabel('Position [m]')
        plt.xlabel('Times [s]')
        plt.legend()

        #************ Control Input ****************
        plt.figure()
        plt.plot(time, fx, label='Fx [N]')
        plt.plot(time, fy, label='Fy [N]')
        plt.plot(time, fz, label='Fz [N]')
        plt.ylabel('Force')
        plt.xlabel('Time [s]')
        plt.legend()

        #************* TRAJECTORY *******************
        fig = plt.figure()
        ax   = fig.add_subplot(111, projection='3d')  # Project into 3 dims
        scat_plot=ax.scatter(y, x, z, s=1, label='trajectory')
        ax.scatter(y[-1], x[-1], z[-1],  s=20, label= 'end')
        ax.scatter(y[0], x[0], z[0], s=20, label='start')
        ax.invert_xaxis()
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_zlabel('z')

        plt.title('Trajectory')
        plt.legend() 

        #************* VELOCITY *******************
        plt.figure()
        plt.plot(time, vx, label='xvelocity')
        plt.plot(time, vy, label='yvelocity')
        plt.plot(time, vz, label='zvelocity')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.legend()

    elif env_name == 'ActuatedDocking6DOF':
        x  = states[:,:,0].squeeze()
        y  = states[:,:,1].squeeze()
        z  = states[:,:,2].squeeze()
        vx = states[:,:,3].squeeze()
        vy = states[:,:,4].squeeze()
        vz = states[:,:,5].squeeze()
        q = states[:,:,6:10].squeeze()
        omegax = states[:,:,10].squeeze()
        omegay = states[:,:,11].squeeze()
        omegaz = states[:,:,12].squeeze()
        nux = states[:,:,13].squeeze()
        nuy = states[:,:,14].squeeze()
        nuz = states[:,:,15].squeeze()

        ux = actions[:, 0]
        uy = actions[:, 1]
        uz = actions[:, 2]
        fx = actions[:, 3]
        fy = actions[:, 4]
        fz = actions[:, 5]

        Euler = QuaternionToEuler(q)
        psi   = Euler[:,0]
        theta = Euler[:,1]
        phi   = Euler[:,2]

        #************* POSITION *******************
        plt.figure()
        plt.plot(time, x, label='x')
        plt.plot(time, y, label='y')
        plt.plot(time, z, label='z')
        rho = np.linalg.norm([x,y, z], axis=0)
        plt.plot(time, rho, label='r_mag')
        plt.ylabel('Position [m]')
        plt.xlabel('Times [s]')
        plt.legend()

        #************* VELOCITY *******************
        plt.figure()
        plt.plot(time, vx, label='xvelocity')
        plt.plot(time, vy, label='yvelocity')
        plt.plot(time, vz, label='zvelocity')
        vH = np.linalg.norm([vx, vy, vz], axis=0)    # Velocity Magnitude
        plt.plot(time, vH, label='vel_mag')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.legend()
        
        #************* ANGLE *******************
        plt.figure()
        plt.plot(time, psi, label='psi')
        plt.plot(time, theta, label='theta')
        plt.plot(time, phi, label='phi')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.legend() 

        #************* ANGULAR VELOCITY *******************
        plt.figure()
        plt.plot(time, omegax, label='omegax')
        plt.plot(time, omegay, label='omegay')
        plt.plot(time, omegaz, label='omegaz')
        omegaH = np.linalg.norm([omegax, omegay, omegaz], axis=0)    # Angular Velocity Magnitude
        plt.plot(time, omegaH, label='omega_mag')
        plt.xlabel('Time [s]')
        plt.ylabel('Angular Velocity [rad/s]')
        plt.legend()

        #************* Flywheel velocity *******************
        plt.figure()
        plt.plot(time, nux, label='nux')
        plt.plot(time, nuy, label='nuy')
        plt.plot(time, nuz, label='nuz')
        nuH = np.linalg.norm([nux, nuy, nuz], axis=0)    # Flywheel Velocity Magnitude
        plt.plot(time, nuH, label='nu_mag')
        plt.xlabel('Time [s]')
        plt.ylabel('Flywheel Velocity [rad/s]')
        plt.legend()

        #************* ACTION *******************
        plt.figure()
        plt.plot(time, ux, label='ux [rad/s^2]')
        plt.plot(time, uy, label='uy [rad/s^2]')
        plt.plot(time, uz, label='uz [rad/s^2]')
        uH = np.linalg.norm([ux, uy, uz], axis=0)    # Torque Magnitude
        plt.plot(time, uH, label='u_mag')
        plt.plot(time, fx, label='Fx [N]')
        plt.plot(time, fy, label='Fy [N]')
        plt.plot(time, fz, label='Fz [N]')
        fH = np.linalg.norm([fx, fy, fz], axis=0)    # Force Magnitude
        plt.plot(time, fH, label='F_mag')
        plt.ylabel('Force [N]')
        plt.xlabel('Time [s]')
        plt.legend()

        #************* TRAJECTORY *******************
        p, q = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        r = np.cos(p)*np.sin(q) * 100
        s = np.sin(p)*np.sin(q) * 100
        t = np.cos(q) * 100

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # Project into 3 dims
        ax.plot_wireframe(r, s, t, color='k', alpha=0.15)
        scat_plot=ax.scatter(y, x, z, s=1, label='trajectory')
        ax.scatter(y[-1], x[-1], z[-1], s=20, label= 'end')
        ax.scatter(y[0], x[0], z[0], s=20, label='start')
        ax.invert_xaxis()
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_zlabel('z')
        ax.set_aspect('equal')
        ax.grid(False)
        plt.title('Trajectory')
        plt.legend()

    elif env_name == 'UnderactuatedDocking':

        theta = states[:, : ,2].squeeze()
        vx    = states[:,:,3]
        vy    = states[:,:,4]
        omega = states[:,:,5]

        fx  = actions[:, 0]
        psi = actions[:, 1]
    
        #************* POSITION *******************
        plt.figure()
        plt.plot(time, x, label='x')
        plt.plot(time, y, label='y')
        plt.ylabel('Position [m]')
        plt.xlabel('Times [s]')
        plt.legend()

        #************* ANGLE *******************
        plt.figure()
        plt.plot(time, theta, label='theta')
        plt.ylabel('Angle [rad]')
        plt.xlabel('Time [s]')

        #************ Control Input ****************
        plt.figure()
        plt.plot(time, fx, alpha = 0.3, label='Fx [N]')
        plt.plot(time, psi, label='fluwheel input [rad/s^2]')
        plt.ylabel('Force')
        plt.xlabel('Time [s]')
        plt.legend()

        #************* TRAJECTORY *******************
        plt.figure()  
        plt.scatter(y, x, s=2)
        plt.plot(y[0], x[0], 'o', label='start')
        plt.plot(y[-1], x[-1], 'o', label= 'end')
        plt.gca().invert_xaxis()
        plt.xlabel('Local Horizontal [m]')
        plt.ylabel('Local Vertical [m]')
        plt.title('Trajectory')
        plt.legend() 

        #************* VELOCITY *******************
        plt.figure()
        plt.plot(time, vx, label='xvelocity')
        plt.plot(time, vy, label='yvelocity')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.legend()

        #************* ANGULAR VELOCITY ****************
        plt.figure()
        plt.plot(time, omega)
        plt.ylabel('Angular Velocity [rad/s]')
        plt.xlabel('Time [s]')
        plt.legend() 

        #************* ORIENTATION *******************
        plt.figure()
        i = 32
        ax = plt.gca()
        ax.quiver(y[::i], x[::i], -np.sin(theta[::i]), np.cos(theta[::i]),scale=30, color='r', label='Orientation' )
        ax.quiver(y[::i], x[::i], -fx[::i]*np.sin(theta[::i]), fx[::i]*np.cos(theta[::i]), scale=100, label='Force')
        ax.invert_xaxis()
        ax.set_xlabel('Local Horizontal [m]')
        ax.set_ylabel('Local Vertical [m]')
        ax.legend()

    elif env_name == 'UnderactuatedDocking6DOF':

        z  = states[:,:,2].squeeze()
        vx = states[:,:,3].squeeze()
        vy = states[:,:,4].squeeze()
        vz = states[:,:,5].squeeze()
        mux = states[:,:,6].squeeze()
        muy = states[:,:,7].squeeze()
        muz = states[:,:,8].squeeze()
        omegax = states[:,:,9].squeeze()
        omegay = states[:,:,10].squeeze()
        omegaz = states[:,:,11].squeeze()
        nux = states[:,:,12].squeeze()
        nuy = states[:,:,13].squeeze()
        nuz = states[:,:,14].squeeze()

        ux = actions[:, 0]
        uy = actions[:, 1]
        uz = actions[:, 2]
        fx = actions[:, 3]

        #************* POSITION *******************
        plt.figure()
        plt.plot(time, x, label='x')
        plt.plot(time, y, label='y')
        plt.plot(time, z, label='z')
        plt.ylabel('Position [m]')
        plt.xlabel('Times [s]')
        plt.legend()

        #************* VELOCITY *******************
        plt.figure()
        plt.plot(time, vx, label='xvelocity')
        plt.plot(time, vy, label='yvelocity')
        plt.plot(time, vz, label='zvelocity')
        rho = np.linalg.norm([x,y, z], axis=0)
        vH = np.linalg.norm([vx, vy, vz], axis=0)    # Velocity Magnitude
        vH_max = 2 * 0.001027 * rho + 0.2            # Max Velocity
        vH_min = 1/2 * 0.001027 * rho - 0.2          # Min Velocity
        plt.plot(time, vH, label='vel_mag')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.legend()
        
        #************* ANGLE *******************
        plt.figure()
        plt.plot(time, mux, label='mux')
        plt.plot(time, muy, label='muy')
        plt.plot(time, muz, label='muz')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [rad]')
        plt.legend()

        #************* ANGULAR VELOCITY *******************
        plt.figure()
        plt.plot(time, omegax, label='omegax')
        plt.plot(time, omegay, label='omegay')
        plt.plot(time, omegaz, label='omegaz')
        plt.xlabel('Time [s]')
        plt.ylabel('Angular Velocity [rad/s]')
        plt.legend()

        #************* Flywheel velocity *******************
        plt.figure()
        plt.plot(time, nux, label='nux')
        plt.plot(time, nuy, label='nuy')
        plt.plot(time, nuz, label='nuz')
        plt.xlabel('Time [s]')
        plt.ylabel('Flywheel Velocity [rad/s]')
        plt.legend()

        #************* ACTION *******************
        plt.figure()
        plt.plot(time, ux, label='ux [rad/s^2]')
        plt.plot(time, uy, label='uy [rad/s^2]')
        plt.plot(time, uz, label='uz [rad/s^2]')
        plt.plot(time, fx, label='Fx [N]')
        plt.ylabel('Force [N]')
        plt.xlabel('Time [s]')
        plt.legend()

        #************* TRAJECTORY *******************
        p, q = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        r = np.cos(p)*np.sin(q) * 100
        s = np.sin(p)*np.sin(q) * 100
        t = np.cos(q) * 100

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # Project into 3 dims
        ax.plot_wireframe(r, s, t, color='k', alpha=0.15)
        scat_plot=ax.scatter(y, x, z, s=1, label='trajectory')
        ax.scatter(y[-1], x[-1], z[-1], s=20, label= 'end')
        ax.scatter(y[0], x[0], z[0], s=20, label='start')
        ax.invert_xaxis()
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_zlabel('z')
        ax.set_aspect('equal')
        ax.grid(False)
        plt.title('Trajectory')
        plt.legend() 

    plt.show()
    