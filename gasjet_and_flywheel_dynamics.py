from utils import *


def RotVelocityA(t, R, omega_A):
    '''
    This function gets the time rate derivative of the rotation matrix R and angular velocity of form omega^{A}_{AB}
    '''

    R = R.reshape(3, 3)
    dRdt = skew(omega_A)@R
    dRdt = dRdt.reshape(9, 1)

    return dRdt

def RotVelocityB(t, R, omega_B):
    '''
    This function gets the time rate derivative of the rotation matrix R and angular velocity of form omega^{B}_{AB}
    '''

    R = R.reshape(3, 3)
    dRdt = R@skew(omega_B)
    dRdt = dRdt.reshape(9,)

    return dRdt

def MRPVelocityA(t, MRP, omega_A):
    # This function gets the time rate derivative of the quaternion q and angular velocity of form omega^{A}_{AB}

    zeta = MRP[:3]
    sigma = MRP[3:]

    eye3 = np.eye(3)
    zeta_skew = np.array([[0, -zeta[2], zeta[1]],
                          [zeta[2], 0, -zeta[0]],
                          [-zeta[1], zeta[0], 0]])
    sigma_skew = np.array([[0, -sigma[2], sigma[1]],
                           [sigma[2], 0, -sigma[0]],
                           [-sigma[1], sigma[0], 0]])

    dzetadt = -(1/2) * ((1/2) * (1 - np.dot(zeta, zeta)) * eye3 + zeta_skew + np.outer(zeta, zeta)) @ omega_A
    dsigmadt = -(1/2) * ((1/2) * (1 - np.dot(sigma, sigma)) * eye3 + sigma_skew + np.outer(sigma, sigma)) @ omega_A

    dMRPdt = np.concatenate((dzetadt, dsigmadt))

    return dMRPdt

def MRPVelocityB(t, MRP, omega_B):
    '''
    This function gets the time rate derivative of the quaternion q and angular velocity of form omega^{B}_{AB}
    '''
    omega_B = omega_B.reshape(3,1)
    zeta = np.array([MRP[0], MRP[1], MRP[2]]).reshape(3,1)
    sigma = np.array([MRP[3], MRP[4], MRP[5]]).reshape(3,1)

    dzetadt = -1/2 *( 1/2 * (1 - zeta.T@zeta)*np.eye(3) - skew(zeta) + zeta@zeta.T)@omega_B
    dsigmadt = -1/2 *( 1/2 * (1 - sigma.T@sigma)*np.eye(3) - skew(sigma) + sigma@sigma.T)@omega_B
    dMRPdt = np.vstack([dzetadt, dsigmadt])

    return dMRPdt.squeeze() 

def EulerVelocityA(t, Gamma, omega_A):
    # This function gets the time rate derivative of the Euler angle set Gamma and angular velocity of form omega^{A}_{AB}

    psi = Gamma[0]
    theta = Gamma[1]
    phi = Gamma[2]

    # Note that we will have to keep each Euler angle onto the [0, 2pi] range.
    # Here we must enforce angle restrictions as well

    # Yaw constraint
    while psi > 2 * np.pi:
        psi -= 2 * np.pi

    while psi < 0:
        psi += 2 * np.pi

    # Pitch constraint
    while theta > 2 * np.pi:
        theta -= 2 * np.pi

    while theta < 0:
        theta += 2 * np.pi

    # Roll constraint
    while phi > 2 * np.pi:
        phi -= 2 * np.pi

    while phi < 0:
        phi += 2 * np.pi

    # Calculate the time rate derivative of Gamma
    dGammadt = np.array([
        [0, -np.sin(phi) / np.cos(theta), -np.cos(phi) / np.cos(theta)],
        [0, -np.cos(phi), np.sin(phi)],
        [-1, -np.sin(phi) / np.tan(theta), -np.cos(phi) / np.tan(theta)]
    ]) @ omega_A

    return dGammadt

def EulerVelocityB(t, Gamma, omega_B):
    # This function gets the time rate derivative of the Euler angle set Gamma and angular velocity of form omega^{B}_{AB}

    psi   = Gamma[0]
    theta = Gamma[1]
    phi   = Gamma[2]

    # Here we must enforce angle restrictions as well

    # Yaw constraint
    while psi > 2 * pi:
        psi -= 2 * pi

    while psi < 0:
        psi += 2 * pi

    # Pitch constraint
    while theta > 2 * pi:
        theta -= 2 * pi

    while theta < 0:
        theta += 2 * pi

    # Roll constraint
    while phi > 2 * pi:
        phi -= 2 * pi

    while phi < 0:
        phi += 2 * pi

    # Calculate the time rate derivative of Gamma
    dGammadt = np.array([
        [-cos(psi) * np.tan(theta), -sin(psi) * np.tan(theta), -1],
        [sin(psi), -cos(psi), 0],
        [-cos(psi) / cos(theta), -sin(psi) / cos(theta), 0]
    ]) @ omega_B

    return dGammadt

def QuatVelocityA(t, q, omega_A):
    # This function gets the time rate derivative of the quaternion q and angular velocity of 
    # form omega^{A}_{AB}

    eta = q[0]
    rho = q[1:]

    S_rho = np.array([[0, -rho[2], rho[1]],
                      [rho[2], 0, -rho[0]],
                      [-rho[1], rho[0], 0]])

    dqdt = -(1/2) * np.concatenate(([-rho], eta*np.eye(3) + S_rho)) @ omega_A

    return dqdt

def QuatVelocityB(t, q, omega_B):
    # This function gets the time rate derivative of the quaternion q and angular velocity of form omega^{B}_{AB}

    eta = q[0]
    rho = q[1:]

    S_rho = np.array([[0, -rho[2], rho[1]],
                      [rho[2], 0, -rho[0]],
                      [-rho[1], rho[0], 0]])

    dqdt = -(1/2) * np.concatenate(([-rho], eta*np.eye(3) - S_rho)) @ omega_B

    return dqdt
