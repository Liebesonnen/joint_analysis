import numpy as np
from scipy.linalg import expm, logm, norm


def hat(vec):
    """
    Take the 3- or 6-vector representing an isomorphism of so(3) or se(3) and
    writes this as element of so(3) or se(3).

    Parameters:
    vec (numpy.ndarray): 3- or 6-vector. Isomorphism of so(3) or se(3)

    Returns:
    numpy.ndarray: element of so(3) or se(3)
    """
    if len(vec) == 3:
        res = np.array([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0]
        ])
    elif len(vec) == 6:
        skew = np.array([
            [0, -vec[5], vec[4]],
            [vec[5], 0, -vec[3]],
            [-vec[4], vec[3], 0]
        ])
        v = vec[0:3].reshape(3, 1)
        res = np.block([
            [skew, v],
            [np.zeros((1, 4))]
        ])
    return res


def vee(mat):
    """
    Takes an element of so(3) or se(3) and returns its isomorphism in R^n.

    Parameters:
    mat (numpy.ndarray): element of so(3) or se(3)

    Returns:
    numpy.ndarray: 3- or 6-vector. Isomorphism of so(3) or se(3)
    """
    xi1 = (mat[2, 1] - mat[1, 2]) / 2
    xi2 = (mat[0, 2] - mat[2, 0]) / 2
    xi3 = (mat[1, 0] - mat[0, 1]) / 2

    if mat.shape[0] == 3:
        res = np.array([xi1, xi2, xi3])
    elif mat.shape[0] == 4:
        res = np.concatenate([mat[0:3, 3], [xi1, xi2, xi3]])

    return res


def expSO3(a):
    """
    Computes the exponential mapping on SO(3)

    Parameters:
    a (numpy.ndarray): 3 vector, isomorphism to element of so(3)

    Returns:
    numpy.ndarray: element of SO(3)
    """
    phi = np.linalg.norm(a)
    if phi != 0:
        a_hat = hat(a)
        res = np.eye(3) + np.sin(phi) / phi * a_hat + (1 - np.cos(phi)) / (phi ** 2) * a_hat @ a_hat
    else:
        res = np.eye(3)
    return res


def dexpSO3(a):
    """
    Computes the right trivialized tangent d exp on SO(3)

    Parameters:
    a (numpy.ndarray): 3 vector, isomorphism to element of so(3)

    Returns:
    numpy.ndarray: diff exponential of a
    """
    phi = np.linalg.norm(a)
    a_hat = hat(a)

    if phi != 0:
        beta = (np.sin(phi / 2) ** 2) / ((phi / 2) ** 2)
        alpha = np.sin(phi) / phi
        res = np.eye(3) + 0.5 * beta * a_hat + 1 / (phi ** 2) * (1 - alpha) * a_hat @ a_hat
    else:
        # Handle the case when phi is 0
        res = np.eye(3)

    return res


def DdexpSO3(x, z):
    """
    Directional derivative of the dexp at x in the direction of z

    Parameters:
    x (numpy.ndarray): 3-vector, element of so(3)
    z (numpy.ndarray): 3-vector, element of so(3)

    Returns:
    numpy.ndarray: resulting derivative
    """
    hat_x = hat(x)
    hat_z = hat(z)
    phi = np.linalg.norm(x)

    if phi != 0:
        beta = (np.sin(phi / 2) ** 2) / ((phi / 2) ** 2)
        alpha = np.sin(phi) / phi

        res = 0.5 * beta * hat_z \
              + 1 / (phi ** 2) * (1 - alpha) * (hat_x @ hat_z + hat_z @ hat_x) \
              + 1 / (phi ** 2) * (alpha - beta) * (x @ z) * hat_x \
              + 1 / (phi ** 2) * (beta / 2 - 3 / (phi ** 2) * (1 - alpha)) * (x @ z) * hat_x @ hat_x
    else:
        # Handle the case when phi is 0
        res = 0.5 * hat_z

    return res


def tight_subplot(Nh, Nw, gap, marg_h, marg_w):
    """
    Creates subplot axes with adjustable gaps and margins

    Parameters:
    Nh (int): number of axes in height (vertical direction)
    Nw (int): number of axes in width (horizontal direction)
    gap (float or list): gaps between the axes in normalized units (0...1)
                        or [gap_h gap_w] for different gaps in height and width
    marg_h (float or list): margins in height in normalized units (0...1)
                           or [lower upper] for different lower and upper margins
    marg_w (float or list): margins in width in normalized units (0...1)
                           or [left right] for different left and right margins

    Returns:
    ha (list): array of handles of the axes objects
    pos (list): positions of the axes objects
    """
    import matplotlib.pyplot as plt

    if gap is None:
        gap = 0.02
    if marg_h is None:
        marg_h = 0.05
    if marg_w is None:
        marg_w = 0.05

    if isinstance(gap, (int, float)):
        gap = [gap, gap]
    if isinstance(marg_w, (int, float)):
        marg_w = [marg_w, marg_w]
    if isinstance(marg_h, (int, float)):
        marg_h = [marg_h, marg_h]

    axh = (1 - sum(marg_h) - (Nh - 1) * gap[0]) / Nh
    axw = (1 - sum(marg_w) - (Nw - 1) * gap[1]) / Nw

    py = 1 - marg_h[1] - axh

    ha = []
    pos = []

    ii = 0
    for ih in range(Nh):
        px = marg_w[0]

        for ix in range(Nw):
            ii += 1
            ax = plt.axes([px, py, axw, axh])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ha.append(ax)
            pos.append([px, py, axw, axh])
            px = px + axw + gap[1]

        py = py - axh - gap[0]

    return ha, pos


def sgolayfiltSO3(R, p, n, freq):
    """
    This function applies a Savitzky-Golay finite impulse response (FIR)
    smoothing filter of polynomial order p and frame length n to the data in
    the sequence of noisy rotation matrices R.

    Parameters:
    R (numpy.ndarray): Noisy sequence of rotation matrices, specified as
                      a 3-by-3-by-N array containing N rotation matrices
    p (int): Polynomial order, specified as a positive integer,
            less than the window size, n
    n (int): Window size, specified as a positive integer.
    freq (int): Sample frequency, specified as positive integer.

    Returns:
    R_est (numpy.ndarray): Estimated rotation matrices.
    omg_est (numpy.ndarray): Estimated angular velocity.
    domg_est (numpy.ndarray): Estimated angular acceleration.
    tf (numpy.ndarray): Time vector of the filtered signal.
    """
    # Check inputed values
    if p < 0 or p >= n or p != int(p):
        raise ValueError("The polynomial order, p, must be a positive integer less than the window size, n.")
    if n < 0 or n != int(n):
        raise ValueError("The window size, n, should be a positive integer.")
    if freq < 0 or freq != int(freq):
        raise ValueError("The frequency, freq, should be a positive integer.")

    # Computed values
    N = R.shape[2]  # Number of samples in the sequence    [-]
    dt = 1 / freq  # Time step lower sampled              [s]
    te = N * dt  # Total length of the sequence         [s]
    ts = np.arange(0, te + dt, dt)  # Signal time vector                   [s]
    w = np.arange(-n, n + 1)  # Window for Golay                     [-]
    I = np.eye(3)  # Short hand notation                  [-]

    # Adjust to match MATLAB's indexing
    # In MATLAB: tf = ts((n+1):(N-(n+1)))
    tf = ts[n + 1:N - (n + 1) + 1]  # Time vector filtered signal          [s]

    # Explicitly calculate filtered size to match the loop iterations
    filtered_size = N - 2 * n - 1

    # Preallocate memory with the correct size
    R_est = np.zeros((3, 3, filtered_size))
    omg_est = np.zeros((3, filtered_size))
    domg_est = np.zeros((3, filtered_size))

    # Savitzky-Golay
    # For each time step (where we can apply the window)
    # Adjust loop to match MATLAB's for ii = (n+1):(N-(n+1))
    cnt = 0
    for ii in range(n + 1, N - (n + 1) + 1):
        # Build matrix A and vector b based on window size w
        A = np.zeros((3 * len(w), 3 * (p + 1)))
        b = np.zeros(3 * len(w))

        row = 0
        for jj in range(len(w)):
            # Time difference between 0^th element and w(jj)^th element
            Dt = (ts[ii + w[jj]] - ts[ii])

            # Determine row of A matrix
            Ajj = I.copy()
            for kk in range(1, p + 1):
                Ajj = np.hstack((Ajj, (1 / kk) * (Dt ** kk) * I))  # Concatenation based on order p

            A[row:row + 3, :] = Ajj
            b[row:row + 3] = vee(logm(R[:, :, ii + w[jj]] @ np.linalg.inv(R[:, :, ii])))
            row += 3  # Update to next row

        # Solve the LS problem
        rho = np.linalg.lstsq(A.T @ A, A.T @ b, rcond=None)[0]

        # Obtain the coefficients of rho
        rho0 = rho[0:3]
        rho1 = rho[3:6]
        rho2 = rho[6:9]

        # Compute analytically the rotation matrices, ang. vel., and ang. acc.
        R_est[:, :, cnt] = expSO3(rho0) @ R[:, :, ii]
        omg_est[:, cnt] = dexpSO3(rho0) @ rho1
        domg_est[:, cnt] = DdexpSO3(rho0, rho1) @ rho1 + dexpSO3(rho0) @ rho2

        # Update the index counter
        cnt += 1

    return R_est, omg_est, domg_est, tf