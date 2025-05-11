import numpy as np
from scipy.linalg import expm, logm, norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from so3_functions import (hat, vee, expSO3, dexpSO3, DdexpSO3,
                           sgolayfiltSO3, tight_subplot)

# Set matplotlib to use LaTeX for rendering if available
try:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
except:
    print("LaTeX rendering not available, using default text renderer")

# Constants and settings
# User inputs
doSave = True  # Boolean: set true if you want to save figures
Fc = 1  # Signal frequency                  [Hz]
a = 2  # Signal amplitude                  [deg]
te = 2  # Signal length                     [s]
Fs = 1000  # Sampling frequency fine grid      [Hz]
m = 5  # Down-sampling rate                [-]
sigma = 0.06  # Standard deviation of added noise [rad]
n = 20  # Window size SG-filter             [-]
p = 3  # Savitzky Golay filter order       [-]

# Computed values
dt1 = 1 / Fs  # Time step                         [s]
dt2 = m / Fs  # Time step lower sampled           [s]
t1 = np.arange(0, te + dt1, dt1)  # Signal time vector                [s]
t2 = np.arange(0, te + dt2, dt2)  # Signal time vector lower sampled  [s]
N1 = len(t1)  # Number of samples                 [-]
N2 = len(t2)  # Number of samples lower sampled   [-]

# Preallocate memory
omg = np.zeros((3, N1))
omg_FD = np.zeros((3, N2))
domg = np.zeros((3, N1))
domg_FD = np.zeros((3, N2))
R = np.zeros((3, 3, N1))
R_noise = np.zeros((3, 3, N2))

phi = np.zeros((3, N1))
dphi = np.zeros((3, N1))
ddphi = np.zeros((3, N1))

# Creating data on SO(3)
# Use the predefined values from the MATLAB code
lambda0 = np.array([-0.4831, 0.6064, -2.6360])
lambda1 = np.array([0.9792, 1.4699, -0.4283])

for ii in range(N1):
    freq = 2 * np.pi * Fc
    phi[:, ii] = lambda0 + lambda1 * a * np.sin(freq * t1[ii])
    dphi[:, ii] = lambda1 * a * (freq) * np.cos(freq * t1[ii])
    ddphi[:, ii] = -lambda1 * a * (freq) ** 2 * np.sin(freq * t1[ii])

    # Compute analytically the rotation matrices, ang. vel., and ang. acc.
    R[:, :, ii] = expSO3(phi[:, ii])
    omg[:, ii] = dexpSO3(phi[:, ii]) @ dphi[:, ii]
    domg[:, ii] = DdexpSO3(phi[:, ii], dphi[:, ii]) @ dphi[:, ii] + dexpSO3(phi[:, ii]) @ ddphi[:, ii]

# Noisy, lower sampled signal ("measurement")
cnt = 0
for ii in range(0, N1, m):
    # R_noise[:,:,cnt] = expSO3(phi[:,ii]+sigma*np.random.randn(3))
    R_noise[:, :, cnt] = expSO3(sigma * np.random.randn(3)) @ R[:, :, ii]
    cnt += 1

# Finite differencing from noisy lower sampled signal ("measurement"):
for ii in range(1, N2 - 1):
    omg_FD[:, ii] = vee(1 / (2 * dt2) * (logm(R_noise[:, :, ii + 1] @ np.linalg.inv(R_noise[:, :, ii])) -
                                         logm(R_noise[:, :, ii - 1] @ np.linalg.inv(R_noise[:, :, ii]))))

for ii in range(1, N2 - 1):
    domg_FD[:, ii] = 1 / (2 * dt2) * (omg_FD[:, ii + 1] - omg_FD[:, ii - 1])

# Applying the Savitzky-Golay filter
# Now, from the noisy lower sampled data, we want to get back the estimated
# rotation matrix, angular velocity and angular acceleration
R_est, omg_est, domg_est, t3 = sgolayfiltSO3(R_noise, p, n, 1 / dt2)

# Computing errors, plotting results
# Time indices of R for which we have a measurement:
plt.close('all')

# Find matching indices between t1 and t2
# In MATLAB: tR1 = find(ismember(t1,t2)==1)
tR1 = []
tR1_idx_in_t2 = []  # Store the corresponding indices in t2
for i, t in enumerate(t1):
    matches = np.where(np.abs(t - t2) < 1e-10)[0]
    if len(matches) > 0:
        tR1.append(i)
        tR1_idx_in_t2.append(matches[0])

# Find matching indices between t1 and t3
# In MATLAB: tR2 = find(ismember(t1,t3)==1)
tR2 = []
tR2_idx_in_t3 = []  # Store the corresponding indices in t3
for i, t in enumerate(t1):
    matches = np.where(np.abs(t - t3) < 1e-10)[0]
    if len(matches) > 0:
        tR2.append(i)
        tR2_idx_in_t3.append(matches[0])

# Preallocate error arrays
eR_meas = np.zeros((3, 3, len(tR1)))
NeR_meas = np.zeros(len(tR1))
eomg_FD = np.zeros((3, len(tR1)))
edomg_FD = np.zeros((3, len(tR1)))

# Compute errors for measurements
for ii in range(len(tR1)):
    t1_idx = tR1[ii]
    t2_idx = tR1_idx_in_t2[ii]
    eR_meas[:, :, ii] = logm(R[:, :, t1_idx] @ np.linalg.inv(R_noise[:, :, t2_idx]))
    NeR_meas[ii] = np.linalg.norm(eR_meas[:, :, ii])
    eomg_FD[:, ii] = omg_FD[:, t2_idx] - omg[:, t1_idx]
    edomg_FD[:, ii] = domg_FD[:, t2_idx] - domg[:, t1_idx]

# Preallocate error arrays for estimates
eR_est = np.zeros((3, 3, len(tR2)))
NeR_est = np.zeros(len(tR2))
eomg_est = np.zeros((3, len(tR2)))
edomg_est = np.zeros((3, len(tR2)))

# Compute errors for estimates
for ii in range(len(tR2)):
    t1_idx = tR2[ii]
    t3_idx = tR2_idx_in_t3[ii]
    eR_est[:, :, ii] = logm(R[:, :, t1_idx] @ np.linalg.inv(R_est[:, :, t3_idx]))
    NeR_est[ii] = np.linalg.norm(eR_est[:, :, ii])
    eomg_est[:, ii] = omg_est[:, t3_idx] - omg[:, t1_idx]
    edomg_est[:, ii] = domg_est[:, t3_idx] - domg[:, t1_idx]

Eomg_FD = np.linalg.norm(eomg_FD, axis=0)
Eomg_est = np.linalg.norm(eomg_est, axis=0)
Edomg_FD = np.linalg.norm(edomg_FD, axis=0)
Edomg_est = np.linalg.norm(edomg_est, axis=0)

# Mean error in rotation
mean_ER_est = np.mean(NeR_est)
mean_ER_meas = np.mean(NeR_meas)

print(f'Mean rotation error measured: {mean_ER_meas} rad')
print(f'Mean rotation error SG-estimate: {mean_ER_est} rad')

# Mean errors in velocity
mean_Eomg_FD = np.nanmean(Eomg_FD)
mean_Eomg_est = np.nanmean(Eomg_est)

print(f'Mean velocity error finite differencing: {mean_Eomg_FD} rad/s')
print(f'Mean velocity error SG-estimate: {mean_Eomg_est} rad/s')

# Mean errors in acceleration
mean_Edomg_FD = np.nanmean(Edomg_FD)
mean_Edomg_est = np.nanmean(Edomg_est)

print(f'Mean acceleration error finite differencing: {mean_Edomg_FD} rad/s^2')
print(f'Mean acceleration error SG-estimate: {mean_Edomg_est} rad/s^2')

# Check if figures directory exists, if not, it will create one.
if not os.path.exists('figures'):
    os.makedirs('figures')

# Plot the orientation error
plt.figure(figsize=(8, 5))
plt.plot(t2, NeR_meas, label=r'$e_{\widetilde{\mathbf{R}}}$')
plt.plot(t3, NeR_est, label=r'$e_{\widehat{\mathbf{R}}}$')
plt.xlim(0, 2)
plt.xlabel('Time [s]')
plt.ylabel('Orientation error [rad]')
plt.legend(ncol=2, fontsize=9)
plt.grid(True)

if doSave:
    plt.savefig('figures/norm_eR.pdf')

# Plot rotation on sphere
fig = plt.figure(figsize=(12, 4))

# Create 3D subplots
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

# Create a sphere for each subplot
u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

# Plot sphere and data for x-component
ax1.plot_surface(x, y, z, color='gray', alpha=0.3)
ax1.plot3D([R[0, 0, 0]], [R[1, 0, 0]], [R[2, 0, 0]], '*', markersize=10, color='k', linewidth=2)
ax1.plot3D(R[0, 0, 1:1001], R[1, 0, 1:1001], R[2, 0, 1:1001], color=[0.9290, 0.6940, 0.1250], linewidth=1.2)
ax1.plot3D(R_est[0, 0, :], R_est[1, 0, :], R_est[2, 0, :], color=[0.8500, 0.3250, 0.0980], linewidth=1.2)
ax1.plot3D(R_noise[0, 0, :], R_noise[1, 0, :], R_noise[2, 0, :], color=[0, 86 / 255, 140 / 255], linewidth=1.2)
ax1.view_init(-122, 31)
ax1.set_axis_off()
ax1.text(0, 0, -1.5, 'x')

# Plot sphere and data for y-component
ax2.plot_surface(x, y, z, color='gray', alpha=0.3)
ax2.plot3D([R[0, 1, 0]], [R[1, 1, 0]], [R[2, 1, 0]], '*', markersize=10, color='k', linewidth=2)
ax2.plot3D(R[0, 1, 1:1001], R[1, 1, 1:1001], R[2, 1, 1:1001], color=[0.9290, 0.6940, 0.1250], linewidth=1.2)
ax2.plot3D(R_est[0, 1, :], R_est[1, 1, :], R_est[2, 1, :], color=[0.8500, 0.3250, 0.0980], linewidth=1.2)
ax2.plot3D(R_noise[0, 1, :], R_noise[1, 1, :], R_noise[2, 1, :], color=[0, 86 / 255, 140 / 255], linewidth=1.2)
ax2.view_init(90, 32)
ax2.set_axis_off()
ax2.text(0, 0, -1.5, 'y')

# Plot sphere and data for z-component
ax3.plot_surface(x, y, z, color='gray', alpha=0.3)
ax3.plot3D([R[0, 2, 0]], [R[1, 2, 0]], [R[2, 2, 0]], '*', markersize=10, color='k', linewidth=2)
g1, = ax3.plot3D(R[0, 2, 1:1001], R[1, 2, 1:1001], R[2, 2, 1:1001], color=[0.9290, 0.6940, 0.1250], linewidth=1.2)
g2, = ax3.plot3D(R_est[0, 2, :], R_est[1, 2, :], R_est[2, 2, :], color=[0.8500, 0.3250, 0.0980], linewidth=1.2)
g3, = ax3.plot3D(R_noise[0, 2, :], R_noise[1, 2, :], R_noise[2, 2, :], color=[0, 86 / 255, 140 / 255], linewidth=1.2)
ax3.view_init(0, 0)
ax3.set_axis_off()
ax3.text(0, 0, -1.5, 'z')

# Legend
plt.figlegend([g1, g3, g2], [r'$\mathbf{R}$', r'$\widetilde{\mathbf{R}}$', r'$\widehat{\mathbf{R}}$'],
              ncol=3, loc='upper center', fontsize=9)

plt.tight_layout()

if doSave:
    plt.savefig('figures/Rotation.pdf')

# Plot angular velocity
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# x-component
g1, = ax1.plot(t2, omg_FD[0, :], color=[0, 0.4470, 0.7410, 0.6], alpha=0.6)
g2, = ax1.plot(t3, omg_est[0, :], linewidth=1.5)
g3, = ax1.plot(t1, omg[0, :], linewidth=1.5)
ax1.set_xlim(0, te)
ax1.set_ylim(-20, 20)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Angular velocity [rad/s]')
ax1.text(1, -18, 'x-component', fontsize=9)
ax1.grid(True)

# y-component
ax2.plot(t2, omg_FD[1, :], color=[0, 0.4470, 0.7410, 0.6], alpha=0.6)
ax2.plot(t3, omg_est[1, :], linewidth=1.5)
ax2.plot(t1, omg[1, :], linewidth=1.5)
ax2.set_xlim(0, te)
ax2.set_ylim(-20, 20)
ax2.set_xlabel('Time [s]')
ax2.text(1, -18, 'y-component', fontsize=9)
ax2.grid(True)

# z-component
ax3.plot(t2, omg_FD[2, :], color=[0, 0.4470, 0.7410, 0.6], alpha=0.6)
ax3.plot(t3, omg_est[2, :], linewidth=1.5)
ax3.plot(t1, omg[2, :], linewidth=1.5)
ax3.set_xlim(0, te)
ax3.set_ylim(-20, 20)
ax3.set_xlabel('Time [s]')
ax3.text(1, -18, 'z-component', fontsize=9)
ax3.grid(True)

# Legend
plt.figlegend([g3, g1, g2], [r'Analytical solution $\mathbf{\omega}$',
                             r'Finite differencing $\breve{\mathbf{\omega}}$',
                             r'Savitzky-Golay $\widehat{\mathbf{\omega}}$'],
              ncol=3, loc='upper center', fontsize=9)

plt.tight_layout()

if doSave:
    plt.savefig('figures/omg.pdf')

# Plot angular acceleration
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# x-component
g1, = ax1.plot(t2, domg_FD[0, :], color=[0, 0.4470, 0.7410, 0.6], alpha=0.6)
g2, = ax1.plot(t3, domg_est[0, :], linewidth=1.5)
g3, = ax1.plot(t1, domg[0, :], linewidth=1.5)
ax1.set_xlim(0, te)
ax1.set_ylim(-200, 200)
ax1.set_yticks([-200, -150, -100, -50, 0, 50, 100, 150, 200])
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Angular acceleration [rad/s$^2$]')
ax1.text(1, -180, 'x-component', fontsize=9)
ax1.grid(True)

# y-component
ax2.plot(t2, domg_FD[1, :], color=[0, 0.4470, 0.7410, 0.6], alpha=0.6)
ax2.plot(t3, domg_est[1, :], linewidth=1.5)
ax2.plot(t1, domg[1, :], linewidth=1.5)
ax2.set_xlim(0, te)
ax2.set_ylim(-200, 200)
ax2.set_yticks([-200, -150, -100, -50, 0, 50, 100, 150, 200])
ax2.set_xlabel('Time [s]')
ax2.text(1, -180, 'y-component', fontsize=9)
ax2.grid(True)

# z-component
ax3.plot(t2, domg_FD[2, :], color=[0, 0.4470, 0.7410, 0.6], alpha=0.6)
ax3.plot(t3, domg_est[2, :], linewidth=1.5)
ax3.plot(t1, domg[2, :], linewidth=1.5)
ax3.set_xlim(0, te)
ax3.set_ylim(-200, 200)
ax3.set_yticks([-200, -150, -100, -50, 0, 50, 100, 150, 200])
ax3.set_xlabel('Time [s]')
ax3.text(1, -180, 'z-component', fontsize=9)
ax3.grid(True)

# Legend
plt.figlegend([g3, g1, g2], [r'Analytical solution $\dot{\mathbf{\omega}}$',
                             r'Finite differencing $\breve{\dot{\mathbf{\omega}}}$',
                             r'Savitzky-Golay $\widehat{\dot{\mathbf{\omega}}}$'],
              ncol=3, loc='upper center', fontsize=9)

plt.tight_layout()

if doSave:
    plt.savefig('figures/domg.pdf')

plt.show()