import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
import time

L = 2e4
h_estuary = 6
h_river = 3
b_river = 500
b_estuary = 2000
Q_river = 800.0 

tide_amplitude = 1
tide_period = 12 * 3600 + 25 * 60

n_manning = 0.01

nx = 1024
dx = L / nx
x = np.linspace(0, L, nx)

g = 9.81
cfl = 0.1
dt = cfl * dx / (np.sqrt(g * h_estuary + tide_amplitude))
tmax = 0.1*tide_period
t = 0.0


b = b_river + (b_estuary - b_river) * np.exp(-4 * x / L)
zb = -h_river + (h_river - h_estuary) * (1 - x / L)**2
#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
# ax1.plot(x/1000, b, 'b-', linebidth=2)
# ax1.set_ylabel("Largeur (m)")
# ax1.set_title("Évolution de la largeur de la Garonne (estuaire → Podensac)")
# ax1.grid(True)

# ax2.plot(x/1000, zb, 'r-', linebidth=2)
# ax2.set_ylabel("Profondeur (m)")
# ax2.set_xlabel("Distance (km)")
# ax2.set_title("Évolution de la profondeur")
# ax2.grid(True)

# plt.tight_layout()  # Interpolation de u aux points entiers
# plt.show()

@njit(fastmath=True, cache=True)
def compute_tidal_bore(nframes, nx, dx, dt, g, rho, b, zb, h_estuary, tide_amplitude, tide_period, n_manning=0.001):
    h = np.zeros(nx)
    u = np.zeros(nx+1) #demi indice
    q = np.zeros(nx)

    h[:] = h_estuary
    u[:] = 0.0
    q[:] = 0.0
    
    all_eta = np.zeros((nframes, nx))
    all_u = np.zeros((nframes, nx+1))
    all_time = np.zeros(nframes)
    
    t = 0.0

    for frame in range(nframes):

        h[0] = h_estuary + tide_amplitude * np.sin(2 * np.pi * t / tide_period)
        u[0] = np.sqrt(g * h[0])
        h[1] = h[0]
        h[-1] = 2 * h[-2] - h[-3]
        u[-1] = u[-2]
        
        for j in range(1, nx):
            h_phalf = 0.5 * (h[j] + h[j+1])
            h_mhalf = 0.5 * (h[j] + h[j-1])
            b_phalf = 0.5 * (b[j] + b[j+1])
            b_mhalf = 0.5 * (b[j] + b[j-1])
            h[j] = h[j] - dt/dx * ((h_phalf * u[j])/b_phalf * (b_phalf - b_mhalf) + (h_phalf*u[j] - h_mhalf*u[j-1]))

        for j in range(1, nx):
            u[j] = u[j] - dt/dx * (g*(h[j+1] - h[j]) + (u[j] - u[j-1]))- dt * g * n_manning**2 * u[j] * abs(u[j]) / h[j]**(4/3)
        
        all_eta[frame, :] = h - h_estuary
        all_u[frame, :] = u[:]
        all_time[frame] = t
        t += dt
        
    return all_eta, all_u, all_time


start = time.time()

nframes = int(tmax/dt)
print(nframes)

all_eta, all_u, all_time = compute_tidal_bore(nframes, nx, dx, dt, g, 1000, b, zb, h_estuary, tide_amplitude, tide_period, n_manning)
print (f"Temps de calcul : {time.time() - start:.2f} secondes")

all_u = all_u[:, :-1]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
ax1.set_xlim(0, L / 1000)
ax1.set_ylim(np.min(zb), np.max(all_eta) + h_estuary)
ax1.set_xlabel("Distance (km)")
ax1.set_ylabel("Elevation (m)")
ax1.axhline(y=-h_estuary, color='k', linestyle='--', label='Fond de la mer')
ax1.grid(True, alpha=0.3)

ax2.set_xlim(0, L / 1000)
ax2.set_ylim(np.min(all_u), np.max(all_u))
ax2.set_xlabel("Distance (km)")
ax2.set_ylabel("Vitesse (m/s)")
ax2.grid(True, alpha=0.3)

def update(frame):
    line_water.set_ydata(all_eta[frame, :])
    line_velocity.set_ydata(all_u[frame, :])
    ax1.set_title(f"Mascaret dans la Garonne - t = {all_time[frame] / 60:.2f} min")
    return line_water, line_velocity

line_water, = ax1.plot(x / 1000, all_eta[0, :], 'b-', lw=2, label="Surface")
line_velocity, = ax2.plot(x / 1000, all_u[0, :], 'r-', lw=2, label="Vitesse")
ax1.legend()
ax2.legend()

step = 20
anim = FuncAnimation(fig, update, frames=range(0, int(tmax / dt), step), interval=50, blit=False)
plt.tight_layout()
plt.show()