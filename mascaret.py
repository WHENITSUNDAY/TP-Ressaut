import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
import time

L = 2e4
h_estuary = 10
h_river = 3
b_river = 500
b_estuary = 2000
Q_river = 800.0 

tide_amplitude = 2
tide_period = 12 * 3600 + 25 * 60

n_manning = 0.001

nx = 500
dx = L / nx
x = np.linspace(0, L, nx)

g = 9.81
cfl = 0.2
dt = cfl * dx / (np.sqrt(g * h_estuary + tide_amplitude))
tmax = 1000
t = 0.0


b = b_river + (b_estuary - b_river) * np.exp(-4 * x / L)
zb = -h_river + (h_river - h_estuary) * (1 - x / L)**2
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))


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
    u = np.zeros(nx+1)
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
        h[1] = h[0]
        u[0] = np.sqrt(g * h[0])

        h[-1] = h[-2]
        u[-1] = u[-2]
        print(h[1], u[1])
        for j in range(1, nx-1):
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

nframes = int(tmax/dt)
start = time.time()
print(nframes)
all_eta, all_u, all_time = compute_tidal_bore(nframes, nx, dx, dt, g, 1000, b, zb, h_estuary, tide_amplitude, tide_period, n_manning)
print (f"Temps de calcul : {time.time() - start:.2f} secondes")


fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, L / 1000)
ax.set_ylim(np.min(zb), np.max(all_eta) + h_estuary)
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Elevation (m)")
ax.grid(True, alpha=0.3)

def update(frame):
    line_water.set_ydata(all_eta[frame, :])
    ax.set_title(f"Garonne Tidal Bore - t = {all_time[frame] / 3600:.2f} h")
    return line_water,

line_water, = ax.plot(x / 1000, all_eta[0, :], 'b-', lw=2, label="Surface")
ax.legend()

step = 100
anim = FuncAnimation(fig, update, frames=range(0, int(tmax / dt), step), interval=50, blit=False)

plt.show()
