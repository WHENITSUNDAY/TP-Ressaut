import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
import time
from matplotlib.patches import Polygon
"""Le but de code est de simuler numériquement le phénomène de Tsunami, pour cela on utilise les équations de St Venant avec frottement.
La simulation en elle même s'inspire du scénario du séisme de Tohoku de 2011"""

show_velocity = False      #True si on veut afficher le profil de la vitesse pendant la simulation
show_bathymetry = True     #True si on veut afficher la bathymétrie (profil de la profondeur) de la rivière
protection_type = 0        #1 pour une protection type "mangrove, 0 sinon"
L = 60000             #Longueur caractéristique de l'écoulement
h_ocean = 2000       #Profondeur en aval (gauche)

magnitude = 8 #Magnitude du séisme !
quake_center = 0
quake_amplitude = 10**(0.4 * (magnitude - 6)) #Relation empirique (approximation)
print(quake_amplitude)
quake_width = 2000

n_manning_ocean = 0.007       #Coefficient de frottement de manning
n_manning_coast = 0.02
n_manning_mangrove = 1.0

nx = 2048
dx = L / nx
x = np.linspace(0, L, nx)
g = 9.81
cfl = 0.1
dt = cfl * dx / (np.sqrt(g * h_ocean + quake_amplitude))
c = np.sqrt(g * h_ocean)   # vitesse d'onde approximative
tmax = 1.5 * L / c  

n_manning = np.full(nx, n_manning_ocean)

zb = -h_ocean * (1 - (np.arange(nx)/(nx-1))**2)

h_transition = -200 #A partir de quelle profondeur on veut que le littoral commence
transition_idx = np.argmin(np.abs(zb - h_transition))
zb[transition_idx:] = h_transition - h_transition / (L - x[transition_idx]) * (x[transition_idx:] - x[transition_idx])

if protection_type == 0:
    n_manning[transition_idx:] = n_manning_coast

elif protection_type == 1:
    n_manning[transition_idx:] = n_manning_mangrove


if show_bathymetry :
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_ylim(-h_ocean - 2, 0)
    ax.set_xlim(0, L/1000)
    ax.plot(x / 1000, np.zeros_like(x), color='deepskyblue', lw=2, label="Surface de l'eau (z=0)")
    ax.plot(x / 1000, zb, color='saddlebrown', lw=2, label="Fond océanique")
    ax.fill_between(x / 1000, zb, 0, where=zb < 0, color='dodgerblue', alpha=0.7)
    ax.fill_between(x / 1000, zb, np.min(zb)-2, color='saddlebrown', alpha=0.6)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Bathymétrie de l'océan")
    transition_x = x[transition_idx] / 1000

    if protection_type == 0:
        ax.fill_between(x/1000, np.min(zb)-2, zb, where=(x >= x[transition_idx]), color='peru', alpha=0.4)
        ax.axvline(transition_x, color='darkorange', linestyle='--', lw=1.5, label=f'Littoral')
        ax.text(transition_x + 0.5, np.min(zb) + 5, "Littoral", ha='left', va='bottom', color='darkorange', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    if protection_type == 1:
        ax.fill_between(x/1000, np.min(zb)-2, zb, where=(x >= x[transition_idx]), color='forestgreen', alpha=0.2)
        ax.axvline(transition_x, color='forestgreen', linestyle='--', lw=1.5, label=f'Littoral (mangrove)')
        ax.text(transition_x + 0.5, np.min(zb) + 5, "Littoral (mangrove)", ha='left', va='bottom', color='forestgreen', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

@njit(fastmath=True, cache=False)
def compute_tsunami(nframes, nx, dx, dt, g, zb, quake_center, quake_width, quake_amplitude, n_manning):
    h = np.zeros(nx)
    u = np.zeros(nx+1) #demi indice (pour avoir un schéma conservatif)
    q = np.zeros(nx)

    h[:] = -zb
    u[:] = 0.0
    q[:] = 0.0

    h += quake_amplitude * np.exp(-(np.arange(nx) * dx - quake_center)**2 / (2 * quake_width**2))
    h[:] = np.maximum(0, h[:])
    all_eta = np.zeros((nframes, nx))
    all_u = np.zeros((nframes, nx+1))
    all_time = np.zeros(nframes)

    t = 0.0
    eps = 1e-6
    for frame in range(nframes):

        c = np.sqrt(g * max(h[0], eps))
        h[0] = h[-1]
        u[0] = u[1] + (u[0] - c) * dt/dx * (u[1] - u[0])

        for j in range(1, nx-1):
            h_phalf = 0.5 * (h[j] + h[j+1]) if h[j+1] > 0 else 0 
            h_mhalf = 0.5 * (h[j] + h[j-1]) if h[j-1] > 0 else 0 
            h[j] = h[j] - dt/dx * ((h_phalf*u[j] - h_mhalf*u[j-1]))

            if h[j] <= 0:
                h[j] = 0

        for j in range(1, nx-1):

            if zb[j] >= h[j] +zb[j]:
                h[j] = 0
                u[j] = 0
                continue

            if u[j] >= 0: 
                u[j] = u[j] - dt/dx * (g*(h[j+1] - h[j] + zb[j+1] - zb[j]) + u[j]*(u[j] - u[j-1]))- dt * g * n_manning[j]**2 * u[j] * abs(u[j]) / (h[j] + eps)**(4/3)
            elif u[j] < 0:
                u[j] = u[j] - dt/dx * (g*(h[j+1] - h[j] + zb[j+1] - zb[j]) + u[j]*(u[j+1] - u[j]))- dt * g * n_manning[j]**2 * u[j] * abs(u[j]) / (h[j] + eps)**(4/3)


        h[-1] = 0
        u[-1] = u[-2]
        all_eta[frame, :] = h + zb
        all_u[frame, :] = u[:]
        all_time[frame] = t
        t += dt
        
    return all_eta, all_u, all_time


start = time.time()
nframes = int((tmax)/dt)

all_eta, all_u, all_time = compute_tsunami(nframes, nx, dx, dt, g, zb, quake_center, quake_width, quake_amplitude, n_manning)
print (f"Temps de calcul des solutions: {time.time() - start:.2f} secondes")

all_u = all_u[:, :-1]

start_time = 0
start_frame = int(start_time / dt)
nframes = nframes - start_frame
all_eta = all_eta[start_frame:, :]
all_u = all_u[start_frame:, :-1]
all_time = all_time[start_frame:]

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_yscale('symlog', linthresh=20, linscale=1)
ax.set_xlim(0, L / 1000)
ax.set_ylim(np.min(zb) - 2, 30)
ax.set_xlabel("Distance du séisme jusqu'à la côte (km)", fontsize=12, labelpad=10)
ax.set_ylabel("Élévation (m)", fontsize=12, labelpad=10)
ax.set_title(f"Propagation du tsunami - Temps: 0 min", fontsize=14, pad=20)
ax.grid(True, alpha=0.2)
ax.text(0.02, 0.55, "Séisme de grande magnitude", transform=ax.transAxes, fontsize=12, fontweight='bold', color='navy')
transition_x = x[transition_idx] / 1000

if protection_type == 0:
    ax.fill_between(x/1000, np.min(zb)-2, zb, where=(x >= x[transition_idx]), color='peru', alpha=0.4)
    ax.axvline(transition_x, color='darkorange', linestyle='--', lw=1.5, label=f'Littoral')
    ax.text(transition_x + 0.5, np.min(zb) + 5, "Littoral", ha='left', va='bottom', color='darkorange', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

if protection_type == 1:
    ax.fill_between(x/1000, np.min(zb)-2, zb, where=(x >= x[transition_idx]), color='forestgreen', alpha=0.2)
    ax.axvline(transition_x, color='forestgreen', linestyle='--', lw=1.5, label=f'Littoral (mangrove)')
    ax.text(transition_x + 0.5, np.min(zb) + 5, "Littoral (mangrove)", ha='left', va='bottom', color='forestgreen', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

ax.plot(x / 1000, zb, color='saddlebrown', lw=2, label="Fond océanique")
ax.fill_between(x/1000, np.min(zb)-2, zb, color='saddlebrown', alpha=0.6)

water_surface, = ax.plot(x/1000, all_eta[0, :], color='skyblue', lw=2, label="Surface de l'eau")

if show_velocity:
    stride = max(1, nx // 50)
    x_quiver = x[::stride] / 1000
    eta_mid = (all_eta[0, ::stride] + zb[::stride]) / 2
    quiver = ax.quiver(x_quiver, eta_mid, all_u[0, ::stride], np.zeros_like(x_quiver), scale=150, color='royalblue', width=0.003)


water_polygon = Polygon(np.vstack([np.concatenate([x/1000, x[::-1]/1000]),np.concatenate([all_eta[0, :], zb[::-1]])]).T,color='dodgerblue', alpha=0.7, label='Eau')
ax.add_patch(water_polygon)
ax.legend(loc='lower left', framealpha=1)

paused = False

def on_press(event):
    global paused
    if event.key == ' ':
        paused = not paused
        if paused:
            anim.event_source.stop()
        else:
            anim.event_source.start()

def update(frame):
    water_polygon.set_xy(np.vstack([np.concatenate([x/1000, x[::-1]/1000]), 
                                  np.concatenate([all_eta[frame, :], zb[::-1]])]).T)
    water_surface.set_ydata(all_eta[frame, :])
    
    if show_velocity:
        eta_mid = (all_eta[frame, ::stride] + zb[::stride]) / 2
        quiver.set_offsets(np.column_stack([x_quiver, eta_mid]))
        quiver.set_UVC(all_u[frame, ::stride], np.zeros_like(x_quiver))
    
    ax.set_title(f"Propagation du tsunami - Temps: {int(all_time[frame] // 60)} min")
    return water_polygon, water_surface

interval = 30
step = max(1, nframes // (30 * 1000 // interval))

fig.canvas.mpl_connect('key_press_event', on_press)
anim = FuncAnimation(fig, update, frames=range(0, nframes, step), interval=interval, blit=False, repeat=True)

plt.tight_layout()
plt.show()
