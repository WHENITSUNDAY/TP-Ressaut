import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
import time
from matplotlib.patches import Polygon
"""Le but de code est de simuler numériquement le phénomène de Mascaret se produisant dans une rivière. Plus particulièrement, on considérera une forme en entonnoir 
pour la rivière, avec une pente quadratique. Cela permettra d'accentuer le ressaut etde mettre en avant le phénomène ondulatoire du Mascaret. 
Le schéma se base en partie sur cela d'article "Numerical Simulation of Tidal Bore Bono at Kampar River (JAFM, A. C. Bayu et al)"""

show_velocity = False      #True si on veut afficher le profil de la vitesse pendant la simulation
show_river = False          #True si on veut afficher la forme de la rivière en largeur
show_bathymetry = False     #True si on veut afficher la bathymétrie (profil de la profondeur) de la rivière
bathymetry_shape = 3       #1 pour constant, 2 pour linéaire, 3 pour quadratique (le plus réaliste)
obstacle_shape = 1         #1 pour un créneau, 2 pour une rampe, 3 pour une bosse (gausienne)

L = 80000             #Longueur caractéristique de l'écoulement
h_estuary = 6       #Profondeur en aval (gauche)
h_river = 3         #Profondeur en amont (droite) -> Utilisée seulement si profondeur non constante
b_river = 200       #Larguer de la rivière en amont
b_estuary = 13000   #Largeur de la rivière en aval (estuaire)

tide_amplitude = 2                  #Amplitude de la marée
tide_period = 12 * 3600 + 25 * 60   #Période de l'onde de marée -> 12h25m



nx = 1024
dx = L / nx
x = np.linspace(0, L, nx)
g = 9.81
cfl = 0.1
dt = cfl * dx / (np.sqrt(g * h_estuary + tide_amplitude))
c = np.sqrt(g * h_estuary)   # vitesse d'onde (approximation mascaret)
n_manning = 0.001       #Coefficient de frottement de manning
tmax = 1.2 * L / c  



#Création du profil de profondeur de la rivière : 
if bathymetry_shape == 1: #Constant
    h_static = h_estuary
    zb = np.full(nx, -h_static)

elif bathymetry_shape == 2: #Linéaire (pour les élèves)
    zb = 0

elif bathymetry_shape == 3: #Quadratique
    zb = -h_river + (h_river - h_estuary) * (1 - x / L)**2

    
if obstacle_shape == 1:
    slot_height = 3
    slot_width = 5000
    slot_center = L/2
    start_idx = int((slot_center - slot_width/2) / L * nx)
    end_idx = int((slot_center + slot_width/2) / L * nx)
    zb[start_idx:end_idx] += slot_height  

elif obstacle_shape == 2:
    ramp_width = 10000
    ramp_x0 = L/2
    ramp_height = 3
    start_idx = int((ramp_x0) / L * nx)
    end_idx = int((ramp_x0 + ramp_width) / L * nx)

    for i in range(start_idx, end_idx):

        zb[i] += ramp_height * (i - start_idx) / (end_idx - start_idx)

elif obstacle_shape == 3:
    bump_height = 5
    bump_width = 20000
    bump_center = L/2
    zb = zb + bump_height * np.exp(-(x - bump_center)**2 / (2 * (bump_width / 4)**2))


#Création du profil de largeur de la rivière :
b = b_river + (b_estuary - b_river) * np.exp(-8 * x / L)

if show_river :
    fig, ax = plt.subplots(figsize=(12, 4))
    river_left = -b / 2
    river_right = b / 2
    river_coords = np.vstack([np.column_stack((x / 1000, river_left)), np.column_stack((x[::-1] / 1000, river_right[::-1]))])
    river_polygon = Polygon(river_coords, closed=True, color='dodgerblue', alpha=0.8, linewidth=0)
    ax.add_patch(river_polygon)
    land_width = 1.2 * np.max(b)
    ax.set_ylim(-land_width, land_width)
    ax.set_xlim(0, L/1000)
    ax.fill_between(x / 1000, river_right, land_width, color='darkgreen', alpha=0.7, linewidth=0)
    ax.fill_between(x / 1000, -land_width, river_left, color='darkgreen', alpha=0.7, linewidth=0)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Largeur relative (m)")
    ax.set_title("Forme de la rivière (vue en plan)")
    
    plt.tight_layout()
    plt.show()

if show_bathymetry :
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_ylim(-h_estuary - 2, 0)
    ax.set_xlim(0, L/1000)
    ax.plot(x / 1000, np.zeros_like(x), color='deepskyblue', lw=2, label="Surface de l'eau (z=0)")
    ax.plot(x / 1000, zb, color='saddlebrown', lw=2, label="Lit de la rivière")
    ax.fill_between(x / 1000, zb, 0, where=zb < 0, color='dodgerblue', alpha=0.7)
    ax.fill_between(x / 1000, zb, np.min(zb)-2, color='saddlebrown', alpha=0.6)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Bathymétrie de la rivière (surface et profondeur)")
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

@njit(fastmath=True, cache=True)
def compute_tidal_bore(nframes, nx, dx, dt, g, b, zb, h_estuary, tide_amplitude, tide_period, n_manning=0.001):
    h = np.zeros(nx)
    u = np.zeros(nx+1) #demi indice (pour avoir un schéma conservatif)
    q = np.zeros(nx)

    h[:] = -zb
    u[:] = 0.0
    q[:] = 0.0
    
    h[:] = np.maximum(0, h[:])
    all_eta = np.zeros((nframes, nx))
    all_u = np.zeros((nframes, nx+1))
    all_time = np.zeros(nframes)
    
    t = 0.0
    eps = 1e-4
    for frame in range(nframes):

        h[0] = h_estuary + tide_amplitude * np.sin(2 * np.pi * t / tide_period)
        u[0] = u[1] + (h[0] - h[1]) * np.sqrt(g / h[0]) 

        for j in range(1, nx-1):
            h_phalf = 0.5 * (h[j] + h[j+1]) if h[j+1] > 0 else 0 
            h_mhalf = 0.5 * (h[j] + h[j-1]) if h[j-1] > 0 else 0 
            b_phalf = 0.5 * (b[j] + b[j+1])
            b_mhalf = 0.5 * (b[j] + b[j-1])
            h[j] = h[j] - dt/dx * ((h_phalf * u[j])/b_phalf * (b_phalf - b_mhalf) + (h_phalf*u[j] - h_mhalf*u[j-1]))

            if h[j] <= 0:
                h[j] = 0

        for j in range(1, nx-1):

            if zb[j] >= h[j] +zb[j]:
                h[j] = 0
                u[j] = 0
                continue

            if u[j] >= 0: 
                u[j] = u[j] - dt/dx * (g*(h[j+1] - h[j] + zb[j+1] - zb[j]) + u[j]*(u[j] - u[j-1]))- dt * g * n_manning**2 * u[j] * abs(u[j]) / (h[j] + eps)**(4/3)
            elif u[j] < 0:
                u[j] = u[j] - dt/dx * (g*(h[j+1] - h[j] + zb[j+1] - zb[j]) + u[j]*(u[j+1] - u[j]))- dt * g * n_manning**2 * u[j] * abs(u[j]) / (h[j] + eps)**(4/3)


        h[-1] = h_river
        u[-1] = 0
        all_eta[frame, :] = h + zb
        all_u[frame, :] = u[:]
        all_time[frame] = t
        t += dt
        
    return all_eta, all_u, all_time


start = time.time()
nframes = int((tmax)/dt)

all_eta, all_u, all_time = compute_tidal_bore(nframes, nx, dx, dt, g, b, zb, h_estuary, tide_amplitude, tide_period, n_manning)
print (f"Temps de calcul des solutions: {time.time() - start:.2f} secondes")

all_u = all_u[:, :-1]

start_time = 0
start_frame = int(start_time / dt)
nframes = nframes - start_frame
all_eta = all_eta[start_frame:, :]
all_u = all_u[start_frame:, :-1]
all_time = all_time[start_frame:]

fig, ax = plt.subplots(figsize=(14, 8))

ax.set_xlim(0, L / 1000)
ax.set_ylim(np.min(zb) - 2, tide_amplitude * 2 )
ax.set_xlabel("Distance de l'estuaire vers Podensac (km)", fontsize=12, labelpad=10)
ax.set_ylabel("Élévation (m)", fontsize=12, labelpad=10)
ax.set_title(f"Propagation du mascaret - Temps: 0 min", fontsize=14, pad=20)
ax.grid(True, alpha=0.2)
ax.text(0.02, 0.55, "Estuaire de la Gironde\n (marée montante)", transform=ax.transAxes, fontsize=12, fontweight='bold', color='navy')
ax.text(0.85, 0.55, "Podensac", transform=ax.transAxes, fontsize=12, fontweight='bold', color='navy')
ax.plot(x / 1000, zb, color='saddlebrown', lw=2, label="Lit de la rivière")
ax.fill_between(x/1000, np.min(zb)-2, zb, color='saddlebrown', alpha=0.6)

water_surface, = ax.plot(x/1000, all_eta[0, :], color='skyblue', lw=2, label="Surface de l'eau")

if show_velocity:
    stride = max(1, nx // 50)
    x_quiver = x[::stride] / 1000
    eta_mid = (all_eta[0, ::stride] + zb[::stride]) / 2
    quiver = ax.quiver(x_quiver, eta_mid, all_u[0, ::stride], np.zeros_like(x_quiver), scale=150, color='royalblue', width=0.003)


water_polygon = Polygon(np.vstack([np.concatenate([x/1000, x[::-1]/1000]),np.concatenate([all_eta[0, :], zb[::-1]])]).T,color='dodgerblue', alpha=0.7, label='Eau')
ax.add_patch(water_polygon)
ax.legend(loc='upper left', framealpha=1)

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
    
    ax.set_title(f"Propagation du mascaret - Temps: {int(all_time[frame] // 60)} min")
    return water_polygon, water_surface

interval = 30
step = max(1, nframes // (30 * 1000 // interval))

fig.canvas.mpl_connect('key_press_event', on_press)
anim = FuncAnimation(fig, update, frames=range(0, nframes, step), interval=interval, blit=False, repeat=True)

plt.tight_layout()
plt.show()
