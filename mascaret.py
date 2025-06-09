import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
import time
from matplotlib.patches import Polygon

"""Le but de code est de simuler numériquement le phénomène de Mascaret se produisant dans une rivière. Plus particulièrement, on considérera une forme en entonnoir 
pour la rivière, avec une pente quadratique. Cela permettra d'accentuer le ressaut etde mettre en avant le phénomène ondulatoire du Mascaret. 
Le schéma se base en partie sur cela d'article "Numerical Simulation of Tidal Bore Bono at Kampar River (JAFM, A. C. Bayu et al)"""

show_river = True          #True si on veut afficher la forme de la rivière en largeur
show_bathymetry = True     #True si on veut afficher la bathymétrie (profil de la profondeur) de la rivière

L = 8e4             #Longueur caractéristique de l'écoulement
h_estuary = 6       #Profondeur en aval (gauche)
h_river = 3         #Profondeur en amont (droite) -> Utilisée seulement si profondeur non constante
b_river = 200       #Larguer de la rivière en amont
b_estuary = 12000   #Largeur de la rivière en aval (estuaire)
Q_river = 800.0     #Débit de la rivière en amont (pas utilisé pour l'instant)

tide_amplitude = 2                  #Amplitude de la marée
tide_period = 12 * 3600 + 25 * 60   #Période de l'onde de marée -> 12h25m

n_manning = 0.001       #Coefficient de frottement de manning

nx = 1024
dx = L / nx
x = np.linspace(0, L, nx)

g = 9.81
cfl = 0.01
dt = cfl * dx / (np.sqrt(g * h_estuary + tide_amplitude))
c = np.sqrt(g * h_estuary)   # vitesse d'onde (approximation mascaret)
tmax = 1.2 * L / c  



#Création du profil de profondeur de la rivière : 
zb = -h_river + (h_river - h_estuary) * (1 - x / L)**2
#zb = np.full(nx, -h_estuary)
#zb[int(nx/2):int(nx/2) + 20] += 4
#zb = -h_estuary + 2 * np.exp(-(x - L/2)**2 / (2 * (5000 / 4)**2))

#Création du profil de largeur de la rivière :
b = b_river + (b_estuary - b_river) * np.exp(-8 * x / L)

if show_river :
    fig, ax = plt.subplots(figsize=(12, 4))
    river_left = -b / 2
    river_right = b / 2
    river_coords = np.vstack([np.column_stack((x / 1000, river_left)), np.column_stack((x[::-1] / 1000, river_right[::-1]))
    ])
    river_polygon = Polygon(river_coords, closed=True, color='royalblue', alpha=0.8, linewidth=0)
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
    ax.set_ylim(-h_estuary, 0)
    ax.set_xlim(0, L/1000)
    ax.plot(x / 1000, np.zeros_like(x), color='deepskyblue', lw=2, label="Surface de l'eau (z=0)")
    ax.plot(x / 1000, zb, color='saddlebrown', lw=2, label="Lit de la rivière")
    ax.fill_between(x / 1000, zb, 0, where=zb < 0, color='deepskyblue', alpha=0.7)
    ax.fill_between(x / 1000, zb, np.min(zb)-1, color='saddlebrown', alpha=0.6)
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
    
    all_eta = np.zeros((nframes, nx))
    all_u = np.zeros((nframes, nx+1))
    all_time = np.zeros(nframes)
    
    t = 0.0

    for frame in range(nframes):

        h[0] = h_estuary + tide_amplitude * np.sin(2 * np.pi * t / tide_period)
        c_in = np.sqrt(g * h[0])
        c1 = np.sqrt(g * h[1])
        u[0] = u[1] + 2 * (c_in - c1)
        
        
        for j in range(1, nx-1):
            h_phalf = 0.5 * (h[j] + h[j+1])
            h_mhalf = 0.5 * (h[j] + h[j-1])
            b_phalf = 0.5 * (b[j] + b[j+1])
            b_mhalf = 0.5 * (b[j] + b[j-1])
            h[j] = h[j] - dt/dx * ((h_phalf * u[j])/b_phalf * (b_phalf - b_mhalf) + (h_phalf*u[j] - h_mhalf*u[j-1]))

        for j in range(1, nx-1):
            if h[j] + zb[j] != 0 :
                u[j] = u[j] - dt/dx * (g*(h[j+1] - h[j] + zb[j+1] - zb[j]) + u[j]*(u[j] - u[j-1]))- dt * g * n_manning**2 * u[j] * abs(u[j]) / h[j]**(4/3)

        h[-1] = 2 * h[-2] - h[-3]
        u[-2] = 2 * u[-3] - u[-4]
        u[-1] = 2 * u[-2] - u[-3]
        all_eta[frame, :] = h + zb
        all_u[frame, :] = u[:]
        all_time[frame] = t
        t += dt
        
    return all_eta, all_u, all_time


start = time.time()
nframes = int((tmax)/dt)

all_eta, all_u, all_time = compute_tidal_bore(nframes, nx, dx, dt, g, b, zb, h_estuary, tide_amplitude, tide_period, n_manning)
print (f"Temps de calcul : {time.time() - start:.2f} secondes")

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
ax.text(0.02, 0.85, "Estuaire de la Gironde\n (marée montante)", transform=ax.transAxes, fontsize=12, fontweight='bold', color='navy')
ax.text(0.85, 0.85, "Podensac", transform=ax.transAxes, fontsize=12, fontweight='bold', color='navy')
ax.axhline(y=-h_estuary, color='k', linestyle='--', linewidth=1, label='Fond marin')
ax.fill_between(x/1000, np.min(zb)-2, zb, color='saddlebrown', alpha=0.7, label='Lit de la rivière')
water_surface, = ax.plot(x/1000, all_eta[0, :], color='deepskyblue', lw=2, label="Surface de l'eau")

def update(frame):
    water_polygon.set_xy(np.vstack([np.concatenate([x/1000, x[::-1]/1000]), np.concatenate([all_eta[frame, :], zb[::-1]])]).T)
    water_surface.set_ydata(all_eta[frame, :])
    time = int(all_time[frame] // 60)
    ax.set_title(f"Propagation du mascaret - Temps: {time} min", fontsize=14, pad=20)
    return water_polygon

water_polygon = Polygon(np.vstack([np.concatenate([x/1000, x[::-1]/1000]),np.concatenate([all_eta[0, :], zb[::-1]])]).T,color='dodgerblue', alpha=0.7, label='Eau')
ax.add_patch(water_polygon)
ax.legend(loc='upper left', framealpha=1)
plt.tight_layout()

interval = 30
step = max(1, nframes // (30 * 1000 // interval))

anim = FuncAnimation( fig, update,frames=range(0, nframes, step),interval=interval,blit=False)

plt.show()
