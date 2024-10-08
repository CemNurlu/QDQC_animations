from matplotlib import pyplot as plt
from matplotlib import animation as anim
import qutip
import numpy as np
from Scripts.util.bloch import bloch_vector, rot_matrix
from tqdm import tqdm
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec



size = 10

fig, axes = plt.subplots(1,2, figsize=(size*2, size), gridspec_kw={'width_ratios': [1, 1]})
axes[0].remove()
axes[1].remove()
ax_sphere0 = fig.add_subplot(1,2,1, projection="3d", azim=120, elev=30)
ax_sphere1 = fig.add_subplot(1,2,2, projection="3d", azim=-60, elev=30)
# ax_plot = axes[1]


# ax_sphere.axis("off")

if True:
    sphere0 = qutip.Bloch(axes=ax_sphere0)

    sphere0.point_marker = ["o"]
    sphere0.point_color = ["red"]
    sphere0.vector_color = ["red", "blue"]
    sphere0.point_marker = ["o"]
    sphere0.point_color = ["blue"]
    sphere0.vector_width = 6

    sphere1 = qutip.Bloch(axes=ax_sphere1)

    sphere1.point_marker = ["o"]
    sphere1.point_color = ["red"]
    sphere1.vector_color = ["red", "blue"]
    sphere1.point_marker = ["o"]
    sphere1.point_color = ["blue"]
    sphere1.vector_width = 6


N_time = 300
t = np.arange(N_time)

theta_rabi = np.pi / 4
phi_rabi = np.linspace(0, 4*np.pi, N_time)
h_rabi = bloch_vector(theta_rabi, phi_rabi)

azims = np.mod(phi_rabi * 180/np.pi + 60, 360)

theta_0_rot_frame = np.pi/10
theta_end_rot_frame = np.pi/10

if theta_0_rot_frame == theta_end_rot_frame:
   tau = 0
else:
   tau = np.log(theta_end_rot_frame/theta_0_rot_frame)/t[-1]

theta_rot_frame = theta_0_rot_frame*np.exp(tau*t)
phi_rot_frame = np.linspace(0, 17*np.pi, N_time)
spin = bloch_vector(theta_rot_frame, phi_rot_frame)

z_hat = np.array([0,0,1])

fixed_r = True
if not fixed_r:
   z0 = spin[0,2].copy()
   spin[:,2] = z0

# Rotate down to h_rabi_theta
rot_axis_0 = np.cross(z_hat, h_rabi[0,:])
rot_matrix_0 = rot_matrix(rot_axis_0, theta_rabi) 

spin = np.einsum("kj, ij -> ki", spin, rot_matrix_0)
# spin = np.einsum("ij, kj -> ki", rot_matrix0, spin)

# Rotate in rotating frame
# rot_matrix_1.shape is (N_time, 3, 3)

rot_matrix_1 = rot_matrix(z_hat, phi_rabi)
spin = np.einsum("tij, tj -> ti", rot_matrix_1, spin)


tail = -1


print(azims)

def init():
    return ax_sphere0, ax_sphere1

def animate(i):

    sphere0.vectors = []
    sphere0.add_vectors([h_rabi[i], spin[i]])
    ax_sphere0.azim = azims[i]
    sphere0.make_sphere()

    sphere1.vectors=[]
    sphere1.add_vectors([h_rabi[i], spin[i]])

    sphere1.points = []


    if i < tail or tail == -1:
        points = spin[:i+1]
    else:
        points = spin[i+1-tail:i+1]
    if i > 0:
        sphere1.add_points([points[:,0], points[:,1], points[:,2]], meth="l")
    
    sphere1.make_sphere()

    return ax_sphere0, ax_sphere1


ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init,
                            blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/rot_frame.{file_type}', fps = 20 )