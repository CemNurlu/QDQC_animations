from matplotlib import pyplot as plt
from matplotlib import animation as anim
import qutip
import numpy as np
from anim_base import  cache_then_save_funcanimation, bloch_vector, rot_matrix, file_type
from tqdm import tqdm

N_time = 300
size = 10

n_plots = 0
fig, axes = plt.subplots(2,2, figsize=(size, size))

axes[0,0].remove()
axes[1,0].remove()

ax_plot_super = axes[0,1]
ax_plot_avg = axes[1,1]

ax_plot_super.set_axis_off()
ax_plot_avg.set_axis_off()

ax_sphere_super = fig.add_subplot(2,2,1, projection="3d", azim=-60, elev=30)
ax_sphere_avg = fig.add_subplot(2,2,3, projection="3d", azim=-60, elev=30)
sphere_avg = qutip.Bloch(axes=ax_sphere_avg)
sphere_super = qutip.Bloch(axes=ax_sphere_super)
if True:
    sphere_super.vector_color = ["red", "blue", "green", "purple", "orange", "pink", "yellow"]
    sphere_super.vector_width = 6
    sphere_avg.vector_color = ["black"]
    sphere_avg.vector_width = 6


w_vec = np.array(
    [11.0859516652828848, 10.48751575588186447,
    10.8744352926749055,  10.6277185865151992,
    11.364233965544978,   11.708209673453967,
    12.415826542789627]
)


theta = np.pi/5
t = np.linspace(0, 5, N_time)

phi_super = w_vec.reshape(-1,1)*t.reshape(1,-1)
bloch_super = np.zeros((len(w_vec), N_time,  3))

for w_i in range(len(w_vec)):
    bloch_super[w_i,:,:] = bloch_vector(theta, phi_super[w_i])

bloch_avg = np.mean(bloch_super, axis=0)

def animate(i):
    sphere_avg.vectors = []
    sphere_super.vectors = []
    sphere_super.add_vectors(bloch_super[:,i,:])
    sphere_avg.add_vectors(bloch_avg[i])

    sphere_super.make_sphere()
    ax_sphere_super.set_title(r"Bloch vectors with different $\omega$", pad = 20, fontsize = 15)
    sphere_avg.make_sphere()
    ax_sphere_avg.set_title("Average of Bloch vectors above", pad = 20, fontsize = 15)
    for w_i in range(len(w_vec)):
        ax_plot_super.lines[w_i+2].set_data(t[:i+1], bloch_super[w_i,:i+1,0])

    ax_plot_avg.lines[2].set_data(t[:i+1], bloch_avg[:i+1,0])
    
    return (ax_plot_avg, ax_plot_super, ax_sphere_super, ax_sphere_avg)

def init():
    ax_plot_super.plot([-1, 6], [0,0], c = "black", lw = 1, ls = "--")
    ax_plot_super.plot([0,0], [np.cos(theta) + 0.05, -np.cos(theta)-0.05], c = "black", lw = 1, ls = "--")
    ax_sphere_super.set_title("Superposition of Bloch vectors")
    for w_i in range(len(w_vec)):
        ax_plot_super.plot(t[0], bloch_super[0,w_i,0])
        print(ax_plot_super.lines)

    ax_plot_avg.plot([-1,6], [0,0], c = "black", lw = 1, ls = "--")
    ax_plot_avg.plot([0,0], [np.cos(theta) + 0.05, -np.cos(theta)-0.05], c = "black", lw = 1, ls = "--")
    ax_plot_avg.plot(t[0], bloch_avg[0,0])

    ax_plot_super.set_title("Projections on |x>", pad = 20, fontsize = 15)
    ax_plot_avg.set_title("Projection on |x>", pad = 20, fontsize = 15)

    return ax_plot_super, ax_plot_avg, ax_sphere_super, ax_sphere_avg


ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init, blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/super_cos.{file_type}', fps = 40 )