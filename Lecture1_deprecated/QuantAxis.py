from matplotlib import pyplot as plt
from matplotlib import animation as anim
import qutip
import numpy as np
from anim_base import  cache_then_save_funcanimation, bloch_vector, rot_matrix, file_type
from tqdm import tqdm
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec

plot_xyz = {
   "X": True, 
   "Y": True, 
   "Z": False
}

for coord in plot_xyz:
   if plot_xyz[coord]:
      make_plot = True
   

size = 10

if make_plot:
   n_plots = sum([int(plot_xyz[b]) for b in plot_xyz])

   fig, axes = plt.subplots(1,2, figsize=(size*2, size), gridspec_kw={'width_ratios': [1, 1]})
   axes[0].remove()
   ax_sphere = fig.add_subplot(1,2,1, projection="3d", azim=-60, elev=30)
   ax_plot = axes[1]

else:
   n_plots = 0
   fig, ax = plt.subplots(figsize=(size, size))
   ax.remove()
   ax_sphere = fig.add_subplot(1,1,1, projection="3d", azim=-60, elev=30)

ax_sphere.axis("off")

if True:
   sphere = qutip.Bloch(axes=ax_sphere)
   sphere.point_marker = ["o"]
   sphere.point_color = ["red"]
   sphere.vector_color = ["red", "blue"]
   sphere.point_marker = ["o"]
   sphere.point_color = ["blue"]
   sphere.vector_width = 6

quant_vector = np.array([0,0,1])

N_time = 300
t = np.arange(N_time)

# If no decay, put theta_end = theta_0
theta_0 = np.pi/3
theta_end = np.pi/100

if theta_0 == theta_end:
   tau = 0
else:
   tau = np.log(theta_end/theta_0)/t[-1]

theta = theta_0*np.exp(tau*t)

phi = np.linspace(0, 8*np.pi, N_time)

bloch_points = bloch_vector(theta, phi)

fixed_r = True
if not fixed_r:
   z0 = bloch_points[0,2]
   bloch_points[:,2] = z0

rot_params = {
   "rot": False,
   "axis" : bloch_vector(np.pi/2, np.pi, 1),
   "angle": np.pi/6
}

if rot_params["rot"]:
   R_matrix = rot_matrix(
      rot_params["axis"],
      rot_params["angle"]
      )

   bloch_points = np.einsum("kj, ij -> ki", bloch_points, R_matrix)
   quant_vector = np.einsum("ij, j -> i", R_matrix, quant_vector)

tail = 30

x,y,z = bloch_points.T.copy()





def animate(i):
   sphere.vectors = sphere.vectors[0:1]
   sphere.points = []
   # new_vec = bloch_vector(theta[i], phi[i])
   sphere.add_vectors(bloch_points[i])

   if i < tail or tail == -1:
      points = bloch_points[:i+1]
   else:
      points = bloch_points[i+1-tail:i+1]
   if i > 0:
      sphere.add_points([points[:,0], points[:,1], points[:,2]], meth="l")
   sphere.make_sphere()

   if make_plot:
      j = 0
      if plot_xyz["X"]:
         ax_plot.lines[j*3 + 2].set_xdata(t[i])
         ax_plot.lines[j*3 + 2].set_ydata(x[i] + x_offset)
         j += 1
      if plot_xyz["Y"]:
         ax_plot.lines[j*3 + 2].set_xdata(t[i])
         ax_plot.lines[j*3 + 2].set_ydata(y[i] + y_offset)
         j += 1
      if plot_xyz["Z"]:
         ax_plot.lines[j*3 + 2].set_xdata(t[i])
         ax_plot.lines[j*3 + 2].set_ydata(z[i] + z_offset)

   if make_plot:
      return ax_sphere, ax_plot
   else:
      return ax_sphere

x_max, x_min = max(x), min(x)
y_max, y_min = max(y), min(y)
z_max, z_min = max(z), min(z)

if n_plots == 2:
   if plot_xyz["X"] and plot_xyz["Y"]:
      x_offset = y_max-x_min + 0.1
      y_offset = 0
      z_offset = 0
   elif plot_xyz["X"] and plot_xyz["Z"]:
      x_offset = z_max-x_min + 0.1
      y_offset = 0
      z_offset = 0
   elif plot_xyz["Y"] and plot_xyz["Z"]:
      x_offset = 0
      y_offset = z_max-y_min + 0.1
      z_offset = 0

elif n_plots == 3:
   x_offset = y_max-x_min + 0.1
   y_offset = 0
   z_offset = - (z_max-y_min + 0.1)


def init():
   sphere.add_vectors(quant_vector)
   sphere.make_sphere()
   if make_plot:

      ax_plot.set_axis_off()

      if plot_xyz["X"]:
         ax_plot.plot([t[0] - len(t)/20, t[-1] + len(t)/20], [x_offset, x_offset], c = "black", lw = 1, ls = "--")
         # ax_plot.plot([0,0], [x_min*1.2, x_max*1.2], c = "black", lw = 1, ls = "--")
         ax_plot.plot(t, x + x_offset, c = "orange")
         ax_plot.plot(t[0], x[0], 'o', c = "orange", label = "X", markersize = 15) #, label='line & marker - no line because only 1 point')
         # ax_plot.scatter(t[0], x[0])
      if plot_xyz["Y"]:
         ax_plot.plot([t[0] - len(t)/20, t[-1] + len(t)/20], [y_offset, y_offset], c = "black", lw = 1, ls = "--")
         # ax_plot.plot([0,0], [y_min*1.2, y_max*1.2], c = "black", lw = 1, ls = "--")
         ax_plot.plot(t, y + y_offset, c = "green")
         ax_plot.plot(t[0], y[0], 'o', c = "green", label = "Y", markersize = 15)
      if plot_xyz["Z"]:
         ax_plot.plot([t[0] - len(t)/20, t[-1] + len(t)/20], [z_offset, z_offset], c = "black", lw = 1, ls = "--")
         # ax_plot.plot([0,0], [z_min*1.2, z_max*1.2], c = "black", lw = 1, ls = "--")
         ax_plot.plot(t,z + z_offset, c = "purple")
         ax_plot.plot(t[0], z[0], 'o', c = "purple", label = "Z", markersize = 15)
      ax_plot.legend(labelspacing=0.3, fontsize = 15)
      return ax_sphere, ax_plot
   else:
      return ax_sphere

ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init, blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/quant_ax2.{file_type}', fps = 20 )