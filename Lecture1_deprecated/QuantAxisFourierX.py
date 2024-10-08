from matplotlib import pyplot as plt
from matplotlib import animation as anim
from pyparsing import alphanums
import qutip
import numpy as np
from anim_base.util import bloch_vector, rot_matrix
from tqdm import tqdm
import matplotlib.patches as mpatches

   

size = 10

fig, axes = plt.subplots(1,2, figsize=(size*2, size), gridspec_kw={'width_ratios': [1, 1]})
axes[0].remove()
ax_sphere = fig.add_subplot(1,2,1, projection="3d", azim=-60, elev=30)
ax_plot = axes[1]

ax_sphere.axis("off")

if True:
   sphere = qutip.Bloch(axes=ax_sphere)
   
   sphere.frame_alpha = 0
   sphere.sphere_alpha = 0
   
   sphere.vector_color = ["red", "blue"]
   sphere.vector_alpha = [0. , 0.]
   sphere.vector_width = 6
   
   sphere.point_alpha = [0.]
   sphere.point_marker = ["o"]
   sphere.point_color = ["blue"]

quant_vector = np.array([0,0,1])

N_time = 60
 

t_0 = 10 # Show the bloch sphere
t_1 = 20 # Show the time axis
t_2 = 30 # Show the time evolution
t_3 = 40 # Show the fourier axis
t_4 = 50 # Show the fourier transform
N_time = 60 # Do nothing between t_4 and N_time

t = np.arange(N_time)

# If no decay, put theta_end = theta_0
theta_0 = np.pi/4
theta_end = np.pi/100

if theta_0 == theta_end:
   tau = 0
else:
   tau = np.log(theta_end/theta_0)/(t[t_2] - t[t_1])

theta = theta_0*np.exp(tau*( t[t_1:t_2] - t[t_1]))

phi = np.linspace(0, 6*np.pi, t[t_2]-t[t_1])

bloch_points = bloch_vector(theta, phi)

fixed_r = False
if not fixed_r:
   z0 = bloch_points[0,2]
   bloch_points[:,2] = z0

tail = 30

x,y,z = bloch_points.T.copy()

x_max, x_min = max(x), min(x)

w = t

def animate(i):
   if i < t_0:
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

      ax_plot.lines[2].set_data(t[:i+1], x[:i+1])
      ax_plot.lines[3].set_data(t[i], x[i])
      # ax_plot.lines[3].set_ydata(x[i])

   
   elif i >= t0 and i < t1:
      if i == t0:
         ax_plot.text(t[0] - t0/20, -1 , r"$\mathcal{F} [ S_x(t) ] (\omega)$",  ha = "right", size = 15, alpha = 0)
         ax_plot.text(t0/2, -2.1, r"$\omega_L$", ha = "center", size = 15, alpha = 0)
         if theta_0 == theta_end:
            arrow = mpatches.Arrow(t0/2, -2, 0, 1, width=t0/20, color = "blue", alpha = 0)
            ax_plot.add_patch(arrow)
         else:
            mu_gauss = t0/2
            sigma_gauss = t0/30
            t_gauss = np.linspace(0, t0, 200)
            exponent = - ( (t_gauss-mu_gauss)/sigma_gauss)**2 / 2
            gauss = np.exp(exponent) - 2
            ax_plot.plot(t_gauss, gauss, c = "navy", lw = 4, alpha = 0)
         ax_plot.plot([t[0] - t0/20], t[t0-1] + t0/20], [-2,-2], c = "black", alpha = 0, ls = "--")
         ax_plot.plot([0, 0], [-2.1, -0.9], c = "black", alpha = 0, ls = "--")

      else:
         alpha = (i - t0)/(t1-t0)
         for child in ax_plot._children[4:]:
            child.set_alpha(alpha)
   elif i >= t1:
      pass

   return ax_sphere, ax_plot

def init():
   sphere.add_vectors(quant_vector)
   sphere.make_sphere()
   
   ax_plot.set_axis_off()
   ax_plot.plot([t[t_0], t[0]], [x_min - 0.1, x_max + 0.1], c = "black", lw = 1, ls = "--")
   ax_plot.plot([t[0] - t0/20, t[t0-1] + t0/20], [0,0], c = "black", lw = 1, ls = "--")
   ax_plot.plot(t[0], x[0], c = "royalblue", lw = 4)
   ax_plot.plot(t[0], x[0], 'o', c = "royalblue", label = r"X", markersize = 15) #, label='line & marker - no line because only 1 point')
   ax_plot.set_ylim(-2.2, x_max + 0.2)


   ax_plot.text(t[0] - t0/20, x_max, r"$S_x(t)$", ha = "right", size = 15)

   # ax_plot.legend(labelspacing=0.3, fontsize = 15)

   return ax_sphere, ax_plot

ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init, blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/fourier.{file_type}', fps = 40 )