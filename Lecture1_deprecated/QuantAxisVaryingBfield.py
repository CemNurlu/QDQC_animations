from matplotlib import pyplot as plt
from matplotlib import animation as anim
import qutip
import numpy as np
from anim_base import  cache_then_save_funcanimation, bloch_vector, rot_matrix, init_bloch_sphere, PrettyAxis, file_type
from tqdm import tqdm
import matplotlib.patches as mpatches


# DEFINE ALL TIME VARIABLES HERE
N_time = 100
t_0 = 10 # Show the bloch sphere
t_1 = 20 # Show the time axises
t_2 = 70 # Show the time evolution 
t_3 = 90 # Show levels image
t_4 = N_time #Do nothing between t_2 and N_time

delta_t_0 = t_0
delta_t_1 = t_1 - t_0
delta_t_2 = t_2 - t_1
delta_t_3 = t_3 - t_2



# DEFINE MAGNETIC FIELD HERE
# Time axis for B-field and phase of spin
B_time = np.linspace(8, 40, t_2-t_1)
# Bigger multiplier -> bigger fluctuations in B
B_fluctuation_multiplier = 1
# Constant offset of B
B_0 = 0.3
B = np.ones_like(B_time) * B_0

# INTEGRATE B TO GET PHASE
if B_fluctuation_multiplier != 0:
   # Add fluctuations to B
   w_subs = [0.81, 2.56, 1.72, 3.31, 1.93, 2.24]
   for w_sub in w_subs:
      B += np.cos(w_sub*B_time)/6 * B_fluctuation_multiplier
   # Integrate B to get phase ( integral is cumulative sum times steplength)
phi = np.cumsum(B) * (B_time[1] - B_time[0])



# COMPUTE SPIN VECTOR
# No decay
theta = np.pi/4
spin_vector = bloch_vector(theta, phi)
B_vector = bloch_vector(0, 0, B)
tail = 15
B_min, B_max = min(B), max(B)
S_x, S_y, S_z = spin_vector.T.copy()

# Shift the value to align with the B-field
# S_x_shift = - max(S_x)  - 0.35
# S_x += S_x_shift
S_x_max, S_x_min = max(S_x), min(S_x)


# INITIALISE OBJECTS FOR THE ANIMATION
size = 10
fig, axes = plt.subplots(1,2, figsize=(size*2, size), gridspec_kw={'width_ratios': [1, 1]})
axes[0].remove()
ax_sphere = fig.add_subplot(1,2,1, projection="3d", azim=-60, elev=30)
ax_plot = axes[1]
sphere = init_bloch_sphere(ax_sphere, 
               # sphere_alpha = 0,
               # frame_alpha = 0,
               # vector_alpha = 0,
               point_marker=["o"], 
               point_color=["blue"], 
               vector_color=["red", "blue"], 
               vector_width=6)
ax_sphere.axis("off")
sphere.add_vectors(B_vector[0])
sphere.add_vectors(spin_vector[0])

margin_between_plots = 0.2

# sphere.add_vectors(B_vector[0])
# sphere.make_sphere()

ax_plot.set_axis_off()
pretty_axis_B = PrettyAxis(ax_plot, 
                     x_pos = (B_time[0], B_time[-1], 0),
                     y_pos = (B_min, B_max, B_time[0]),
                     data_y_lim=(B_min, B_max),
                     alpha = 0)

pretty_axis_B.add_line("B", B_time[0], B[0], "red", alpha=0)
pretty_axis_B.add_line("B-blob", B_time[0], B[0], "red", linestyle = "o", alpha=0)
pretty_axis_B.add_label(r"$B(t)$  ", "y", size = 20)
pretty_axis_B.add_label(r"  $t$", "x", size = 20)

pretty_axis_S_x = PrettyAxis(ax_plot,
                     x_pos = (B_time[0], B_time[-1], B_min - (S_x_max-S_x_min)/2 - margin_between_plots),
                     y_pos = (B_min - (S_x_max-S_x_min)-margin_between_plots, B_min-margin_between_plots, B_time[0]),
                     data_y_lim=(S_x_min, S_x_max),
                     alpha = 0)

pretty_axis_S_x.add_line("S_x", B_time[0], S_x[0], "blue", alpha=0)
pretty_axis_S_x.add_line("S_x-blob", B_time[0], S_x[0], "blue", linestyle = "o", alpha=0)
pretty_axis_S_x.add_label(r"$S_x(t)  $", "y", size = 20)
pretty_axis_S_x.add_label(r"  $t$", "x", size = 20)



                     # B_min, B_max, 0.5, 0.5, 0.5, 0.5)
# ax_plot.plot([t[0], t[0]], [-0.1, B_max + 0.1], c = "black", lw = 1, ls = "--")
# ax_plot.plot([t[0]-0.3, t[-1]+0.3], [0,0], c = "black", lw = 1, ls = "--")

# B_line = ax_plot.plot(B_time[0], B[0], c = "red", alpha = 0)
# B_label = ax_plot.text(B_time[0]-0.5, B_max, s=r"$B(t)$", ha = "right", size = 15, alpha = 0)
# B_scatter = ax_plot.plot(B_time[0], B[0], 'o', c = "red", label = "B(t)", markersize = 15, alpha = 0)

# ax_plot.plot([t[0], t[0]], [x_min-0.1, x_max + 0.1], c = "black", lw = 1, ls = "--")
# ax_plot.plot([t[0]-0.3, t[-1]+0.3], [x_shift, x_shift], c = "black", lw = 1, ls = "--")
# x_line = ax_plot.plot(B_time[0], S_x[0], c = "blue", alpha = 0)
# x_label = ax_plot.text(B_time[0]-0.5, S_x_max, r"$S_x(t)$", ha = "right", size = 15, alpha = 0)
# x_scatter = ax_plot.plot(B_time[0], S_x[0], 'o', c = "blue", label = "x(t)", markersize = 15, alpha = 0)


#, label='line & marker - no line because only 1 point')
ax_plot.set_ylim(B_min - (S_x_max-S_x_min)-0.3, B_max + 0.2)

def animate(i):
   # FADE IN SPHERE
   if i <= t_0:
      sphere.vectors = []
      sphere.points = []
      # new_vec = bloch_vector(theta[i], phi[i])
      sphere.add_vectors(B_vector[0], alpha = i / t_0)
      sphere.add_vectors(spin_vector[0], alpha = i / t_0)
      # sphere.sphere_alpha = i / (t_0*5)
      # sphere.frame_alpha = i / (t_0*5)
      sphere.make_sphere()

   # FADE IN B and S_x
   elif i <= t_1:
      new_alpha = (i - t_0) / (t_1 - t_0)
      pretty_axis_B.alpha = new_alpha
      pretty_axis_S_x.alpha = new_alpha
      # for obj in [B_line, B_label, B_scatter, x_line, x_label, x_scatter]:
      #    obj.set_alpha(new_alpha)
      
   # TIME EVOLUTION
   elif i <= t_2:
      B_index = i - t_1 - 1
      # print(B_index)
      sphere.vectors = []
      sphere.points = []
      # new_vec = bloch_vector(theta[i], phi[i])
      sphere.add_vectors(B_vector[B_index])
      sphere.add_vectors(spin_vector[B_index])
      if B_index < tail or tail == -1:
         points = spin_vector[:B_index+1]
      else:
         points = spin_vector[B_index+1-tail:B_index+1]
      sphere.add_points([points[:,0], points[:,1], points[:,2]], meth="l")
      sphere.make_sphere()

      pretty_axis_B.update_line("B", B_time[:B_index+1], B[:B_index+1])
      pretty_axis_B.update_line("B-blob", B_time[B_index], B[B_index])

      pretty_axis_S_x.update_line("S_x", B_time[:B_index+1], S_x[:B_index+1])
      pretty_axis_S_x.update_line("S_x-blob", B_time[B_index], S_x[B_index])


      # B_line.set_xdata(B_time[B_index])
      # B_scatter.lines[3].set_ydata(B[i])

      # x_line.lines[7].set_xdata(B_time[i])
      # x_scatter.lines[7].set_ydata(S_x[i])

   else:
      pass

   return ax_sphere, ax_plot

def init():


   return ax_sphere, ax_plot

FFwriter = anim.FFMpegWriter()
ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init, blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/varyB2.{file_type}', fps = 20 )

