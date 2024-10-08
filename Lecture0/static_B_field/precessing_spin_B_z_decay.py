import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from tqdm import tqdm
from matplotlib.patches import FancyArrowPatch

from anim_base import cache_then_save_funcanimation, bloch_vector, PrettyAxis, prepare_bloch_mosaic,math_fontfamily, file_type

N_time = 450
t_0 = 10 # Show bloch sphere
t_1 = 50 # Show time axises
t_2 = 220 # Show time evolution
t_3 = 250 # remove plots except S_x
t_4 = 300 # Move S_x plot
t_5 = 320 # Show T2 arrow and envelope
t_6 = 370 # Do nothing
t_7 = 390 # Show fourier transform and 1/T2 arrow
t_8 = N_time

##########################
# SHORTER DURATIONS FOR DEBUGGING
##########################

DEBUG = False

if DEBUG:
   N_time //= 10
   t_0 //= 10
   t_1 //= 10
   t_2 //= 10
   t_3 //= 10
   t_4 //= 10
   t_5 //= 10
   t_6 //= 10
   t_7 //= 10
   t_8 //= 10



plot_S_xyz = {"S_x":True, "S_y":True, "S_z":True}

bloch_kwargs = {
    "vector_color": ["red", "blue"],
    "point_color": [ "blue"],
    "point_marker": ["o"]
}

bloch_mosaic = [["bloch_0", "plot_0"]]
fig, ax_dict, sphere_dict = prepare_bloch_mosaic(bloch_mosaic, (12,6), bloch_kwargs)

B_vector = np.array([0,0,1])
B_time = np.linspace(0,1, t_2-t_1)

t = np.arange(N_time)

theta_0_spin = np.pi/3
theta_end_spin = np.pi/100
tau = np.log(theta_end_spin/theta_0_spin)

exp_envelope = np.exp(tau*B_time)
theta_spin = theta_0_spin*exp_envelope

phi_spin = np.linspace(np.pi/3, np.pi*(8+1/3), t_2-t_1)



bloch_points_spin = bloch_vector(theta_spin, phi_spin)

bloch_points_spin[:,2] = bloch_points_spin[0,2] 
S_x, S_y, S_z = bloch_points_spin.T.copy()
# S_z = np.ones_like(S_z) * S_z[0]

sphere_dict["bloch_0"].add_vectors(B_vector)
sphere_dict["bloch_0"].add_vectors(bloch_points_spin[0])

sphere_dict["bloch_0"].make_sphere()

n_component_plots = sum([int(val) for val in plot_S_xyz.values()])

if n_component_plots > 0:

    pretty_axis_B = PrettyAxis(ax_dict["plot_0"], (0, 1, -2.4*n_component_plots/4 ), (-2.4*n_component_plots/2 + 0.2, -0.2, 0), 
                            data_x_lim=(0, 1), data_y_lim=(-1, 1), alpha=0)
    pretty_axis_B.add_line("B", 0, 0, "red", alpha = 0)
    pretty_axis_B.add_label(r'$B(t)$  ', "y", size = 20)

plot_index = 0
pretty_axises = []
for coord, include_plot in plot_S_xyz.items():
    if include_plot:
        x_pos = (1.4, 2.4, -1.2 - 2.4*plot_index)
        y_pos = ( -2.2 - 2.4*plot_index, - 0.2 - 2.4*plot_index, 1.4)
        data_x_lim = (0, 1)
        data_y_lim = (-1, 1)

        pretty_axis = PrettyAxis(ax_dict["plot_0"], x_pos, y_pos, 
                                data_x_lim=data_x_lim, data_y_lim=data_y_lim, alpha=0)
        
        if coord == "S_x":
            component_value = S_x[0]
        elif coord == "S_y":
            component_value = S_y[0]
        elif coord == "S_z":
            component_value = S_z[0]
        
        pretty_axis.add_line(coord, 0, component_value, "blue", alpha = 0)
        pretty_axis.add_label(fr'$ \langle {coord}(t) \rangle$  ', "y", size = 20)
        pretty_axis.add_label(r'  $t$', "x", size = 15)

        pretty_axises.append(pretty_axis)
        plot_index += 1

if n_component_plots > 0:
   pretty_axises[0].add_line

if n_component_plots > 0:
   pretty_axis_fourier = PrettyAxis(ax_dict["plot_0"], 
                           (0, 2.4, -2.4*n_component_plots + 0.2), 
                           (-2.4*n_component_plots + 0.2, -2.4*n_component_plots/2 - 0.2, 0),
                           data_x_lim=(0, 1), data_y_lim=(0, 1), alpha=0)
   
   omega_vec = np.linspace(0, 1, 500)
   mu = 0.5
   sigma = 0.08
   F_S_x = np.exp(-0.5*((omega_vec - mu)/sigma)**2)

   
   pretty_axis_fourier.add_line("F_S_x", 
                       omega_vec, F_S_x,
                        "blue", alpha = 0)
   pretty_axis_fourier.add_label(r'$\mathcal{F} \; [ S_x ] $  ', "y", size = 20)
   pretty_axis_fourier.add_label(r'  $\omega$', "x", size = 22)
   omega_L_text = ax_dict["plot_0"].text(1.1, -2.4*n_component_plots -0.2 , r'$\omega_L$', size = 15, alpha = 0)

T2_arrow = FancyArrowPatch((0.0, -0.15), (1, -0.15), 
        arrowstyle='<->', mutation_scale=20,
        lw = 1.5, color = "black",
        alpha = 0
    )

T2_text = ax_dict["plot_0"].text(0.36, -0.05, r'$ \sim T_2$', size = 25, alpha = 0, math_fontfamily = math_fontfamily)

T2_inv_arrow = FancyArrowPatch((0.98, -5.45), (1.42, -5.45), 
        arrowstyle='<->', mutation_scale=20,
        lw = 1.5, color = "black",
        alpha = 0
    )

T2_inv_text = ax_dict["plot_0"].text(0.96, -5.95, r'$ \sim 1 / \, T_2$', size = 25, alpha = 0, math_fontfamily = math_fontfamily)
   
for arrow in [T2_arrow, T2_inv_arrow]:
    ax_dict["plot_0"].add_patch(arrow)
    arrow.set_zorder(10)

if n_component_plots > 0:
    ax_dict["plot_0"].set_xlim(-0.2, 2.5)
    ax_dict["plot_0"].set_ylim(0 - 2.4*n_component_plots, 0)
    ax_dict["plot_0"].set_axis_off()

tail = len(B_time)//6

def animate(i):
   if i <= t_0:
      pass

   elif i <= t_1:
      new_alpha = (i-t_0)/(t_1-t_0)
      if n_component_plots > 0:
         for pretty_axis in pretty_axises:
               pretty_axis.alpha = new_alpha
         pretty_axis_B.alpha = new_alpha

   elif i <= t_2:
      S_index = i - t_1 - 1
      plot_index = 0
      for coord, include_plot in plot_S_xyz.items():
         if include_plot:

               if coord == "S_x":
                  component_values = S_x[:S_index+1]
               elif coord == "S_y":
                  component_values = S_y[:S_index+1]
               elif coord == "S_z":
                  component_values = S_z[:S_index+1]

               pretty_axises[plot_index].update_line(coord, B_time[:S_index+1], component_values)

               plot_index += 1
      pretty_axis_B.update_line("B", B_time[:S_index+1], np.ones(S_index+1)*0.7)
      
      sphere_dict["bloch_0"].vectors = sphere_dict["bloch_0"].vectors[0:1]
      sphere_dict["bloch_0"].points = []
      # new_vec = bloch_vector(theta[i], phi[i])
      sphere_dict["bloch_0"].add_vectors(bloch_points_spin[S_index])

      if S_index < tail or tail == -1:
         points = bloch_points_spin[:S_index+1]
      else:
         points = bloch_points_spin[S_index+1-tail:S_index+1]
      if S_index > 0:
         sphere_dict["bloch_0"].add_points([points[:,0], points[:,1], points[:,2]], meth="l")
      sphere_dict["bloch_0"].make_sphere()  
   
   elif i <= t_3:
      new_alpha = (t_3-i)/(t_3-t_2)
      if n_component_plots > 0:
         for pretty_axis in pretty_axises:
            for key in pretty_axis.plot_lines:
                  if key != "S_x":
                     pretty_axis.alpha = new_alpha
         pretty_axis_B.alpha = new_alpha

   elif i <= t_4:

      new_x_pos = (1.4 * (t_4-i)/(t_4-t_3),
                   2.4, 
                   -1.2 - 0.6*(i-t_3)/(t_4-t_3))
      new_y_pos = (-2.2 - 1.2*(i-t_3)/(t_4-t_3), 
               - 0.2,
                1.4 * (t_4-i)/(t_4-t_3))
      
      S_x_pretty_axis = pretty_axises[0]
      S_x_pretty_axis.update_x_y_pos(new_x_pos, new_y_pos)

   elif i <= t_5:
      if i == t_4 + 1:
         pretty_axises[0].add_line("pos_env", B_time, exp_envelope, c = "silver", linestyle = "--", alpha = 0, lw = 1.5)
         pretty_axises[0].add_line("neg_env", B_time, -exp_envelope, c = "silver", linestyle = "--", alpha = 0, lw = 1.5)
      new_alpha = (i-t_4)/(t_5-t_4)
      T2_arrow.set_alpha(new_alpha)
      T2_text.set_alpha(new_alpha)
      pretty_axises[0]._plot_lines["pos_env"].set_alpha(new_alpha)
      pretty_axises[0]._plot_lines["neg_env"].set_alpha(new_alpha)
   
   elif i <= t_6:
      pass

   elif i <= t_7:
      new_alpha = (i-t_6)/(t_7-t_6)
      pretty_axis_fourier.alpha = new_alpha
      omega_L_text.set_alpha(new_alpha)
      T2_inv_arrow.set_alpha(new_alpha)
      T2_inv_text.set_alpha(new_alpha)
   
   elif i <= t_8:
      pass
      

   if n_component_plots > 0:
      return ax_dict["bloch_0"], ax_dict["plot_0"]
   else:
      return ax_dict["bloch_0"]

def init():
   if n_component_plots > 0:
      return ax_dict["bloch_0"], ax_dict["plot_0"]
   else:
      return ax_dict["bloch_0"]


ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init, blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/precessing_spin_B_Z_decay.{file_type}', fps = 20 )