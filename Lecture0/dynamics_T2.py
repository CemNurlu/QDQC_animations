import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from tqdm import tqdm
from matplotlib.patches import FancyArrowPatch

from anim_base import cache_then_save_funcanimation, bloch_vector, PrettyAxis, PrettySlider, math_fontfamily, file_type

N_time = 300
t_0 = 10 # Show time graph
t_1 = 40 # Do nothing
t_2 = 50 # Show Fourier transform
t_3 = 80 # Do nothing
t_4 = 100 # Show T2 slider initiated in 1/T2 = 0
t_5 = 140 # Do nothing
t_6 = 260 # Push T2 -> infinity
t_7 = N_time


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


t = np.arange(N_time)

fig, ax = plt.subplots(1,1, figsize=(6,6))

N_T2_values = t_6 - t_5 + 1

N_timepoints = 200

time_vector = np.linspace(0,1, N_timepoints)

omega = 4*np.pi

log_T_2_min, log_T_2_max = -1, 1.5

T2_values = np.logspace( log_T_2_max, log_T_2_min, N_T2_values)

cosine = np.cos(omega*time_vector)

decays = np.exp(-time_vector.reshape(1,-1)/T2_values.reshape(-1,1))

cosine_decays = cosine.reshape(1,-1)*decays

figT2, axT2 = plt.subplots(1,1, figsize=(6,6))
axT2.plot(np.arange(len(T2_values)), T2_values)


pretty_axis_t = PrettyAxis(ax, (0, 4, -2.2), (-3.2, -1.2, 0),
                            data_x_lim=(0, 1), data_y_lim=(-1, 1), alpha=0)
pretty_axis_t.add_line("spin", time_vector, cosine_decays[0], "blue", alpha = 0)
pretty_axis_t.add_line("pos_env", time_vector, decays[0], c = "silver", linestyle = "--", alpha = 0, lw = 1.5)
pretty_axis_t.add_line("neg_env", time_vector, -decays[0], c = "silver", linestyle = "--", alpha = 0, lw = 1.5)
pretty_axis_t.add_label(r'$\langle S_x(t) \rangle$', "y", size = 20)
pretty_axis_t.add_label(r'  $t$', "x", size = 20)

omega_vec = np.linspace(0, 1, 500)
omega_mu = 0.5

sigmas = 0.05/T2_values
fourier = np.exp(-0.5*((omega_vec.reshape(1,-1) - omega_mu)/sigmas.reshape(-1,1))**2)/sigmas.reshape(-1,1)

max_vals = np.max(fourier, axis = 1)
fourier = fourier/(max_vals.reshape(-1,1))**0.3




# exit()
pretty_axis_fourier = PrettyAxis(ax, (0, 4, -5.1), ( -5.1, -3.6, 0),
                           data_x_lim=(0, 1), data_y_lim=(0, np.max(fourier)), alpha=0)
# print(fourier[middle_T2_index])
pretty_axis_fourier.add_line("F_spin", omega_vec, fourier[0], "blue", alpha = 0, lw = 3.5)
                       
pretty_axis_fourier.add_label(r'$\mathcal{F} \; [ S_x ] $  ', "y", size = 20)
pretty_axis_fourier.add_label(r'  $\omega$', "x", size = 20)
omega_L_text = ax.text(1.9, -5.3, r'$\omega_L$', size = 25, va = "center", alpha = 0, math_fontfamily = math_fontfamily)

T2_slider = PrettySlider(ax, (0.3, 3.8), (-0.5, -0.5), (-0.05, 1.08),
                arrow_style='->', slider_dot_data= 0,
                alpha = 0, c = ("black", "teal"), 
                labels = (r"$0$  ",r" $\infty$", r"$1/T_2$"), 
                label_size = 25, 
                center_label_offset=0.25, label_c= ("black", "black", "teal"))
ax.set_axis_off()

def animate(i):
    if i <= t_0:
        new_alpha = i/t_0
        pretty_axis_t.alpha = new_alpha

    elif i <= t_1:
        pass

    elif i <= t_2:
        new_alpha = (i-t_1)/(t_2-t_1)
        pretty_axis_fourier.alpha = new_alpha
        omega_L_text.set_alpha(new_alpha)
    
    elif i <= t_3:
        pass

    elif i <= t_4:
        new_alpha = (i-t_3)/(t_4-t_3)
        T2_slider.alpha = new_alpha
    
    elif i <= t_5:
        pass

    elif i <= t_6:
        new_T2_index = i - t_5
        pretty_axis_t.update_line("spin", time_vector, cosine_decays[new_T2_index])
        pretty_axis_t.update_line("pos_env", time_vector, decays[new_T2_index])
        pretty_axis_t.update_line("neg_env", time_vector, -decays[new_T2_index])
        pretty_axis_fourier.update_line("F_spin", omega_vec, fourier[new_T2_index])

        slider_dot_data = new_T2_index/(N_T2_values-1)
        T2_slider.update_slider_dot(slider_dot_data)

    elif i <= t_7:
        pass

    
    return ax

def init():
   return ax


ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init, blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/dynamics_T2.{file_type}', fps = 20 )