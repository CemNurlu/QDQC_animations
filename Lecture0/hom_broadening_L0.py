from anim_base import  (cache_then_save_funcanimation, bloch_vector, PrettyAxis,file_type,
                        prepare_bloch_mosaic, math_fontfamily,
                        fit_damped_cosine)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from tqdm import tqdm
import random
from matplotlib.patches import FancyArrowPatch


##########################
# LONGER DURATIONS FOR ACTUAL ANIMATION
##########################



N_time = 850
t_0 = 10 # Show bloch sphere
t_1 = 510 # Show time evolution of all spins
t_2 = 540 # remove B_plot and move S_x plot up
t_3 = 560 # Show S_x_avg equation
t_4 = 610 # Do nothing
t_5 = 640 # Move S_x_avg equation up and show S_x_avg axis
t_6 = 740 # Show S_x_avg plot time evolution together with all spins
t_7 = 770 # Remove S_x plot and S_x_avg equation and move S_x_avg plot up
t_8 = 790 # Show fourier transform and T2 arrows
t_9 = N_time

##########################
# SHORTER DURATIONS FOR DEBUGGING
##########################

DEBUG = False

if DEBUG:
   N_time //= 5
   t_0 //= 5
   t_1 //= 5
   t_2 //= 5
   t_3 //= 5
   t_4 //= 5
   t_5 //= 5
   t_6 //= 5
   t_7 //= 5
   t_8 //= 5
    


bloch_mosaic = [["bloch", "plot"]]

vector_colors = ["red", "blue", "green", "purple", "orange", "cyan"]


bloch_kwargs = {
    "vector_color": vector_colors,
    "vector_width": 6,
    "vector_color": ["red", "blue", "green", "purple", "orange", "pink", "yellow"]
}


gridspec_kw = {"height_ratios":[1], "width_ratios":[1,1.8]}
fig, ax_dict, sphere_dict = prepare_bloch_mosaic(bloch_mosaic, (14,8), bloch_kwargs, gridspec_kw=gridspec_kw)

ax_dict["plot"].set_axis_off()

n_spins = 5

assert (t_1 - t_0) % (n_spins*5) == 0, "t_1 - t_0 must be divisible by n_spins"

n_omegas = 5

SEED = 12378
np.random.seed(SEED)
w_array = np.random.rand(n_spins, n_omegas)*2 + 0.5

B_time_start = 0
B_time_end = 25

B_time = np.linspace(B_time_start, B_time_end, t_1-t_0)
delta_t_B = B_time[1] - B_time[0]
B_0 = 1.5
B_fluctuation_multiplier = 0.6



B = np.ones((n_spins, t_1 - t_0)) * B_0

B_offsets = np.linspace(0, 2*np.pi/B_time_end, n_spins, endpoint=False)
np.random.shuffle(B_offsets)
B += B_offsets.reshape(-1,1)


for s_i in range(n_spins):
    for w_i in range(n_omegas):
        B[s_i,:] += np.cos(w_array[s_i,w_i]*B_time)/n_omegas*B_fluctuation_multiplier

B_min = np.min(B) - 0.1
B_max = np.max(B) + 0.1

phi = np.cumsum(B, axis=1)*delta_t_B


theta = np.pi/3
S_x_max = np.sin(theta)

spin_vectors = np.zeros((n_spins, t_1-t_0, 3))
for s_i in range(n_spins):
    spin_vectors[s_i,:,:] = bloch_vector(theta, phi[s_i,:])


actual_spin_avg = np.mean(spin_vectors, axis=0)
_ , (A_fit, omega_fit, phi_fit, tau_fit)= fit_damped_cosine(actual_spin_avg[:,0], B_time)
fitted_spin_avg_x = np.sin(theta)*np.cos(omega_fit*B_time + phi_fit)*np.exp(-B_time/tau_fit)

fitted_spin_avg_y = np.sin(theta)*np.sin(omega_fit*B_time + phi_fit)*np.exp(-B_time/tau_fit)
fitted_spin_avg_z = np.cos(theta)*np.ones_like(B_time)
fitted_spin_avg = np.stack((fitted_spin_avg_x, fitted_spin_avg_y, fitted_spin_avg_z), axis=1)


pretty_axises_B = []
for s_i in range(n_spins):
    pretty_axis_B = PrettyAxis(ax_dict["plot"], (0, 2, -1.4), (-1.4, -0.2, 0),
                            data_x_lim=(B_time_start,B_time_end+0.01),
                            data_y_lim=(B_min, B_max),
                            alpha = 0)
    pretty_axis_B.add_line(f"B_{s_i}", B_time[0], B[s_i,0], c = "red", alpha=0, lw = 2.5) 
    pretty_axis_B.add_label(fr'$B_{s_i+1}(t) \;$', "y", size = 18)
    pretty_axis_B.add_label(r'$\; t$', "x", size = 18)
    pretty_axises_B.append(pretty_axis_B)


pretty_axis_spins = PrettyAxis(ax_dict["plot"], (0,2, -2.8), (-3.8, -1.8, 0),
                        data_x_lim=(B_time_start,B_time_end+0.01),
                        data_y_lim=(-S_x_max - 0.1, S_x_max + 0.1),
                        alpha = 0)
pretty_axis_spins.add_label(r'$\langle S_x^{(i)}(t) \rangle $  ', "y", size = 20)
pretty_axis_spins.add_label(r' $t$', "x", size = 20)


pretty_axis_spin_avg = PrettyAxis(ax_dict["plot"], (0,2, -3), (-4, -2, 0),
                        data_x_lim=(B_time_start,B_time_end+0.01),
                        data_y_lim=(-S_x_max - 0.1, S_x_max + 0.1),
                        alpha = 0)
pretty_axis_spin_avg.add_label(r'$\langle S_x^{avg}(t) \rangle$  ', "y", size = 20)
pretty_axis_spin_avg.add_label(r' $t$', "x", size = 15)
pretty_axis_spin_avg.add_line("S_x_avg", B_time[0], fitted_spin_avg[0,0], c = "black", alpha=0, lw= 3.5)

y_lim_max = 1.08/n_spins + 0.2

ax_dict["plot"].set_xlim(-0.2, 2.2)
ax_dict["plot"].set_ylim(-4.2, y_lim_max)

avg_eq_string = r"$\langle S^{avg}_x (t) \rangle = \frac{1}{N} \sum_{i=1}^N \langle S_x^{(i)} (t) \rangle $"
avg_eq_x_y_start = np.array([0.1, -3.])
avg_eq_x_y_end = np.array([1.4, -1.9])

avg_eq_start_size, avg_eq_end_size = 50, 20
avg_eq = ax_dict["plot"].text(*avg_eq_x_y_start, avg_eq_string, size = avg_eq_start_size, alpha = 0, math_fontfamily = math_fontfamily)


pretty_axis_fourier = PrettyAxis(ax_dict["plot"], (0,2, -4), (-4, -2, 0),
                           data_x_lim=(0, 1), data_y_lim=(0, 1), alpha=0)
omega_fourier_vec = np.linspace(0, 1, 500)
mu = 0.5
sigma = 0.06
F_S_x = np.exp(-0.5*((omega_fourier_vec - mu)/sigma)**2)
pretty_axis_fourier.add_line("F_S_x", 
                    omega_fourier_vec, F_S_x,
                    "black", alpha = 0)
pretty_axis_fourier.add_label(r'$\mathcal{F} \; [ \langle S_x^{avg} \rangle ] $  ', "y", size = 20)
pretty_axis_fourier.add_label(r'  $\omega$', "x", size = 20)
omega_L_text = ax_dict["plot"].text(0.95, -4.1, r'$\omega_L$', size = 18, alpha = 0, math_fontfamily=math_fontfamily)

T2_arrow = FancyArrowPatch((0.03, 0.12), (1, 0.12), 
        arrowstyle='<->', mutation_scale=20,
        lw = 1.5, color = "black",
        alpha = 0
    )

T2_text = ax_dict["plot"].text(0.34, 0.23, r'$ \sim T_2$', size = 30, alpha = 0, math_fontfamily = math_fontfamily)

T2_inv_arrow = FancyArrowPatch((0.86, -3.13), (1.14, -3.13), 
        arrowstyle='<->', mutation_scale=20,
        lw = 1.5, color = "black",
        alpha = 0
    )

T2_inv_text = ax_dict["plot"].text(0.83, -3.43, r'$ \sim 1 / \, T_2$', size = 30, alpha = 0, math_fontfamily = math_fontfamily)

for arrow in [T2_arrow, T2_inv_arrow]:
    ax_dict["plot"].add_patch(arrow)
    arrow.set_zorder(10)


def animate(i):

    if i <= t_0:
        pass

    elif i <= t_1:
        frames_per_spin = (t_1 - t_0)//n_spins

        s_i = (i - t_0 - 1)//frames_per_spin

        local_t_i = (i - t_0 - 1)%frames_per_spin

        tail = frames_per_spin//10
        

        if local_t_i <= 0.1*frames_per_spin:
            new_alpha = local_t_i /(0.1* frames_per_spin)
            pretty_axises_B[s_i].alpha = new_alpha
            if s_i == 0:
                pretty_axis_spins.alpha = new_alpha

        elif local_t_i <= 0.9*frames_per_spin:

            B_time_index = int((len(B_time) - 1) * (local_t_i-0.1*frames_per_spin) /(0.8*frames_per_spin))
            
            sphere_dict["bloch"].vectors = []
            sphere_dict["bloch"].points = []
            sphere_dict["bloch"].vector_color = ["red", vector_colors[s_i+1]]
            sphere_dict["bloch"].point_color = [vector_colors[s_i+1]]
            sphere_dict["bloch"].add_vectors(
                [[0,0, B[s_i, B_time_index]/B_max],
                 spin_vectors[s_i,B_time_index]]
                 )
            if tail > B_time_index:
                sphere_dict["bloch"].add_points(spin_vectors[s_i,0:B_time_index+1].T, meth = "l")
            else:
                
                sphere_dict["bloch"].add_points(spin_vectors[s_i,B_time_index-tail+1:B_time_index+1].T, meth = "l")
            sphere_dict["bloch"].make_sphere()

            pretty_axises_B[s_i].update_line(f"B_{s_i}", B_time[0:B_time_index+1], B[s_i,0:B_time_index+1])

            if local_t_i - 0.1*frames_per_spin <= 1:
                pretty_axis_spins.add_line(f"spin_{s_i}_x", B_time[:B_time_index+1], spin_vectors[s_i,0:B_time_index+1,0], c = vector_colors[s_i+1], alpha=1, lw = 2.5)
            else:
                pretty_axis_spins.update_line(f"spin_{s_i}_x", B_time[:B_time_index+1], spin_vectors[s_i,:B_time_index+1,0])
        else:

            if local_t_i == frames_per_spin - 1:
                new_alpha = 1
            else:
                new_alpha = (local_t_i - 0.9*frames_per_spin) / ( 0.1 * frames_per_spin)
            
            x_0_target = s_i * 2/n_spins
            
            x1_target = (s_i+0.9) * 2/n_spins 
            x_height_target = 0.1 + 1.08/n_spins/6

            y_0_target = 0.1
            y_1_target = 0.1 + 1.08/n_spins

            new_x_pos = (0 + ( x_0_target - 0)  * new_alpha,
                        2 + (x1_target - 2) * new_alpha,
                        -1.4 + (y_0_target + 1.4)* new_alpha)

            new_y_pos = ( -1.4 + (y_0_target + 1.4)* new_alpha,
                        -0.2 + (y_1_target + 0.2) * new_alpha,
                        new_x_pos[0])

            pretty_axises_B[s_i].update_x_y_pos(new_x_pos, new_y_pos)   

        if i == t_1 - 1:
            sphere_dict["bloch"].vectors = []
            sphere_dict["bloch"].points = []
            sphere_dict["bloch"].vector_color = ["black"]
            sphere_dict["bloch"].point_color = ["black"]
            sphere_dict["bloch"].make_sphere()                 

    elif i <= t_2:
        new_alpha = ( t_2 - i) / (t_2 - t_1)
        
        for s_i in range(n_spins):
            pretty_axises_B[s_i].alpha = new_alpha
        
        new_y_pos = ( y_lim_max - 2.1  + (-3.8 - (y_lim_max - 2.1) ) * new_alpha, 
                    y_lim_max - 0.1  + (-1.8 - (y_lim_max - 0.1) ) * new_alpha, 
                    0)

        new_x_pos = ( 0,
                    2, 
                    (new_y_pos[0] + new_y_pos[1])/2)
                    
        pretty_axis_spins.update_x_y_pos(new_x_pos, new_y_pos)

    elif i <= t_3:
        new_alpha = (i - t_2) / (t_3 - t_2)
        avg_eq.set_alpha(new_alpha)
    
    elif i <= t_4:
        pass

    elif i <= t_5:
        new_alpha = (i - t_4) / (t_5 - t_4)
        pretty_axis_spin_avg.alpha = new_alpha
        new_avg_eq_x, new_avg_eq_y = avg_eq_x_y_start + (avg_eq_x_y_end - avg_eq_x_y_start) * new_alpha
        new_size = avg_eq_start_size + (avg_eq_end_size - avg_eq_start_size) * new_alpha
        avg_eq.set(x = new_avg_eq_x, y = new_avg_eq_y, fontsize = new_size)
    

    elif i <= t_6:
        new_alpha = (i - t_5) / (t_6 - t_5)
        B_time_index = int((len(B_time) - 1) * new_alpha)
        for s_i in range(n_spins):
            pretty_axis_spins.update_line(f"spin_{s_i}_x", B_time[:B_time_index+1], spin_vectors[s_i,:B_time_index+1,0])
        
        pretty_axis_spin_avg.update_line("S_x_avg", B_time[:B_time_index+1], fitted_spin_avg[:B_time_index+1,0])

        sphere_dict["bloch"].vectors = []
        sphere_dict["bloch"].add_vectors(fitted_spin_avg[B_time_index])
        sphere_dict["bloch"].make_sphere()

    elif i <= t_7:
        new_alpha = (t_7 - i) / (t_7 - t_6)

        new_y_pos = ( y_lim_max - 2.1  + (-3.8 - (y_lim_max - 2.1) ) * new_alpha, 
                    y_lim_max - 0.1  + (-1.8 - (y_lim_max - 0.1) ) * new_alpha, 
                    0)

        new_x_pos = ( 0,
                    2, 
                    (new_y_pos[0] + new_y_pos[1])/2)
        
        pretty_axis_spin_avg.update_x_y_pos(new_x_pos, new_y_pos)
        pretty_axis_spins.alpha = new_alpha
        avg_eq.set_alpha(new_alpha)

        
    elif i <= t_8:
        new_alpha = (i - t_7) / (t_8 - t_7)
        pretty_axis_fourier.alpha = new_alpha
        omega_L_text.set_alpha(new_alpha)
        T2_arrow.set_alpha(new_alpha)
        T2_text.set_alpha(new_alpha)
        T2_inv_arrow.set_alpha(new_alpha)
        T2_inv_text.set_alpha(new_alpha)

    return [ax for key, ax in ax_dict.items()]

def init():
    return [ax for key, ax in ax_dict.items()]



ani = anim.FuncAnimation(fig, animate, tqdm(range(N_time)), interval=50,
                            init_func=init, 
                            blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/hom_broadening_L0.{file_type}', fps = 20 )