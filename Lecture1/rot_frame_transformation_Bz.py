from anim_base import  cache_then_save_funcanimation, PrettySlider, prepare_bloch_mosaic, math_fontfamily, file_type
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm

##########################
# LONGER DURATIONS FOR ACTUAL ANIMATION
##########################

N_time = 430
t_0 = 30 # Show static bloch sphere and B_z
t_1 = 50 # Show B_z equation
t_2 = 90 # Let it sink in for a bit
t_3 = 110 # Show W transformation equation and omega slider
t_4 = 140 # Let it sink in for a bit
t_5 = 160 # Show rotating bloch sphere and B_z equation
t_6 = 190 # Let it sink in for a bit
t_7 = 300 # Slide omega from 0 to omega_L, shrink B_z and rotate sphere
t_8 = 300 # Stay in omega_L
t_9 = 410 # slide omega from omega_L to omega_max
t_10 = N_time

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
    t_9 //= 10
    t_10 //= 10
   


bloch_mosaic = [["bloch_lab", "plot_between", "bloch_rot"],
                ["plot", "plot", "plot"]]

vector_colors = ["maroon"]#, "blue", "green", vector_colors[0], "orange", "pink", "yellow"]

bloch_kwargs = [{
    "vector_color": vector_colors,
    "vector_width": 6,
    },
    {
    "vector_color": vector_colors,
    "vector_width": 6,
    "xlabel": [r"$x ^\prime $", ''],
    "ylabel": [r"$y ^\prime $", '']
    }
]


gridspec_kw = {"height_ratios":[1,0.1], "width_ratios":[1, 0.7, 1]}
fig, ax_dict, sphere_dict = prepare_bloch_mosaic(bloch_mosaic, (10,4), bloch_kwargs, gridspec_kw=gridspec_kw)

ax_dict["plot"].set_axis_off()
ax_dict["plot_between"].set_axis_off()


B_lab = 0.8
B_min_rot = -0.8

assert B_lab > 0 and B_min_rot < 0, "B_lab and B_min_rot must be positive and negative respectively"

omega_max = np.pi/10
omega_L = omega_max *  B_lab / (B_lab - B_min_rot)




B_rot = np.zeros(t_9-t_6)
B_rot[0:t_7-t_6] = np.linspace(B_lab, 0, t_7-t_6)
B_rot[t_8-t_6:] = np.linspace(0, B_min_rot, t_9-t_8)


# B_inds = int(len(B_rot)*0.4), int(len(B_rot)*0.6)


# B_rot[0:B_inds[0]] = np.linspace(B_lab, 0, B_inds[0])
# B_rot[B_inds[1]:] = np.linspace(0, B_min_rot, len(B_rot)-B_inds[1])

omega = (B_rot - B_lab) / (B_min_rot - B_lab) * omega_max
# print(omega)
# exit()

# B_rot = omega / omega_max * (B_min_rot - B_lab) + B_lab

vector_cutoff = 0.15
B_rot[np.abs(B_rot) < vector_cutoff] = 0


azim_angle_rot_sphere = (-60 - np.cumsum(omega)*(180/np.pi)) % 360
B_lab_equation = ax_dict["plot"].text(1.2, 0.7, r'$B_z = \frac{ \omega_L } { \gamma }$', color = vector_colors[0], alpha = 0, size = 23, math_fontfamily = math_fontfamily)

B_rot_equation = ax_dict["plot"].text(7.5, 0.7, r'$B \prime _z = \frac{ \omega_L - \omega } { \gamma }$', color = vector_colors[0], alpha = 0, size = 23, math_fontfamily = math_fontfamily)

omega_slider = PrettySlider(ax_dict["plot_between"], 
                x_pos=(0.1, 1), y_pos=(0.7, 0.7), data_lim = (-omega_max*0.1, omega_max*1.2),
                arrow_style="->", slider_dot_data=0, alpha = 0, c = ("black", "aquamarine"), 
                labels = (r"$0$ ", r"$\omega$", None), arrow_lw=3,
                label_size = 20, 
                label_c = ("black", "black", "black"))

W_trans_equation = ax_dict["plot_between"].text(0, 0.3, r'$W = \mathrm{exp}(-i \omega t S_z)$', color = "black", alpha = 0, size = 25, math_fontfamily = math_fontfamily)
W_trans_arrow = FancyArrowPatch((0.35, 0.5), (0.75, 0.5), 
        mutation_scale=110,
        lw = 2, 
        ec = "black",
        fc = "aquamarine",
        alpha = 0,
        mutation_aspect=1.
    )
ax_dict["plot_between"].add_patch(W_trans_arrow)
ax_dict["plot_between"].set_xlim(0, 1)
ax_dict["plot_between"].set_ylim(0, 1)

ax_dict["plot"].set_xlim(0, 10)
ax_dict["plot"].set_ylim(0,1)

def animate(i):
    # ax_dict["bloch_avg"].set_title("Average of Bloch vectors")
    # ax_dict["bloch_super"].set_title("Superposition of Bloch vectors")
    if i <= t_0:
        if i == 0:
            sphere_dict["bloch_lab"].add_vectors([0,0,B_lab])
            sphere_dict["bloch_lab"].make_sphere()
    
    elif t_0 < i <= t_1:
        new_alpha = (i-t_0)/(t_1-t_0)
        B_lab_equation.set_alpha(new_alpha)
    
    elif t_1 < i <= t_2:
        pass

    elif t_2 < i <= t_3:
        new_alpha = (i-t_2)/(t_3-t_2)
        omega_slider.alpha = new_alpha
        W_trans_arrow.set_alpha(new_alpha)
        W_trans_equation.set_alpha(new_alpha)
    
    elif t_3 < i <= t_4:
        pass

    elif t_4 < i <= t_5:
        if i == t_4+1:
            sphere_dict["bloch_rot"].add_vectors([0,0,B_lab])
            sphere_dict["bloch_rot"].make_sphere()
        new_alpha = (i-t_4)/(t_5-t_4)
        B_rot_equation.set_alpha(new_alpha)
    
    elif t_5 < i <= t_6:
        pass

    elif t_6 < i <= t_9:
        temp_index = i-t_6-1
        ax_dict["bloch_rot"].azim = azim_angle_rot_sphere[temp_index]
        sphere_dict["bloch_rot"].vectors = []
        if B_rot[temp_index] != 0:
            sphere_dict["bloch_rot"].add_vectors([0,0,B_rot[temp_index]])
        sphere_dict["bloch_rot"].make_sphere()
        omega_slider.update_slider_dot(omega[temp_index])
    

    return [ax for key, ax in ax_dict.items()] 

def init():
    return [ax for key, ax in ax_dict.items()]
    
ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init, 
                              blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/rot_frame_transformation_Bz.{file_type}', fps = 20 )