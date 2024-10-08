from anim_base import  cache_then_save_funcanimation,  prepare_bloch_mosaic, math_fontfamily, file_type
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm

##########################
# LONGER DURATIONS FOR ACTUAL ANIMATION
##########################

N_time = 550
t_0 = 20 # Show lab bloch sphere
t_1 = 100 # Show lab time evolution
t_2 = 120 # Show H lab text 
t_3 = 160 # Let it sink in for a bit
t_4 = 180 # Show H lab equation
t_5 = 220 # Let it sink in for a bit
t_6 = 240 # Show W transformation equation
t_7 = 280 # Let it sink in for a bit
t_8 = 300 # Show rot frame text
t_9 = 340 # Let it sink in for a bit
t_10 = 360 # Show rot frame equation
t_11 = 420 # Let it sink in for a bit
t_12 = 440 # Show B_rot bloch sphere
t_13 = 550 # Show time evolution of B_rot
t_14 = N_time

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
    t_11 //= 10
    t_12 //= 10
    t_13 //= 10
    t_14 //= 10
    
bloch_mosaic = [["bloch_lab", "plot_between", "bloch_rot"],
                ["plot", "plot", "plot"]]

vector_colors = ["maroon", "hotpink","red"]

bloch_kwargs = [{
    "vector_color": vector_colors,
    "vector_alpha" : [1,1,0],
    "vector_width": 6,
    },
    {
    "vector_color": vector_colors[2:],
    "vector_width": 6,
    "xlabel": [r"$x^\prime$", ''],
    "ylabel": [r"$y^\prime$", '']
    }
]

gridspec_kw = {"height_ratios":[1,0.4], "width_ratios":[1, 0.5, 1]}
fig, ax_dict, sphere_dict = prepare_bloch_mosaic(bloch_mosaic, (10,6), bloch_kwargs, gridspec_kw=gridspec_kw)

ax_dict["plot"].set_axis_off()
ax_dict["plot_between"].set_axis_off()

B_x_max = 0.7
B_zeeman_lab_z = 1
# B_zeeman_rot_z = 0

phi_0 = 0
phi_end = 24.3*np.pi
phi = np.linspace(phi_0, phi_end, t_13-t_0)

B_zeeman_lab = np.zeros((len(phi), 3))
B_zeeman_lab[:,2] = B_zeeman_lab_z

B_drive_lab = np.zeros((len(phi), 3))
B_drive_lab[:,0] = B_x_max*np.cos(phi)
B_drive_lab[:,1] = B_x_max * np.sin(phi)

B_total_lab = np.zeros((len(phi), 3))
B_total_lab = B_zeeman_lab + B_drive_lab

B_drive_rot = np.zeros((len(phi), 3))
B_drive_rot[:,0] = B_x_max

sphere_dict["bloch_lab"].add_vectors([B_zeeman_lab[0], B_drive_lab[0]])

# B_x_dec_equation = ax_dict["plot"].text(-0.6, 1.4, r'$H_{\mathrm{driving}} \; = \; 2hS_x \mathrm{cos}(\omega t)$', color = vector_colors[0], alpha = 0, size = 18)
B_lab_H_text = ax_dict["plot"].text(-1.35, 0.2, r'$H \; =$', color = "red", alpha = 0, size = 20, math_fontfamily = math_fontfamily)
B_lab_H_zeeman_text = ax_dict["plot"].text(-1.13, 0.2, r'$H_{\mathrm{Zeeman}} + $', color = vector_colors[0], alpha = 0, size = 20, math_fontfamily = math_fontfamily)
B_lab_H_drive_text = ax_dict["plot"].text(-0.7, 0.2, r"$H_{\mathrm{driving}} \; = \;$", color = vector_colors[1], alpha = 0, size = 20, math_fontfamily = math_fontfamily)
B_lab_H_zeeman_eq = ax_dict["plot"].text(-1.55, 0.1, r'$\omega_L S_z + $', color = vector_colors[0], alpha = 0, size = 20, math_fontfamily = math_fontfamily)
B_lab_H_drive_eq = ax_dict["plot"].text(-1.23, 0.1, r"$h [ S_x \: \mathrm{cos}(\omega t) + S_y \: \mathrm{sin}(\omega t) ]$", color = vector_colors[1], alpha = 0, size = 20, math_fontfamily = math_fontfamily)

B_rot_H_text = ax_dict["plot"].text(0.9, 0.2, r"$H \prime \; =$", color = "red", alpha = 0, size = 20, math_fontfamily = math_fontfamily)
B_rot_H_zeeman_text = ax_dict["plot"].text(1.15, 0.2, r"$H \prime _{\mathrm{Zeeman}} + $", color = vector_colors[0], alpha = 0, size = 20, math_fontfamily = math_fontfamily)
B_rot_H_drive_text = ax_dict["plot"].text(1.6, 0.2, r"$H \prime _{\mathrm{driving}} \; = \;$", color = vector_colors[1], alpha = 0, size = 20, math_fontfamily = math_fontfamily)
B_rot_H_zeeman_eq = ax_dict["plot"].text(1.04, 0.1, r"$( \omega_L - \omega )  S \prime _z + $", color = vector_colors[0], alpha = 0, size = 20, math_fontfamily = math_fontfamily)
B_rot_H_drive_eq_1 = ax_dict["plot"].text(1.61, 0.1, r"$h S \prime _x = $", color = vector_colors[1], alpha = 0, size = 20, math_fontfamily = math_fontfamily)
B_rot_H_drive_eq_2 = ax_dict["plot"].text(1.38, 0.0, r"$h S \prime _x$", color = vector_colors[2], alpha = 0, size = 20, math_fontfamily = math_fontfamily)

W_trans_equation = ax_dict["plot_between"].text(0.1, 0.3, r'$W = \mathrm{exp}(-i \omega t S_z)$', color = "black", alpha = 0, size = 20, math_fontfamily = math_fontfamily)
W_trans_arrow = FancyArrowPatch((0.3, 0.5), (0.9, 0.5), 
        # arrowstyle='->',
        mutation_scale=120,
        lw = 2, 
        ec = "black",
        fc = "aquamarine",
        alpha = 0
    )

omega_is_omega_L = ax_dict["plot_between"].text(0.25, 0.75, r'$\omega = \omega_L$', color = "black", alpha = 0, size = 30, math_fontfamily = math_fontfamily)


ax_dict["plot_between"].add_patch(W_trans_arrow)
ax_dict["plot_between"].set_xlim(0, 1)
ax_dict["plot_between"].set_ylim(0, 1)


ax_dict["plot"].set_xlim(-1.6, 2.1)
ax_dict["plot"].set_ylim(-0.15, 0.25)

def animate(i):
    if i <= t_0:
        new_alpha = i/t_0
        sphere_dict["bloch_lab"].make_sphere()

    if t_0 < i <= t_13:

        if t_1 < i <= t_2:
            new_alpha = (i-t_1)/(t_2-t_1)
            sphere_dict["bloch_lab"].vector_alpha = [1,1,new_alpha]
        

        B_time_index = i - t_0 - 1
        sphere_dict["bloch_lab"].vectors = []
        sphere_dict["bloch_lab"].add_vectors( [B_zeeman_lab[B_time_index], B_drive_lab[B_time_index], B_total_lab[B_time_index] ])
        sphere_dict["bloch_lab"].make_sphere()


        if i > t_11:
            sphere_dict["bloch_rot"].vectors = []
            sphere_dict["bloch_rot"].add_vectors(B_drive_rot[B_time_index])
            sphere_dict["bloch_rot"].make_sphere()
    


    if t_1 < i <= t_2:
        new_alpha = (i-t_1)/(t_2-t_1)
        B_lab_H_text.set_alpha(new_alpha)
        B_lab_H_zeeman_text.set_alpha(new_alpha)
        B_lab_H_drive_text.set_alpha(new_alpha)

    if t_3 < i <= t_4:
        new_alpha = (i-t_3)/(t_4-t_3)
        B_lab_H_zeeman_eq.set_alpha(new_alpha)
        B_lab_H_drive_eq.set_alpha(new_alpha)
    
    if t_5 < i <= t_6:
        new_alpha = (i-t_5)/(t_6-t_5)
        W_trans_arrow.set_alpha(new_alpha)
        W_trans_equation.set_alpha(new_alpha)
        omega_is_omega_L.set_alpha(new_alpha)
    
    if t_7 < i <= t_8:
        new_alpha = (i-t_7)/(t_8-t_7)
        B_rot_H_text.set_alpha(new_alpha)
        B_rot_H_zeeman_text.set_alpha(new_alpha)
        B_rot_H_drive_text.set_alpha(new_alpha)
    
    if t_9 < i <= t_10:
        new_alpha = (i-t_9)/(t_10-t_9)
        B_rot_H_zeeman_eq.set_alpha(new_alpha)
        B_rot_H_drive_eq_1.set_alpha(new_alpha)
        B_rot_H_drive_eq_2.set_alpha(new_alpha)
    
    if t_11 < i <= t_12:
        pass
        
    return [ax for key, ax in ax_dict.items()] 

def init():
    return [ax for key, ax in ax_dict.items()]
    
ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init, 
                              blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/zero_detuning.{file_type}', fps = 20 )