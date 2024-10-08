from anim_base import  cache_then_save_funcanimation, bloch_vector, PrettyAxis,   prepare_bloch_mosaic, math_fontfamily, file_type
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm

##########################
# LONGER DURATIONS FOR ACTUAL ANIMATION
##########################

N_time = 600
t_0 = 20 # Show decomposed bloch sphere and pretty axis
t_1 = 100 # Show B_dec time evolution
t_2 = 120 # Show B_dec equation 
t_3 = 160 # Let it sink in for a bit
t_4 = 180 # Show W transformation equation
t_5 = 220 # Let it sink in for a bit
t_6 = 240 # Show B_rot equation version 1
t_7 = 300 # Let it sink in for a bit
t_8 = 320 # Show B_rot bloch sphere, pretty axis
t_9 = 400 # Show time evolution of B_rot
t_10 = 420 # Show B_rot equation version 2
t_11 = 460 # Let it sink in for a bit
t_12 = 490 # Fade out counter rotating arrows and equation
t_13 = 570 # Show time evolution of B_rot
t_14 = N_time

##########################
# SHORTER DURATIONS FOR DEBUGGING, COMMEN OUT WHEN DONE DEBUGGING
##########################

DEBUG = False

if DEBUG:
    N_time = N_time//10
    t_0 = t_0//10
    t_1 = t_1//10
    t_2 = t_2//10
    t_3 = t_3//10
    t_4 = t_4//10
    t_5 = t_5//10
    t_6 = t_6//10
    t_7 = t_7//10
    t_8 = t_8//10
    t_9 = t_9//10
    t_10 = t_10//10
    t_11 = t_11//10
    t_12 = t_12//10
    t_13 = t_13//10
    t_14 = t_14//10


bloch_mosaic = [["bloch_lab", "plot_between", "bloch_rot"],
                ["plot", "plot", "plot"]]

vector_colors = ["purple", "hotpink", "deepskyblue"]#, "blue", "green", vector_colors[0], "orange", "pink", "yellow"]

bloch_kwargs = [{
    "vector_color": vector_colors[1:],
    "vector_width": 6,
    },
    {
    "vector_color": vector_colors[1:],
    "vector_width": 6,
    "xlabel": [r"$x ^\prime $", ''],
    "ylabel": [r"$y ^\prime $", '']
    }
]


gridspec_kw = {"height_ratios":[1,1.2], "width_ratios":[1, 0.5, 1]}
fig, ax_dict, sphere_dict = prepare_bloch_mosaic(bloch_mosaic, (10,8), bloch_kwargs, gridspec_kw=gridspec_kw)

ax_dict["plot"].set_axis_off()
ax_dict["plot_between"].set_axis_off()


B_x_max = 1
phi_0 = 0
phi_end = 40*np.pi
phi = np.linspace(phi_0, phi_end, t_13-t_0)

# B_x = B_x_max*np.cos(phi)

B_dec = np.zeros((2,t_13-t_0, 3))


B_dec[:,:,0] = B_x_max*np.cos(phi)
B_dec[0,:,1] = B_x_max * np.sin(phi)
B_dec[1,:,1] = -B_x_max * np.sin(phi)

B_rot = np.zeros((2,t_13-t_0, 3))
B_rot[0,:,0] = B_x_max
B_rot[1,:,0] = B_x_max * np.cos(2*phi)
B_rot[1,:,1] = - B_x_max * np.sin(2*phi)


sphere_dict["bloch_lab"].add_vectors([B_dec[0,0,:], B_dec[1,0,:]])

pretty_axis_B_dec = PrettyAxis(ax_dict["plot"], (-1.1, 1.1, 0), (-1.1, 1.1, 0),
                        data_x_lim=(-1.1, 1.1),
                        data_y_lim=(-1.1, 1.1),
                        alpha = 0)

pretty_axis_B_dec.add_label(r"$B_x(t)$  ", "x", size = 20)
pretty_axis_B_dec.add_label(r"$B_y(t)$  ", "y", size = 20)


pretty_axis_B_rot = PrettyAxis(ax_dict["plot"], (3, 5.2, 0), (-1.1, 1.1, 4.1),
                        data_x_lim=(-1.1, 1.1),
                        data_y_lim=(-1.1, 1.1),
                        alpha = 0)

pretty_axis_B_rot.add_label(r"$B \prime _{x}(t)$  ", "x", size = 20)
pretty_axis_B_rot.add_label(r"$B \prime _{y}(t)$  ", "y", size = 20)

B_arrow_dec_0 = FancyArrowPatch((0,0), (B_dec[0,0,0], B_dec[0,0,1]), 
        arrowstyle='-|>', mutation_scale=20,
        lw = 5, color = vector_colors[1],
        alpha = 0
    )

B_arrow_dec_1 = FancyArrowPatch((0,0), (B_dec[1,0,1], B_dec[1,0,1]), 
        arrowstyle='-|>', mutation_scale=20,
        lw = 5, color = vector_colors[2],
        alpha = 0
    )

B_arrow_rot_0 = FancyArrowPatch((2.6,0), (B_dec[0,0,0] + 4.1, B_dec[0,0,1]),
        arrowstyle='-|>', mutation_scale=20,
        lw = 5, color = vector_colors[1],
        alpha = 0
    )

B_arrow_rot_1 = FancyArrowPatch((2.6,0), (B_dec[1,0,0] + 4.1, B_dec[1,0,1]),
        arrowstyle='-|>', mutation_scale=20,
        lw = 5, color = vector_colors[2],
        alpha = 0
    )

B_dec_H_equation = ax_dict["plot"].text(-0.4, 2.05, r'$H_{driving} \; = \;$', color = vector_colors[0], alpha = 0, size = 23, math_fontfamily = math_fontfamily)
B_dec_0_equation = ax_dict["plot"].text(-1.1, 1.75, r'$h [ S_x \: \mathrm{cos}(\omega t) + S_y \: \mathrm{sin}(\omega t) ]$', color = vector_colors[1], alpha = 0, size = 23, math_fontfamily = math_fontfamily)
B_dec_1_equation = ax_dict["plot"].text(-1.3, 1.45, r'$+ \; h [ S_x \: \mathrm{cos}(\omega t) - S_y \: \mathrm{sin}(\omega t) ]$', color = vector_colors[2], alpha = 0, size = 23, math_fontfamily = math_fontfamily)

B_rot_H_equation = ax_dict["plot"].text(3.6, 2.05, r"$H \prime_{driving}  \; = \;$", color = vector_colors[0], alpha = 0, size = 23, math_fontfamily = math_fontfamily)
B_rot_0_equation = ax_dict["plot"].text(3.85, 1.75, r"$ h S \prime_{x}$", color = vector_colors[1], alpha = 0, size = 23, math_fontfamily = math_fontfamily)
B_rot_1_equation_harmonic = ax_dict["plot"].text(2.7, 1.45, r"$+ \; h [ S \prime _{x} \mathrm{cos}(2 \omega t) - S \prime _{y} \mathrm{sin}(2 \omega t) ]$", color = vector_colors[2], alpha = 0, size = 23, math_fontfamily = math_fontfamily)
B_rot_1_equation_neglect = ax_dict["plot"].text(2.7, 1.45, r"+ $rapidly \; oscillating \; term$", color = vector_colors[2], alpha = 0, size = 23, math_fontfamily = math_fontfamily)


for arrow in [B_arrow_dec_0, B_arrow_dec_1, B_arrow_rot_0, B_arrow_rot_1]:#, W_trans_arrow]:
    ax_dict["plot"].add_patch(arrow)
    arrow.set_zorder(10)

W_trans_equation = ax_dict["plot_between"].text(0.1, 0.05, r'$W = \mathrm{exp}(-i \omega t S_z)$', color = "black", alpha = 0, size = 20, math_fontfamily = math_fontfamily)
W_trans_arrow = FancyArrowPatch((0.2, 0.25), (1, 0.25), 
        # arrowstyle='->',
        mutation_scale=100,
        lw = 2, 
        ec = "black",
        fc = "aquamarine",
        alpha = 0
    )
ax_dict["plot_between"].add_patch(W_trans_arrow)
ax_dict["plot_between"].set_xlim(0, 1)
ax_dict["plot_between"].set_ylim(0, 1)

ax_dict["plot"].set_xlim(-1.4, 5.3)
ax_dict["plot"].set_ylim(-1.2, 1.7)

def animate(i):
    # ax_dict["bloch_avg"].set_title("Average of Bloch vectors")
    # ax_dict["bloch_super"].set_title("Superposition of Bloch vectors")
    if i <= t_0:
        new_alpha = i/t_0
        pretty_axis_B_dec.alpha = new_alpha
        B_arrow_dec_0.set_alpha(new_alpha)
        B_arrow_dec_1.set_alpha(new_alpha)

    if t_0 < i <= t_13:
        B_time_index = i - t_0 - 1
        B_arrow_dec_0.set_positions((0,0), (B_dec[0,B_time_index,0], B_dec[0,B_time_index,1]))
        B_arrow_dec_1.set_positions((0,0), (B_dec[1,B_time_index,0], B_dec[1,B_time_index,1]))
        sphere_dict["bloch_lab"].vectors = []
        sphere_dict["bloch_lab"].add_vectors([B_dec[0,B_time_index], B_dec[1,B_time_index]])
        sphere_dict["bloch_lab"].make_sphere()

        if i > t_7:
            B_arrow_rot_0.set_positions((4.1,0), (B_rot[0,B_time_index,0] + 4.1, B_rot[0,B_time_index,1]))
            B_arrow_rot_1.set_positions((4.1,0), (B_rot[1,B_time_index,0] + 4.1, B_rot[1,B_time_index,1]))
            sphere_dict["bloch_rot"].vectors = []
            sphere_dict["bloch_rot"].add_vectors([B_rot[0,B_time_index],B_rot[1,B_time_index]])
            sphere_dict["bloch_rot"].make_sphere()


    if t_1 < i <= t_2:
        new_alpha = (i-t_1)/(t_2-t_1)
        B_dec_H_equation.set_alpha(new_alpha)
        B_dec_0_equation.set_alpha(new_alpha)
        B_dec_1_equation.set_alpha(new_alpha)
    
    if t_3 < i <= t_4:
        new_alpha = (i-t_3)/(t_4-t_3)
        W_trans_arrow.set_alpha(new_alpha)
        W_trans_equation.set_alpha(new_alpha)
    
    if t_5 < i <= t_6:
        new_alpha = (i-t_5)/(t_6-t_5)
        B_rot_H_equation.set_alpha(new_alpha)
        B_rot_0_equation.set_alpha(new_alpha)
        B_rot_1_equation_harmonic.set_alpha(new_alpha)
        # B_rot_1_equation_neglect.set_alpha(new_alpha)
    
    if t_7 < i <= t_8:
        new_alpha = (i-t_7)/(t_8-t_7)
        pretty_axis_B_rot.alpha = new_alpha
        B_arrow_rot_0.set_alpha(new_alpha)
        B_arrow_rot_1.set_alpha(new_alpha)

        # sphere_dict["bloch_rot"].frame_alpha = new_alpha*0.2
        # sphere_dict["bloch_rot"].font_alpha = new_alpha
        # sphere_dict["bloch_rot"].vector_alpha = [new_alpha, new_alpha]
        # sphere_dict["bloch_rot"].ax_eq_alpha = new_alpha
        # sphere_dict["bloch_rot"].sphere_alpha = new_alpha*0.2
        # sphere_dict["bloch_B_rot"].make_sphere()
    
    if t_9 < i <= t_10:
        new_alpha_neglect = (i-t_9)/(t_10-t_9)
        new_alpha_harmonic = 1 - new_alpha_neglect
        B_rot_1_equation_neglect.set_alpha(new_alpha_neglect)
        B_rot_1_equation_harmonic.set_alpha(new_alpha_harmonic)
    
    if t_11 < i <= t_12:
        new_alpha = (t_12 - i)/(t_12-t_11)
        B_arrow_rot_1.set_alpha(new_alpha)
        B_rot_1_equation_neglect.set_alpha(new_alpha)
        sphere_dict["bloch_rot"].vector_alpha = [1, new_alpha]


    return [ax for key, ax in ax_dict.items()] 

def init():
    return [ax for key, ax in ax_dict.items()]
    
ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init, 
                              blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/rot_frame_transformation_old.{file_type}', fps = 20 )