from anim_base import  cache_then_save_funcanimation, bloch_vector, PrettyAxis,   prepare_bloch_mosaic, math_fontfamily, file_type
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm

##########################
# LONGER DURATIONS FOR ACTUAL ANIMATION
##########################

N_time = 540
t_0 = 20 # Show non-decomposed bloch sphere and pretty axis
t_1 = 150 # Show B_x time evolution
t_2 = 170 # Show B_x equation
t_3 = 220 # Let it sink in for a bit
t_4 = 240 # Show decomposed B_x equation
t_5 = 340 # Let it sink in for a bit
t_6 = 360 # Show decomposed bloch sphere and pretty axis
t_7 = 520 # Show decomposed B_x time evolution
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

bloch_mosaic = [["bloch_B", "bloch_B_dec"],
                ["plot", "plot"]]

vector_colors = ["purple", "hotpink", "deepskyblue"]

bloch_kwargs = [{
    "vector_color": vector_colors[0:1],
    "vector_width": 6,
    },
    {
    "vector_color": vector_colors[1:],
    "vector_width": 6,
    }
]


gridspec_kw = {"height_ratios":[1,1.2], "width_ratios":[1,1]}
fig, ax_dict, sphere_dict = prepare_bloch_mosaic(bloch_mosaic, (14,11.2), bloch_kwargs, gridspec_kw=gridspec_kw)

ax_dict["plot"].set_axis_off()


B_x_max = 1.3
phi_0 = 0
phi_end = 18.3*np.pi
phi = np.linspace(phi_0, phi_end, t_7-t_0)


B_x_bloch = B_x_max*np.cos(phi)
B_x_pretty_axis = np.cos(phi)

# Clip values less than 0.1 to 0.1 to make arrow heads look better
arrow_head_threshold_bloch = 0.2
small_B_bloch = np.abs(B_x_bloch) < arrow_head_threshold_bloch
B_x_bloch[small_B_bloch] = np.sign(B_x_bloch[small_B_bloch])*arrow_head_threshold_bloch

arrow_head_threshold_pretty_axis = 0.1
small_B_pretty_axis = np.abs(B_x_pretty_axis) < arrow_head_threshold_pretty_axis
B_x_pretty_axis[small_B_pretty_axis] = np.sign(B_x_pretty_axis[small_B_pretty_axis])*arrow_head_threshold_pretty_axis

B_dec = np.zeros((2,t_7-t_0, 3))

# Divide by 1.5 instead of 2 to make animation look better
B_dec[:,:,0] = B_x_max/2 * np.cos(phi)
B_dec[0,:,1] = B_x_max/2 * np.sin(phi)
B_dec[1,:,1] = -B_x_max/2 * np.sin(phi)
# exit()

sphere_dict["bloch_B"].add_vectors([B_x_bloch[0], 0, 0])


pretty_axis_B = PrettyAxis(ax_dict["plot"], (-1.1, 1.1, 0), (-1.1, 1.1, 0),
                        alpha = 0)

pretty_axis_B.add_label(r'$B_x(t)$  ', "x", size = 25)
pretty_axis_B.add_label(r'$B_y(t)$  ', "y", size = 25)


pretty_axis_B_dec = PrettyAxis(ax_dict["plot"], (1.5, 3.7, 0), (-1.1, 1.1, 2.6),
                        alpha = 0)

pretty_axis_B_dec.add_label(r'$B_x(t)$  ', "x", size = 25)
pretty_axis_B_dec.add_label(r'$B_y(t)$  ', "y", size = 25)

B_arrow = FancyArrowPatch((0,0), (B_x_pretty_axis[0],0), 
        arrowstyle='-|>', mutation_scale=20,
        lw = 5, color = vector_colors[0],
        alpha = 0
    )

B_arrow_dec_0 = FancyArrowPatch((2.1,0), (B_dec[0,0,0] + 2.6, B_dec[0,0,1]), 
        arrowstyle='-|>', mutation_scale=20,
        lw = 5, color = vector_colors[1],
        alpha = 0
    )

B_arrow_dec_1 = FancyArrowPatch((2.1,0), (B_dec[1,0,1] + 2.6, B_dec[1,0,1]), 
        arrowstyle='-|>', mutation_scale=20,
        lw = 5, color = vector_colors[2],
        alpha = 0
    )

for arrow in [B_arrow, B_arrow_dec_0, B_arrow_dec_1]:
    ax_dict["plot"].add_patch(arrow)
    arrow.set_zorder(10)

B_x_equation = ax_dict["plot"].text(-0.6, 1.52, r'$H_{\mathrm{driving}} \; = \; 2hS_x \mathrm{cos}(\omega t)$', color = vector_colors[0], alpha = 0, size = 30, math_fontfamily = math_fontfamily)
B_x_dec_0_equation = ax_dict["plot"].text(1.75, 1.65, r'$= \; h \: [ S_x \: \mathrm{cos}(\omega t) + S_y \: \mathrm{sin}(\omega t) ]$', color = vector_colors[1], alpha = 0, size = 30, math_fontfamily = math_fontfamily)
B_x_dec_1_equation = ax_dict["plot"].text(1.8, 1.4, r'$+ \; h \: [ S_x \: \mathrm{cos}(\omega t) - S_y \: \mathrm{sin}(\omega t) ]$', color = vector_colors[2], alpha = 0, size = 30, math_fontfamily = math_fontfamily)

ax_dict["plot"].set_xlim(-1.2, 3.85)
ax_dict["plot"].set_ylim(-1.2, 1.6)

def animate(i):
    # ax_dict["bloch_avg"].set_title("Average of Bloch vectors")
    # ax_dict["bloch_super"].set_title("Superposition of Bloch vectors")
    if i <= t_0:
        new_alpha = i/t_0
        pretty_axis_B.alpha = new_alpha
        B_arrow.set_alpha(new_alpha)


        # sphere_dict["bloch_B"].frame_alpha = new_alpha*0.2
        # sphere_dict["bloch_B"].font_alpha = new_alpha
        # sphere_dict["bloch_B"].vector_alpha = [new_alpha]
        # sphere_dict["bloch_B"].ax_eq_alpha = new_alpha
        # sphere_dict["bloch_B"].sphere_alpha = new_alpha*0.2
        sphere_dict["bloch_B"].make_sphere()

    if t_0 < i <= t_7:
        B_time_index = i - t_0 - 1
        B_arrow.set_positions((0,0), (B_x_pretty_axis[B_time_index], 0))
        sphere_dict["bloch_B"].vectors = []
        sphere_dict["bloch_B"].add_vectors([B_x_bloch[B_time_index], 0, 0])
        sphere_dict["bloch_B"].make_sphere()

        if i > t_5:
            B_arrow_dec_0.set_positions((2.6,0), (B_dec[0,B_time_index,0] + 2.6, B_dec[0,B_time_index,1]))
            B_arrow_dec_1.set_positions((2.6,0), (B_dec[1,B_time_index,0] + 2.6, B_dec[1,B_time_index,1]))
            sphere_dict["bloch_B_dec"].vectors = []
            sphere_dict["bloch_B_dec"].add_vectors( [B_dec[0,B_time_index], B_dec[1,B_time_index]])
            sphere_dict["bloch_B_dec"].make_sphere()

    if t_1 < i <= t_2:
        new_alpha = (i-t_1)/(t_2-t_1)
        B_x_equation.set_alpha(new_alpha)
    

    if t_3 < i <= t_4:
        new_alpha = (i-t_3)/(t_4-t_3)
        B_x_dec_0_equation.set_alpha(new_alpha)
        B_x_dec_1_equation.set_alpha(new_alpha)

    if t_5 < i <= t_6:
        new_alpha = (i-t_5)/(t_6-t_5)
        B_arrow_dec_0.set_alpha(new_alpha)
        B_arrow_dec_1.set_alpha(new_alpha)

        pretty_axis_B_dec.alpha = new_alpha

        # sphere_dict["bloch_B_dec"].frame_alpha = new_alpha*0.2
        # sphere_dict["bloch_B_dec"].font_alpha = new_alpha
        # sphere_dict["bloch_B_dec"].vector_alpha = [new_alpha, new_alpha]
        # sphere_dict["bloch_B_dec"].ax_eq_alpha = new_alpha
        # sphere_dict["bloch_B_dec"].sphere_alpha = new_alpha*0.2
        # sphere_dict["bloch_B_dec"].make_sphere()

    return [ax for key, ax in ax_dict.items()] 

def init():
    return [ax for key, ax in ax_dict.items()]
    


ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init, 
                              blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/decomposed_B_x.{file_type}', fps = 20 )