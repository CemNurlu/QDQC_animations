import sys
import os

# Get the absolute path of the root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
print(project_root)

# Add project root to sys.path
sys.path.append(project_root)

from anim_base import  cache_then_save_funcanimation, bloch_vector, PrettyAxis,   prepare_bloch_mosaic, math_fontfamily, file_type
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm

##########################
# LONGER DURATIONS FOR ACTUAL ANIMATION
##########################

N_time = 550
t_0 = 10  # Show bloch spheres
t_1 = 80 # Show time evolution
t_2 = 100# Show omega1 = omega2 equation
t_3 = 130# Let it sink in for a bit
t_4 = 150# Show axises and start plotting lines
t_5 = 210# continue time evolution
t_6 = 230 # Remove axises and show coupling equations
t_7 = 290# let it sink in 
t_8 = 310# Remove everything and show omega_1 transformation equation
t_9 = 340# let it sink in
t_10 = 360# Show new bloch spheres and axises
t_11 = 530# time evolution
t_12 = 550# Remove axes and show coupling equations
t_13 = N_time

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


#assert t_5 - t_0 == t_11 - t_10, "Time evolution must be the same length"


bloch_mosaic = [["bloch_1", "bloch_2"],
                ["plot", "plot"]]


spin1_vector_colors = ["maroon", "red"] 
spin2_vector_colors = ["maroon", "deepskyblue"] 

bloch_kwargs = [{
    "vector_color": spin1_vector_colors,
    "vector_width": 6,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    "vector_alpha": [0,0],
    "point_color": ["red"],
    "point_marker": ["o"]
    },

    {
    "vector_color": spin2_vector_colors,
    "vector_width": 6,
    "vector_alpha": [0,0],
    "point_color": ["deepskyblue"],
    "point_marker": ["o"]
    }
]


gridspec_kw = {"height_ratios":[1,2], "width_ratios":[1,1]}
fig, ax_dict, sphere_dict = prepare_bloch_mosaic(bloch_mosaic, (8,12), bloch_kwargs, gridspec_kw=gridspec_kw)

fig.subplots_adjust(top=0.98)

ax_dict["plot"].set_axis_off()

box = ax_dict["bloch_1"].get_position()
ax_dict["bloch_1"].set_position([box.x0, box.y0, box.width, box.height * 1.8])
box = ax_dict["bloch_2"].get_position()
ax_dict["bloch_2"].set_position([box.x0, box.y0, box.width, box.height * 1.8])

B_time = np.linspace(0, 1, t_5-t_0)
omega_1_lab = 8*np.pi
omega_2_lab = 21*np.pi

phi_1_start = np.pi/3
phi_2_start = np.pi*2/3

phi_1_lab = omega_1_lab*B_time + phi_1_start
phi_2_lab = omega_2_lab*B_time + phi_2_start
theta = np.pi/5

B_z = np.array([0,0,1])

bloch_1_lab = bloch_vector(theta, phi_1_lab)
bloch_2_lab = bloch_vector(theta, phi_2_lab)


omega_1_rot = 0
omega_2_rot = omega_2_lab - omega_1_lab

phi_1_rot = omega_1_rot*B_time + phi_1_start
phi_2_rot = omega_2_rot*B_time + phi_2_start

bloch_1_rot = bloch_vector(theta, phi_1_rot)
bloch_2_rot = bloch_vector(theta, phi_2_rot)

tail = len(B_time)//12

# Define pretty plots
if True:

    pretty_axis_spin1_x = PrettyAxis(ax_dict["plot"], (0,2, -1), (-2, 0, 0),
                            data_x_lim=(0,1),
                            data_y_lim=(-1, 1),
                            alpha = 0)

    pretty_axis_spin1_x.add_label(r"$S^x_1$  ", "y", size=21)
    pretty_axis_spin1_x.add_label(r" $t$", "x", size=21)

    pretty_axis_spin1_y = PrettyAxis(ax_dict["plot"], (0,2, -3.2), (-4.2, -2.2, 0),
                            data_x_lim=(0,1),
                            data_y_lim=(-1, 1),
                            alpha = 0)

    pretty_axis_spin1_y.add_label(r"$S^y_1$  ", "y", size=21)
    pretty_axis_spin1_y.add_label(r" $t$", "x", size=21)

    pretty_axis_spin1_z = PrettyAxis(ax_dict["plot"], (0,2, -5.4), (-6.4, -4.4, 0),
                            data_x_lim=(0,1),
                            data_y_lim=(-1, 1),
                            alpha = 0)

    pretty_axis_spin1_z.add_label(r"$S^z_1$  ", "y", size=21)
    pretty_axis_spin1_z.add_label(r" $t$", "x", size=21)

    pretty_axis_spin2_x = PrettyAxis(ax_dict["plot"], (2.5, 4.5, -1), (-2, 0, 2.5),
                            data_x_lim=(0,1),
                            data_y_lim=(-1, 1),
                            alpha = 0)

    pretty_axis_spin2_x.add_label(r"$S^x_2$  ", "y", size=21)
    pretty_axis_spin2_x.add_label(r" $t$", "x", size=21)

    pretty_axis_spin2_y = PrettyAxis(ax_dict["plot"], (2.5, 4.5, -3.2), (-4.2, -2.2, 2.5),
                            data_x_lim=(0,1),
                            data_y_lim=(-1, 1),
                            alpha = 0)

    pretty_axis_spin2_y.add_label(r"$S^y_2$  ", "y", size=21)
    pretty_axis_spin2_y.add_label(r" $t$", "x", size=21)

    pretty_axis_spin2_z = PrettyAxis(ax_dict["plot"], (2.5, 4.5, -5.4), (-6.4, -4.4, 2.5),
                            data_x_lim=(0,1),
                            data_y_lim=(-1, 1),
                            alpha = 0)

    pretty_axis_spin2_z.add_label(r"$S^z_2$  ", "y", size=21)
    pretty_axis_spin2_z.add_label(r" $t$", "x", size=21)


anim_title = ax_dict["plot"].text(1.5, 3.3,"Unlike Spins", color = "black", alpha = 0, size = 30, math_fontfamily = math_fontfamily)
omega_equation = ax_dict["plot"].text(1.9, 2.6, r"$\omega_1 \neq \omega_2$", color = "black", alpha = 0, size = 30, math_fontfamily = math_fontfamily)
coupling_equation_lab = ax_dict["plot"].text(0.8, -0.5, r"$H_c = S^z_1 S^z_2 - \frac{1}{2} ( S^y_1 S^y_2 + S^z_1 S^z_2)$", color = "black", alpha = 0, size = 30, math_fontfamily = math_fontfamily)
coupling_equation_rot = ax_dict["plot"].text(1.85, -0.5, r"$H_c^\prime = S^z_1 S^z_2$", color = "black", alpha = 0, size = 30, math_fontfamily = math_fontfamily)

W_trans_arrow = FancyArrowPatch((2.05, 1.7), (2.75, 1.7), 
        # arrowstyle='->',
        mutation_scale=120,
        lw = 2, 
        ec = "black",
        fc = "aquamarine",
        alpha = 0
    )
W_trans_equation = ax_dict["plot"].text(1.6, 0.6, r'$W = \mathrm{exp}(-i \omega_1 t S_z)$', color = "black", alpha = 0, size = 25, math_fontfamily = math_fontfamily)
W_trans_equation.set_zorder(1)


ax_dict["plot"].add_patch(W_trans_arrow)
W_trans_arrow.set_zorder(10)

ax_dict["plot"].set_xlim(-0.3, 4.7)
ax_dict["plot"].set_ylim(-6.6, 2.3)


def animate(i):
    
    if i == 0:
        sphere_dict["bloch_1"].add_vectors([B_z, bloch_1_lab[0]])
        sphere_dict["bloch_2"].add_vectors([B_z, bloch_2_lab[0]])

    if i <= t_0:


        new_alpha = i/t_0
        for sphere in sphere_dict.values():
            sphere.vector_alpha = [new_alpha, new_alpha]
            sphere.make_sphere()





    if t_0 < i <= t_5:

        
        B_time_index = i - t_0 - 1
        sphere_dict["bloch_1"].vectors = []
        sphere_dict["bloch_1"].points = []
        sphere_dict["bloch_1"].add_vectors([B_z, bloch_1_lab[B_time_index]])

        if B_time_index < tail or tail == -1:
            points_1 = bloch_1_lab[:B_time_index+1]
        else:
            points_1 = bloch_1_lab[B_time_index+1-tail:B_time_index+1]
        if B_time_index > 0:
            sphere_dict["bloch_1"].add_points([points_1[:,0], points_1[:,1], points_1[:,2]], meth="l")
        sphere_dict["bloch_1"].make_sphere()

        sphere_dict["bloch_2"].vectors = []
        sphere_dict["bloch_2"].points = []
        sphere_dict["bloch_2"].add_vectors([B_z, bloch_2_lab[B_time_index]])

        if B_time_index < tail or tail == -1:
            points_2 = bloch_2_lab[:B_time_index+1]
        else:
            points_2 = bloch_2_lab[B_time_index+1-tail:B_time_index+1]
        if B_time_index > 0:
            sphere_dict["bloch_2"].add_points([points_2[:,0], points_2[:,1], points_2[:,2]], meth="l")
        sphere_dict["bloch_2"].make_sphere()


        if i > t_3:
            reverse_rot = -np.degrees(phi_2_lab[B_time_index])
            ax_dict["bloch_2"].view_init(elev=30, azim=-reverse_rot)

        if i == t_3:
            pretty_axis_spin1_x.add_line("spin1_x", B_time[:B_time_index+1], bloch_1_lab[:B_time_index+1,0], c = spin1_vector_colors[1])
            pretty_axis_spin1_y.add_line("spin1_y", B_time[:B_time_index+1], bloch_1_lab[:B_time_index+1,1], c = spin1_vector_colors[1])
            pretty_axis_spin1_z.add_line("spin1_z", B_time[:B_time_index+1], bloch_1_lab[:B_time_index+1,2], c = spin1_vector_colors[1])

            pretty_axis_spin2_x.add_line("spin2_x", B_time[:B_time_index+1], bloch_2_lab[:B_time_index+1,0], c = spin2_vector_colors[1])
            pretty_axis_spin2_y.add_line("spin2_y", B_time[:B_time_index+1], bloch_2_lab[:B_time_index+1,1], c = spin2_vector_colors[1])
            pretty_axis_spin2_z.add_line("spin2_z", B_time[:B_time_index+1], bloch_2_lab[:B_time_index+1,2], c = spin2_vector_colors[1])

            pretty_axis_spin1_x.alpha = 0
            pretty_axis_spin1_y.alpha = 0
            pretty_axis_spin1_z.alpha = 0

            pretty_axis_spin2_x.alpha = 0
            pretty_axis_spin2_y.alpha = 0
            pretty_axis_spin2_z.alpha = 0
        
        if t_3 < i:
            pretty_axis_spin1_x.update_line("spin1_x", B_time[:B_time_index+1], bloch_1_lab[:B_time_index+1,0])
            pretty_axis_spin1_y.update_line("spin1_y", B_time[:B_time_index+1], bloch_1_lab[:B_time_index+1,1])
            pretty_axis_spin1_z.update_line("spin1_z", B_time[:B_time_index+1], bloch_1_lab[:B_time_index+1,2])

            pretty_axis_spin2_x.update_line("spin2_x", B_time[:B_time_index+1], bloch_2_lab[:B_time_index+1,0])
            pretty_axis_spin2_y.update_line("spin2_y", B_time[:B_time_index+1], bloch_2_lab[:B_time_index+1,1])
            pretty_axis_spin2_z.update_line("spin2_z", B_time[:B_time_index+1], bloch_2_lab[:B_time_index+1,2])
    
    if t_1 < i <= t_2:
        new_alpha = (i - t_1)/(t_2-t_1)
        anim_title.set_alpha(new_alpha)
        omega_equation.set_alpha(new_alpha)
    
    if t_3 < i <= t_4:
        new_alpha = (i - t_3)/(t_4-t_3)
        pretty_axis_spin1_x.alpha = new_alpha
        pretty_axis_spin1_y.alpha = new_alpha
        pretty_axis_spin1_z.alpha = new_alpha

        pretty_axis_spin2_x.alpha = new_alpha
        pretty_axis_spin2_y.alpha = new_alpha
        pretty_axis_spin2_z.alpha = new_alpha
    
    if t_5 < i <= t_6:
        new_alpha = (i - t_5)/(t_6-t_5)
        new_alpha_axises = 1 - new_alpha
        coupling_equation_lab.set_alpha(new_alpha)

        pretty_axis_spin1_x.alpha = new_alpha_axises
        pretty_axis_spin1_y.alpha = new_alpha_axises
        pretty_axis_spin1_z.alpha = new_alpha_axises

        pretty_axis_spin2_x.alpha = new_alpha_axises
        pretty_axis_spin2_y.alpha = new_alpha_axises
        pretty_axis_spin2_z.alpha = new_alpha_axises

    if t_7 < i <= t_8:
        new_alpha = (t_8 - i)/(t_8-t_7)
        new_alpha_transformation = 1 - new_alpha

        if i == int((t_7 + t_8)/2):
            ax_dict["bloch_1"].clear()
            ax_dict["bloch_1"].grid(False)
            ax_dict["bloch_1"].set_axis_off()

            ax_dict["bloch_2"].clear()
            ax_dict["bloch_2"].grid(False)
            ax_dict["bloch_2"].set_axis_off()


        coupling_equation_lab.set_alpha(new_alpha)
        # omega_equation.set_alpha(new_alpha)

        W_trans_arrow.set_alpha(new_alpha_transformation)
        W_trans_equation.set_alpha(new_alpha_transformation)

    if i == int((t_9+t_10)/2):
        sphere_dict["bloch_1"].vectors = []
        sphere_dict["bloch_2"].vectors = []
        sphere_dict["bloch_1"].points = []
        sphere_dict["bloch_2"].points = []

        sphere_dict["bloch_1"].add_vectors( [B_z, bloch_1_rot[0]])
        sphere_dict["bloch_2"].add_vectors( [B_z, bloch_2_rot[0]])

        sphere_dict["bloch_1"].xlabel = [r"$x_1^\prime$", '']
        sphere_dict["bloch_1"].ylabel = [r"$y_1^\prime$", '']

        sphere_dict["bloch_2"].xlabel = [r"$x_1^\prime$", '']
        sphere_dict["bloch_2"].ylabel = [r"$y_1^\prime$", '']

        sphere_dict["bloch_1"].make_sphere()
        sphere_dict["bloch_2"].make_sphere()
        
    if t_9 < i <= t_10:
        if i == t_9 + 1:
            pretty_axis_spin1_x.update_line("spin1_x", B_time[:1], bloch_1_lab[:1,0])
            pretty_axis_spin1_y.update_line("spin1_y", B_time[:1], bloch_1_lab[:1,1])
            pretty_axis_spin1_z.update_line("spin1_z", B_time[:1], bloch_1_lab[:1,2])

            pretty_axis_spin2_x.update_line("spin2_x", B_time[:1], bloch_2_lab[:1,0])
            pretty_axis_spin2_y.update_line("spin2_y", B_time[:1], bloch_2_lab[:1,1])
            pretty_axis_spin2_z.update_line("spin2_z", B_time[:1], bloch_2_lab[:1,2])
        
        new_alpha = (i - t_9)/(t_10-t_9)
        pretty_axis_spin1_x.alpha = new_alpha
        pretty_axis_spin1_y.alpha = new_alpha
        pretty_axis_spin1_z.alpha = new_alpha

        pretty_axis_spin2_x.alpha = new_alpha
        pretty_axis_spin2_y.alpha = new_alpha
        pretty_axis_spin2_z.alpha = new_alpha

    if t_10 < i <= t_11:
        B_time_index = i - t_10 - 1

        pretty_axis_spin1_x.update_line("spin1_x", B_time[:B_time_index+1], bloch_1_rot[:B_time_index+1,0])
        pretty_axis_spin1_y.update_line("spin1_y", B_time[:B_time_index+1], bloch_1_rot[:B_time_index+1,1])
        pretty_axis_spin1_z.update_line("spin1_z", B_time[:B_time_index+1], bloch_1_rot[:B_time_index+1,2])

        pretty_axis_spin2_x.update_line("spin2_x", B_time[:B_time_index+1], bloch_2_rot[:B_time_index+1,0])
        pretty_axis_spin2_y.update_line("spin2_y", B_time[:B_time_index+1], bloch_2_rot[:B_time_index+1,1])
        pretty_axis_spin2_z.update_line("spin2_z", B_time[:B_time_index+1], bloch_2_rot[:B_time_index+1,2])
        
        sphere_dict["bloch_1"].vectors = []
        sphere_dict["bloch_2"].vectors = []
        sphere_dict["bloch_1"].points = []
        sphere_dict["bloch_2"].points = []

        sphere_dict["bloch_1"].add_vectors( [B_z, bloch_1_rot[B_time_index]])
        sphere_dict["bloch_2"].add_vectors( [B_z, bloch_2_rot[B_time_index]])

        if B_time_index < tail or tail == -1:
            points_1 = bloch_1_rot[:B_time_index+1]
            points_2 = bloch_2_rot[:B_time_index+1]
        else:
            points_1 = bloch_1_rot[B_time_index+1-tail:B_time_index+1]
            points_2 = bloch_2_rot[B_time_index+1-tail:B_time_index+1]
        if B_time_index > 0:
            sphere_dict["bloch_1"].add_points([points_1[:,0], points_1[:,1], points_1[:,2]], meth="l")
            sphere_dict["bloch_2"].add_points([points_2[:,0], points_2[:,1], points_2[:,2]], meth="l")

        sphere_dict["bloch_1"].xlabel = [r"$x_1^\prime$", '']
        sphere_dict["bloch_1"].ylabel = [r"$y_1^\prime$", '']

        sphere_dict["bloch_2"].xlabel = [r"$x_1^\prime$", '']
        sphere_dict["bloch_2"].ylabel = [r"$y_1^\prime$", '']

        sphere_dict["bloch_1"].make_sphere()
        sphere_dict["bloch_2"].make_sphere()


    if t_11 < i <= t_12:
        new_alpha = (i - t_11)/(t_12-t_11)
        new_alpha_axises = 1 - new_alpha
        coupling_equation_rot.set_alpha(new_alpha)

        pretty_axis_spin1_x.alpha = new_alpha_axises
        pretty_axis_spin1_y.alpha = new_alpha_axises
        pretty_axis_spin1_z.alpha = new_alpha_axises

        pretty_axis_spin2_x.alpha = new_alpha_axises
        pretty_axis_spin2_y.alpha = new_alpha_axises
        pretty_axis_spin2_z.alpha = new_alpha_axises
   

    return [ax for key, ax in ax_dict.items()]

def init():
    return [ax for key, ax in ax_dict.items()]
    


ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval= 50,
                              init_func=init, 
                              blit=False, repeat=False)

if not DEBUG:
    cache_then_save_funcanimation(ani, f'animations/test/bloch_coupling_different_omega_trans1V2_new.{file_type}', fps = 20)
else:
    cache_then_save_funcanimation(ani, f'animations/test/bloch_coupling_different_omega_trans1V2_new_debug.{file_type}', fps = 20)