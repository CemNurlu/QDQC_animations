from anim_base import  cache_then_save_funcanimation, bloch_vector, prepare_bloch_mosaic, rot_matrix, math_fontfamily, file_type


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm

##########################
# LONGER DURATIONS FOR ACTUAL ANIMATION
##########################

N_time = 510
t_0 = 20 # Show bloch spheres and W transformation
t_1 = 80 # Show time evolution
t_2 = 100 # Show equation for H 
t_3 = 130 # Let sink in for a bit
t_4 = 200 # Show resultant magnetic field and fade out components
t_5 = 250 # Let sink in for a bit
t_6 = 280 # Fade in spin vector + time dynamics
t_7 = 500 # Continue time dynamics
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
    
bloch_mosaic = [["bloch_lab", "plot_between", "bloch_rot"],
                ["plot", "plot", "plot"]]

vector_colors = ["maroon", "hotpink", "red", "blue"]

bloch_kwargs = [{
    "vector_color": vector_colors,
    "vector_width": 6,

    "point_color": ["blue"],

    "vector_alpha": [1,1,0,0],
},
    {
    "vector_color": vector_colors,
    "vector_width": 6,

    "point_color": ["blue"],

    "vector_alpha": [1,1, 0,0],

    "xlabel": [r"$x^\prime$", ''],
    "ylabel": [r"$y^\prime$", '']
    }
]

gridspec_kw = {"height_ratios":[1,0.2], "width_ratios":[1, 0.5, 1]}
fig, ax_dict, sphere_dict = prepare_bloch_mosaic(bloch_mosaic, (10,6), bloch_kwargs, gridspec_kw=gridspec_kw)

ax_dict["plot"].set_axis_off()
ax_dict["plot_between"].set_axis_off()

B_x_max = 0.7
B_zeeman_lab_z = 1
B_zeeman_rot_z = -0.5

phi_0 = 0
phi_end = 16.3*np.pi
phi = np.linspace(phi_0, phi_end, t_7-t_0)

B_zeeman_lab = np.zeros((len(phi), 3))
B_zeeman_lab[:,2] = B_zeeman_lab_z


B_drive_rot = np.zeros((len(phi), 3))
B_drive_rot[:,0] = B_x_max

B_zeeman_rot = np.zeros((len(phi), 3))
B_zeeman_rot[:,2] = B_zeeman_rot_z

spin_theta_0_rot_frame = 0
spin_phi_0_rot_frame = 0
spin_0_rot_frame = bloch_vector(spin_theta_0_rot_frame, spin_phi_0_rot_frame)

spin_precession_phi = np.linspace(0, 12*np.pi, len(phi))
spin_precession_axis = np.array([B_x_max, 0, B_zeeman_rot_z]) / np.sqrt(B_x_max**2 + B_zeeman_rot_z**2)
spin_rot_matrix_rot_frame = rot_matrix(spin_precession_axis, spin_precession_phi)
spin_rot_frame = np.einsum("ijk,k->ij", spin_rot_matrix_rot_frame, spin_0_rot_frame)
sphere_dict["bloch_rot"].add_vectors([ B_zeeman_rot[0], B_drive_rot[0], spin_rot_frame[0] ])

rot_to_lab_transform_matrix = rot_matrix(np.array([0,0,1]), phi)
spin_lab_frame = np.einsum("ijk,ik->ij", rot_to_lab_transform_matrix, spin_rot_frame)
B_drive_lab = np.einsum("ijk,ik->ij", rot_to_lab_transform_matrix, B_drive_rot)


B_rot_H_text = ax_dict["plot"].text(0.7, 0.2, r"$H' \; =$", color = "black", alpha = 0, size = 18, math_fontfamily = math_fontfamily)
B_rot_H_zeeman_eq = ax_dict["plot"].text(0.95, 0.2, r"$( \omega_L - \omega )  S'_z + $", color = vector_colors[0], alpha = 0, size = 18, math_fontfamily = math_fontfamily)
B_rot_H_drive_eq = ax_dict["plot"].text(1.6, 0.2, r"$h S'_x$", color = vector_colors[1], alpha = 0, size = 18, math_fontfamily = math_fontfamily)


B_lab_H_text = ax_dict["plot"].text(-1.6, 0.2, r'$H \; =$', color = "black", alpha = 0, size = 18, math_fontfamily = math_fontfamily)
B_lab_H_zeeman_eq = ax_dict["plot"].text(-1.4, 0.2, r'$\omega_L S_z + $', color = vector_colors[0], alpha = 0, size = 18, math_fontfamily = math_fontfamily)
B_lab_H_drive_eq = ax_dict["plot"].text(-1.05, 0.2, r"$h [ S_x \: \mathrm{cos}(\omega_L t) + S_y \: \mathrm{sin}(\omega_L t) ]$", color = vector_colors[1], alpha = 0, size = 18, math_fontfamily = math_fontfamily)

W_trans_equation = ax_dict["plot_between"].text(0.1, 0.3, r'$W = \mathrm{exp}(-i \omega t S_z)$', color = "black", alpha = 1, size = 20, math_fontfamily = math_fontfamily)
W_trans_arrow = FancyArrowPatch((0.3, 0.5), (0.9, 0.5), 
        # arrowstyle='->',
        mutation_scale=120,
        lw = 2, 
        ec = "black",
        fc = "aquamarine",
        alpha = 1
    )


ax_dict["plot_between"].add_patch(W_trans_arrow)
W_trans_arrow.set_zorder(10)

omega_gt_omega_L = ax_dict["plot_between"].text(0.25, 0.7, r'$\omega > \omega_L$', color = "black", alpha = 1, size = 30, math_fontfamily = math_fontfamily)


ax_dict["plot"].set_xlim(-1.7, 2.1)
ax_dict["plot"].set_ylim(0.1, 0.25)

ax_dict["plot_between"].set_xlim(0,1)
ax_dict["plot_between"].set_ylim(0,1)


tail = int(10/(1 + int(DEBUG)))
def animate(i):
    
    if i == 0:
        sphere_dict["bloch_rot"].make_sphere()
        sphere_dict["bloch_lab"].make_sphere()

    if (t_0 < i <= t_3):
        B_time_index = i - t_0 - 1
        sphere_dict["bloch_lab"].vectors = []
        sphere_dict["bloch_lab"].add_vectors([B_zeeman_lab[B_time_index], B_drive_lab[B_time_index]])
        sphere_dict["bloch_lab"].make_sphere()

        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors( [B_zeeman_rot[B_time_index],B_drive_rot[B_time_index] ])
        sphere_dict["bloch_rot"].make_sphere()

    if t_1 < i <= t_2:
        new_alpha = (i - t_1) / (t_2 - t_1)
        B_rot_H_text.set_alpha(new_alpha)
        B_rot_H_drive_eq.set_alpha(new_alpha)
        B_rot_H_zeeman_eq.set_alpha(new_alpha)
        B_lab_H_text.set_alpha(new_alpha)
        B_lab_H_drive_eq.set_alpha(new_alpha)
        B_lab_H_zeeman_eq.set_alpha(new_alpha)

    if t_3 < i <= t_4:
        B_time_index = i - t_0 - 1
        new_alpha_res = (i - t_3) / (t_4 - t_3)
        new_alpha_comp = 1 - new_alpha_res
        
        sphere_dict["bloch_lab"].vector_alpha = [new_alpha_comp, new_alpha_comp, new_alpha_res, 0]
        sphere_dict["bloch_lab"].vectors = []
        sphere_dict["bloch_lab"].add_vectors( [B_zeeman_lab[B_time_index],
                                               B_drive_lab[B_time_index],                                                
                                                B_drive_lab[B_time_index] + B_zeeman_lab[B_time_index]]) 
        sphere_dict["bloch_lab"].make_sphere()

        sphere_dict["bloch_rot"].vector_alpha = [new_alpha_comp, new_alpha_comp, new_alpha_res, 0]
        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors( [B_zeeman_rot[B_time_index],
                                               B_drive_rot[B_time_index],                                                
                                                B_drive_rot[B_time_index] + B_zeeman_rot[B_time_index]]) 
        sphere_dict["bloch_rot"].make_sphere()

    if t_4 < i <= t_5:
        if i == t_4 + 1:
            sphere_dict["bloch_rot"].vector_color = [vector_colors[2], vector_colors[3]]
            sphere_dict["bloch_rot"].vector_alpha = [1, 0]
            sphere_dict["bloch_lab"].vector_color = [vector_colors[2], vector_colors[3]]
            sphere_dict["bloch_lab"].vector_alpha = [1, 0]


        B_time_index = i - t_0 - 1

        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors( B_zeeman_rot[B_time_index] + B_drive_rot[B_time_index])
        sphere_dict["bloch_rot"].make_sphere()
        
        sphere_dict["bloch_lab"].vectors = []
        sphere_dict["bloch_lab"].add_vectors( B_drive_lab[B_time_index] + B_zeeman_lab[B_time_index] ) 
        sphere_dict["bloch_lab"].make_sphere()

    if t_5 < i <= t_7:
        if i <= t_6:
            new_alpha = (i - t_5) / (t_6 - t_5)
            sphere_dict["bloch_rot"].vector_alpha = [1, new_alpha]
            sphere_dict["bloch_lab"].vector_alpha = [1, new_alpha]
            sphere_dict["bloch_lab"].point_alpha = [new_alpha]
            sphere_dict["bloch_rot"].point_alpha = [new_alpha]
        
        B_time_index = i - t_0 - 1
        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].points = []
        sphere_dict["bloch_rot"].add_vectors( [ B_drive_rot[B_time_index] + B_zeeman_rot[B_time_index], spin_rot_frame[B_time_index] ] )
        sphere_dict["bloch_rot"].add_points (  spin_rot_frame[B_time_index - tail: B_time_index].T, meth = "l")

        sphere_dict["bloch_rot"].make_sphere()
        
        sphere_dict["bloch_lab"].vectors = []
        sphere_dict["bloch_lab"].points = []
        sphere_dict["bloch_lab"].add_vectors( [ B_drive_lab[B_time_index] + B_zeeman_lab[B_time_index], spin_lab_frame[B_time_index] ]) 
        sphere_dict["bloch_lab"].add_points (  spin_lab_frame[B_time_index - tail: B_time_index].T, meth = "l")
        sphere_dict["bloch_lab"].make_sphere()
        
    return [ax for key, ax in ax_dict.items()] 

def init():
    return [ax for key, ax in ax_dict.items()]
    
ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init, 
                              blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/negative_detuning_spin.{file_type}', fps = 20 )