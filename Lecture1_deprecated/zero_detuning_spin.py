from anim_base import  cache_then_save_funcanimation, bloch_vector, PrettyAxis,   prepare_bloch_mosaic, rot_matrix, file_type
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm

##########################
# LONGER DURATIONS FOR ACTUAL ANIMATION
##########################

N_time = 340
t_0 = 20 # Show rot bloch sphere
t_1 = 40 # Show rot time evolution
t_2 = 60 # Show rot equation
t_3 = 130 # Let it sink in for a bit
t_4 = 150 # Show W transformation
t_5 = 170 # Let it sink in for a bit
t_6 = 190 # Show lab bloch sphere
t_7 = 200 # Let it sink in for a bit
t_8 = 220 # Show lab equation
t_9 = 320 # Let it sink in for a bit
t_10 = N_time

##########################
# SHORTER DURATIONS FOR DEBUGGING, COMMENT OUT WHEN DONE DEBUGGING
##########################

# N_time = N_time//10
# t_0 = t_0//10
# t_1 = t_1//10
# t_2 = t_2//10
# t_3 = t_3//10
# t_4 = t_4//10
# t_5 = t_5//10
# t_6 = t_6//10
# t_7 = t_7//10
# t_8 = t_8//10
# t_9 = t_9//10
# t_10 = t_10//10

bloch_mosaic = [["bloch_lab", "bloch_rot"],
                ["plot", "plot"]]

vector_colors = ["firebrick", "tomato", "blue"] #, "blue", "green", "purple", "orange", "pink", "yellow"]

bloch_kwargs = [{
    "vector_color": vector_colors,
    "vector_width": 6,
    "point_color": ["blue"],

    "vector_alpha": [0,0,0],
},
    {
    "vector_color": vector_colors[1:],
    "vector_width": 6,
    "point_color": ["blue"],

    "vector_alpha": [0,0],
    "xlabel": ["x'", ''],
    "ylabel": ["y'", '']
    }
]

gridspec_kw = {"height_ratios":[1,0.3], "width_ratios":[1,1]}
fig, ax_dict, sphere_dict = prepare_bloch_mosaic(bloch_mosaic, (10,6), bloch_kwargs, gridspec_kw=gridspec_kw)

ax_dict["plot"].set_axis_off()

B_x_max = 0.7
B_zeeman_lab_z = 1
# B_zeeman_rot_z = 0

phi_0 = 0
phi_end = 16.3*np.pi
phi = np.linspace(phi_0, phi_end, t_9-t_0)

B_zeeman_lab = np.zeros((len(phi), 3))
B_zeeman_lab[:,2] = B_zeeman_lab_z


B_drive_rot = np.zeros((len(phi), 3))
B_drive_rot[:,0] = B_x_max

spin_theta_0_rot_frame = 0
spin_phi_0_rot_frame = 0
spin_0_rot_frame = bloch_vector(spin_theta_0_rot_frame, spin_phi_0_rot_frame)

spin_precession_phi = np.linspace(0, 12*np.pi, len(phi))
spin_precession_axis = np.array([1,0,0])
spin_rot_matrix_rot_frame = rot_matrix(spin_precession_axis, spin_precession_phi)
spin_rot_frame = np.einsum("ijk,k->ij", spin_rot_matrix_rot_frame, spin_0_rot_frame)
sphere_dict["bloch_rot"].add_vectors([ B_drive_rot[0], spin_rot_frame[0] ])

rot_to_lab_transform_matrix = rot_matrix(np.array([0,0,1]), phi)
spin_lab_frame = np.einsum("ijk,ik->ij", rot_to_lab_transform_matrix, spin_rot_frame)
B_drive_lab = np.einsum("ijk,ik->ij", rot_to_lab_transform_matrix, B_drive_rot)


B_rot_H_text = ax_dict["plot"].text(1, 0.2, r"$H' \; =$", color = "black", alpha = 0, size = 18)
# B_rot_H_zeeman_text = ax_dict["plot"].text(0.9, 0.2, r"$H'_{\mathrm{Zeeman}} + $", color = vector_colors[0], alpha = 0, size = 18)
# B_rot_H_drive_text = ax_dict["plot"].text(1.4, 0.2, r"$H'_{\mathrm{driving}} \; = \;$", color = vector_colors[1], alpha = 0, size = 18)
# B_rot_H_zeeman_eq = ax_dict["plot"].text(0.75, 0.1, r"$( \omega_L - \omega_L )  S'_z + $", color = vector_colors[0], alpha = 0, size = 18)
B_rot_H_drive_eq = ax_dict["plot"].text(1.25, 0.2, r"$h S'_x$", color = vector_colors[1], alpha = 0, size = 18)
# B_rot_H_drive_eq_2 = ax_dict["plot"].text(1.18, 0.0, r"$h S'_x$", color = vector_colors[1], alpha = 0, size = 18)


# B_x_dec_equation = ax_dict["plot"].text(-0.6, 1.4, r'$H_{\mathrm{driving}} \; = \; 2hS_x \mathrm{cos}(\omega t)$', color = vector_colors[0], alpha = 0, size = 18)
B_lab_H_text = ax_dict["plot"].text(-1.6, 0.2, r'$H \; =$', color = "black", alpha = 0, size = 18)
# B_lab_H_zeeman_text = ax_dict["plot"].text(-0.95, 0.2, r'$H_{\mathrm{Zeeman}} + $', color = vector_colors[0], alpha = 0, size = 18)
# B_lab_H_drive_text = ax_dict["plot"].text(-0.47, 0.2, r"$H_{\mathrm{driving}} \; = \;$", color = vector_colors[1], alpha = 0, size = 18)
B_lab_H_zeeman_eq = ax_dict["plot"].text(-1.4, 0.2, r'$\omega_L S_z + $', color = vector_colors[0], alpha = 0, size = 18)
B_lab_H_drive_eq = ax_dict["plot"].text(-1.05, 0.2, r"$h [ S_x \: \mathrm{cos}(\omega_L t) + S_y \: \mathrm{sin}(\omega_L t) ]$", color = vector_colors[1], alpha = 0, size = 18)


W_trans_equation = ax_dict["plot"].text(-0.1, 0.05, r"$W = \mathrm{exp}( - i \omega_L t S_z)$", color = "black", alpha = 0, size = 17)
W_trans_arrow = FancyArrowPatch((-0.15, -0.0), (0.7, -0.0), 
        arrowstyle='->', mutation_scale=20,
        lw = 3, color = "black",
        alpha = 0
    )

ax_dict["plot"].add_patch(W_trans_arrow)
W_trans_arrow.set_zorder(10)

ax_dict["plot"].set_xlim(-1.7, 2.1)
ax_dict["plot"].set_ylim(-0.1, 0.25)

tail = int( np.ceil( np.pi/2 * len(spin_precession_phi) / spin_precession_phi[-1]) )
# print(tail)
# print(np.pi/3 * len(spin_precession_phi) / spin_precession_phi[-1])
def animate(i):
    
    if i <= t_0:
        new_alpha = i/t_0
        # pretty_axis_B_dec.alpha = new_alpha
        # B_arrow_dec_0.set_alpha(new_alpha)
        # B_arrow_dec_1.set_alpha(new_alpha)

        sphere_dict["bloch_rot"].frame_alpha = new_alpha*0.2
        sphere_dict["bloch_rot"].font_alpha = new_alpha
        sphere_dict["bloch_rot"].vector_alpha = [new_alpha, new_alpha]
        sphere_dict["bloch_rot"].ax_eq_alpha = new_alpha
        sphere_dict["bloch_rot"].sphere_alpha = new_alpha*0.2
        sphere_dict["bloch_rot"].make_sphere()

    if t_0 < i <= t_9:
        B_time_index = i - t_0 - 1
        # B_arrow_dec_0.set_positions((0,0), (B_dec[0,B_time_index,0], B_dec[0,B_time_index,1]))
        # B_arrow_dec_1.set_positions((0,0), (B_dec[1,B_time_index,0], B_dec[1,B_time_index,1]))
        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors( [B_drive_rot[B_time_index], spin_rot_frame[B_time_index] ])

        if tail > 0:
            sphere_dict["bloch_rot"].points = []
            if tail > B_time_index:
                sphere_dict["bloch_rot"].add_points( spin_rot_frame[:B_time_index+1].T, meth="l" )
            else:
                sphere_dict["bloch_rot"].add_points( spin_rot_frame[B_time_index-tail:B_time_index+1].T, meth="l" )

        sphere_dict["bloch_rot"].make_sphere()

        if i > t_5:
            # B_arrow_rot_0.set_positions((2.6,0), (B_rot[0,B_time_index,0] + 2.6, B_rot[0,B_time_index,1]))
            # B_arrow_rot_1.set_positions((2.6,0), (B_rot[1,B_time_index,0] + 2.6, B_rot[1,B_time_index,1]))
            sphere_dict["bloch_lab"].vectors = []
            sphere_dict["bloch_lab"].add_vectors([B_zeeman_lab[B_time_index], B_drive_lab[B_time_index], spin_lab_frame[B_time_index]]) 
            
            if tail > 0:
                sphere_dict["bloch_lab"].points = []
                if tail > B_time_index:
                    sphere_dict["bloch_lab"].add_points( spin_lab_frame[:B_time_index+1].T, meth="l" )
                else:
                    sphere_dict["bloch_lab"].add_points( spin_lab_frame[B_time_index-tail:B_time_index+1].T, meth="l" )
            
            sphere_dict["bloch_lab"].make_sphere()


    if t_1 < i <= t_2:
        new_alpha = (i-t_1)/(t_2-t_1)
        B_rot_H_text.set_alpha(new_alpha)
        B_rot_H_drive_eq.set_alpha(new_alpha)
        # B_lab_H_equation.set_alpha(new_alpha)

    if t_3 < i <= t_4:
        new_alpha = (i-t_3)/(t_4-t_3)
        W_trans_arrow.set_alpha(new_alpha)
        W_trans_equation.set_alpha(new_alpha)
    
    if t_5 < i <= t_6:
        new_alpha = (i-t_5)/(t_6-t_5)
        sphere_dict["bloch_lab"].frame_alpha = new_alpha*0.2
        sphere_dict["bloch_lab"].font_alpha = new_alpha
        sphere_dict["bloch_lab"].vector_alpha = [new_alpha, new_alpha, new_alpha]
        sphere_dict["bloch_lab"].ax_eq_alpha = new_alpha
        sphere_dict["bloch_lab"].sphere_alpha = new_alpha*0.2
        
    
    if t_7 < i <= t_8:
        new_alpha = (i-t_7)/(t_8-t_7)
        B_lab_H_text.set_alpha(new_alpha)
        B_lab_H_zeeman_eq.set_alpha(new_alpha)
        B_lab_H_drive_eq.set_alpha(new_alpha)
        
    return [ax for key, ax in ax_dict.items()] 

def init():
    return [ax for key, ax in ax_dict.items()]
    
ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init, 
                              blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/zero_detuning_spin.{file_type}', fps = 20 )