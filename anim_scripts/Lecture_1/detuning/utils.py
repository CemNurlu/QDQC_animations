import numpy as np
from matplotlib.patches import FancyArrowPatch
from anim_base import math_fontfamily

def initialize_ax_dict(ax_dict, settings):
    zeeman_color, driving_color, total_color = settings['vector_colors'][0], settings['vector_colors'][1], settings['vector_colors'][2]
    common_kwargs = {"math_fontfamily": math_fontfamily, "size": 20, "alpha": 0}\

    B_lab_H_text = ax_dict["plot"].text(-1.35, 0.2, r'$H \; =$', color="red", **common_kwargs)
    B_lab_H_zeeman_text = ax_dict["plot"].text(-1.13, 0.2, r'$H_{\mathrm{Zeeman}} + $', color=zeeman_color, **common_kwargs)
    B_lab_H_drive_text = ax_dict["plot"].text(-0.7, 0.2, r"$H_{\mathrm{driving}} \; = \;$", color = driving_color, **common_kwargs)
    B_lab_H_zeeman_eq = ax_dict["plot"].text(-1.55, 0.1, r'$=\omega_L S_z + $', color=zeeman_color, **common_kwargs)
    B_lab_H_drive_eq = ax_dict["plot"].text(-1.10, 0.1, r"$h [ S_x \: \mathrm{cos}(\omega t) + S_y \: \mathrm{sin}(\omega t) ]$", color=driving_color, **common_kwargs)
    B_lab_texts = (B_lab_H_text, B_lab_H_zeeman_text, B_lab_H_drive_text, B_lab_H_zeeman_eq, B_lab_H_drive_eq)

    B_rot_H_text = ax_dict["plot"].text(0.9, 0.2, r"$H \prime \; =$", color = "red", **common_kwargs)
    B_rot_H_zeeman_text = ax_dict["plot"].text(1.15, 0.2, r"$H \prime _{\mathrm{Zeeman}} + $", color = zeeman_color, **common_kwargs)
    B_rot_H_drive_text = ax_dict["plot"].text(1.6, 0.2, r"$H \prime _{\mathrm{driving}} \; = \;$", color = driving_color, **common_kwargs)
    B_rot_H_zeeman_eq = ax_dict["plot"].text(1.04, 0.1, r"$=( \omega_L - \omega )  S \prime _z + $", color = zeeman_color, **common_kwargs)
    B_rot_H_drive_eq_1 = ax_dict["plot"].text(1.75, 0.1, r"$h S \prime _x = $", color = driving_color, **common_kwargs)
    B_rot_H_drive_eq_2 = ax_dict["plot"].text(1.38, 0.0, r"$=h S \prime _x$", color = total_color, **common_kwargs)
    B_rot_texts = (B_rot_H_text, B_rot_H_zeeman_text, B_rot_H_drive_text, B_rot_H_zeeman_eq, B_rot_H_drive_eq_1, B_rot_H_drive_eq_2)

    W_trans_equation = ax_dict["plot_between"].text(0.1, 0.3, r'$W = \mathrm{exp}(-i \omega t S_z)$', color = "black", **common_kwargs)
    W_trans_arrow = FancyArrowPatch(
        (0.3, 0.5), (0.9, 0.5),
        mutation_scale=120,
        lw = 2,
        ec = "black",
        fc = "aquamarine",
        alpha = 0
    )
    omega_is_omega_L = ax_dict["plot_between"].text(0.25, 0.75, r'$\omega = \omega_L$', color = "black", alpha = 0, size = 30, math_fontfamily = math_fontfamily)
    transformation_texts = (W_trans_equation, W_trans_arrow, omega_is_omega_L)

    ax_dict["plot_between"].add_patch(W_trans_arrow)
    ax_dict["plot_between"].set_xlim(0, 1)
    ax_dict["plot_between"].set_ylim(0, 1)
    ax_dict["plot"].set_xlim(-1.6, 2.1)
    ax_dict["plot"].set_ylim(-0.15, 0.25)

    return B_lab_texts, B_rot_texts, transformation_texts

def fade_in_texts(i, t_start, t_end, text_list):
    alpha = (i-t_start)/(t_end-t_start)
    for text in text_list:
        text.set_alpha(alpha)

def calculate_labframe_B_fields(phi, settings):
    B_zeeman = np.zeros((len(phi), 3))
    B_drive = np.zeros((len(phi), 3))
    B_total = np.zeros((len(phi), 3))

    B_zeeman[:,2] = settings['B_zeeman_lab_z']
    B_drive[:,0] = settings['B_x_max'] * np.cos(phi)
    B_drive[:,1] = settings['B_x_max'] * np.sin(phi)

    B_total = B_zeeman + B_drive

    return B_drive, B_zeeman, B_total

def calculate_rotating_frame_B_fields(phi, settings):
    B_drive = np.zeros((len(phi), 3))
    B_drive[:,0] = settings['B_x_max']

    return B_drive

def update_bloch_sphere_vectors(i, sphere_dict, ax_dict, B_zeeman_lab, B_drive_lab, B_total_lab, B_drive_rot, azim_angle_rot_sphere, settings):
    t_0, t_1, t_2, t_8, t_11 = settings['time_list'][0], settings['time_list'][1], settings['time_list'][2], settings['time_list'][8], settings['time_list'][11]
    if t_1 < i <= t_2:
        new_alpha = (i-t_1)/(t_2-t_1)
        sphere_dict["bloch_lab"].vector_alpha = [1,1,new_alpha]

    B_time_index = i - t_0 - 1
    sphere_dict["bloch_lab"].vectors = []
    sphere_dict["bloch_lab"].add_vectors([B_zeeman_lab[B_time_index], B_drive_lab[B_time_index], B_total_lab[B_time_index]])
    sphere_dict["bloch_lab"].make_sphere()

    if t_8 < i <= t_11:
        coefficent = (i-t_8-1)/(t_11-t_8-1)
        B_zeeman_auxvector = (1-coefficent)*B_zeeman_lab[i]
        B_total_auxvector = (1-coefficent)*B_total_lab[0] + coefficent*B_drive_lab[0]
        B_zeeman_auxvector = np.sign(B_zeeman_auxvector)*np.maximum(np.abs(B_zeeman_auxvector), settings['arrow_length'])
        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors([B_zeeman_auxvector,B_drive_rot[B_time_index], B_total_auxvector])
        ax_dict["bloch_rot"].azim = azim_angle_rot_sphere[i]
        sphere_dict["bloch_rot"].make_sphere()


    if i > t_11:
        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors([[0,0,0,],[0,0,0],B_drive_rot[B_time_index]])
        sphere_dict["bloch_rot"].make_sphere()