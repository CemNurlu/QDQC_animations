import numpy as np
import qutip
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from anim_scripts.Lecture_1.rabi_driving.utils import (
    initialize_ax_dict,
    fade_in_texts,
    fade_out_texts,
    fade_in_axes,
    fade_out_axes,
    interpolate_between,
    calculate_labframe_B_fields,
    calculate_rotating_frame_B_fields,
    update_bloch_sphere_vectors,
    calculate_bloch_vectors,
    # calculate_bloch_vectors_rot,
)
from anim_base import (
    cache_then_save_funcanimation,
    prepare_bloch_mosaic,
    bloch_vector,
    PrettyAxis,
    file_type
)

##########################
# LONGER DURATIONS FOR ACTUAL ANIMATION
##########################
N_time = 750
t_0 = 20 # Lab bloch sphere is fully on the screen
t_1 = 70 # Total Hamiltonian arrow and first row text start fading in
t_2 = 90 # Total Hamiltonian arrow and first row text stop fading in
t_3 = 160 # Second row equations starts fading in
t_4 = 180 # Second row equations stop fading in
t_5 = 220 # Rotating frame arrow and text start fading in
t_6 = 240 # Rotating frame arrow an text stop fading in
t_7 = 280 # Rotating frame total Hamiltonian equation first row and bloch sphere start fading in
t_8 = 300 # Rotating frame total Hamiltonian equation first row and bloch sphere stop fading in, bloch vectors start evolving
t_9 = 340 # Second and third row equation rotating frame start fading in
t_10 = 360 # Second and third row equation rotating frame stop fading in
t_11 = 380 # Bloch vectors arrive at their final position
t_12 = 440 # Lab bloch sphere, equations and arrow start fading out
t_13 = 470 # Lab bloch sphere, equations and arrow stop fading out, rotating frame bloch sphere starts moving (and stops rotating)
t_14 = 520 # Rotating frame bloch sphere stops moving, pretty axes start fading, spin vector in bloch sphere starts appearing
t_15 = 550 # Pretty axes stop fading, spin vector in bloch sphere starts rotating and plots start getting drawn
t_16 = 690 # Spin vector in bloch sphere stops rotating and plots stop getting drawn, spin statistics start fading out
t_17 = 720 # Spin statistics stop fading out, lab bloch sphere starts fading in
t_18 = 740 # Lab bloch sphere stops fading in
N_time = 950
time_list = (t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10, t_11, t_12, t_13, t_14, t_15, t_16, t_17, t_18)

##########################
# SHORTER DURATIONS FOR DEBUGGING
##########################
DEBUG = False
show_time_list = False
if DEBUG:
    N_time //= 10
    time_list = [t // 10 for t in time_list]
    t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10, t_11, t_12, t_13, t_14, t_15, t_16, t_17, t_18 = time_list


bloch_mosaic = [["bloch_lab", "plot_between", "bloch_rot"],
                ["plot", "plot", "plot"]]

settings = {
    "B_x_max": 0.7,
    "B_zeeman_lab_z": 1,
    "phi_0": np.pi/4,
    "phi_end_lab": 240*np.pi,
    "phi_end_rot": 5000*np.pi,
    "arrow_length": 0.2,
    "time_list": time_list,
    "vector_colors": ["maroon", "hotpink", "red"],
    "intial_sphere_alpha": 0.2,
    "initial_frame_alpha": 0.2,
    "intial_font_alpha": 1,
    "vector_rotation_speed_rad": 24/528 * np.pi,
    "rabi_frequency": 5/100 * np.pi,
    "detuning": 0, #- 10/528 * np.pi,
    "spin_vector_color": "blue",
    "tail_length": 20,
    "detuning_plot_size":  - 0.3,
    "initial_bloch_vector": "0",
    "counter_rotating_vector_color": "deepskyblue",
}

bloch_kwargs = [{
    "vector_color": settings['vector_colors'],
    "vector_alpha" : [1,1,0],
    "vector_width": 3,
    },
    {
    "vector_color": settings['vector_colors'][:],
    "vector_width": 3,
    "xlabel": [r"$x^\prime$", ''],
    "ylabel": [r"$y^\prime$", '']
    }
]

gridspec_kw = {"height_ratios":[1,0.4], "width_ratios":[1, 0.5, 1]}
fig, ax_dict, sphere_dict = prepare_bloch_mosaic(bloch_mosaic, (10,6), bloch_kwargs, gridspec_kw=gridspec_kw)

ax_dict["plot"].set_axis_off()
ax_dict["plot_between"].set_axis_off()

phi = np.arange(settings["phi_0"], settings['phi_end_lab'], settings['vector_rotation_speed_rad'])
phi_rot = np.arange(settings["phi_0"], settings['phi_end_rot'], settings['vector_rotation_speed_rad'])
azim_angle_rot_sphere = (-60 - phi_rot[:] * 180 / np.pi) % 360
B_drive_lab, B_drive_counter_lab, B_zeeman_lab, B_total_lab = calculate_labframe_B_fields(phi, settings)
B_drive_rot, B_drive_counter_rot, B_zeeman_rot, B_total_rot = calculate_rotating_frame_B_fields(phi_rot, settings)

sphere_dict["bloch_lab"].vectors = []
sphere_dict["bloch_lab"].add_vectors([B_zeeman_lab[0], B_drive_lab[0], B_drive_counter_lab[0]])
sphere_dict["bloch_lab"].vector_color = [settings['vector_colors'][0], settings['vector_colors'][1], settings['counter_rotating_vector_color']]
sphere_dict["bloch_lab"].vector_alpha = [0, 0, 0]

omega_0 = settings['vector_rotation_speed_rad']
omega_d = omega_0 + settings['detuning']
if settings['initial_bloch_vector'] == "0":
    psi_0 = qutip.basis(2, 0)
elif settings['initial_bloch_vector'] == "1":
    psi_0 = qutip.basis(2, 1)
elif settings['initial_bloch_vector'] == "+":
    psi_0 = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
elif settings['initial_bloch_vector'] == "-":
    psi_0 = (qutip.basis(2, 0) - qutip.basis(2, 1)).unit()
else:
    psi_0 = qutip.basis(2, 0)

# precessing_bloch_vector_lab, precessing_bloch_vector_rot = calculate_bloch_vectors(
#     omega_d, omega_0, settings['rabi_frequency'], np.arange(N_time), psi_0
# )
# for idx in range(len(precessing_bloch_vector_lab)):
#     precessing_bloch_vector_lab[idx] /= np.linalg.norm(precessing_bloch_vector_lab[idx])
#     precessing_bloch_vector_rot[idx] /= np.linalg.norm(precessing_bloch_vector_rot[idx])

precessing_bloch_vector_lab, precessing_bloch_vector_rot = calculate_bloch_vectors(
    settings['rabi_frequency'], np.arange(N_time), np.array([1, 0, 0]), B_total_rot[1]/np.linalg.norm(B_total_rot[1]), B_total_lab
)

initial_bloch_vector = precessing_bloch_vector_rot[0]
B_lab_texts, B_rot_texts, transformation_texts = initialize_ax_dict(ax_dict, settings)

B_lab_H_text, B_lab_H_zeeman_text, B_lab_H_drive_text, B_lab_H_zeeman_eq, B_lab_H_drive_eq = B_lab_texts
B_rot_H_text, B_rot_H_zeeman_text, B_rot_H_drive_text, B_rot_H_zeeman_eq, B_rot_H_drive_eq_1, B_rot_H_drive_eq_2 = B_rot_texts
W_trans_equation, W_trans_arrow, omega_is_omega_L = transformation_texts

# Time array
t = np.arange(0, N_time)

# Effective magnetic field in the lab frame
Bx = settings['rabi_frequency'] * np.cos(settings['vector_rotation_speed_rad'] * t)
By = settings['rabi_frequency'] * np.sin(settings['vector_rotation_speed_rad'] * t)
Bz = np.full_like(t, settings['vector_rotation_speed_rad'])   

# Initial Bloch vector (start at north pole of the sphere)
r = np.zeros((len(t), 3))
r[0] = [0, 0, 1]  # [x, y, z]

# Solve Bloch equations using Euler method
for i in range(1, len(t)):
    B = np.array([Bx[i], By[i], Bz[i]])
    drdt = np.cross(B, r[i - 1])
    r[i] = r[i - 1] + drdt
    r[i] /= np.linalg.norm(r[i])

if show_time_list:
    new_ax = fig.add_axes([0.9, 0.9, 0.1, 0.1])
    new_ax.set_axis_off()

abs_pos_text_rot = np.array([0, -1.5, 1.15])
abs_pos_text_lab = np.array([-0.65, 0.55, 1.45])
azim_angle_lab = ax_dict["bloch_lab"].azim
rot_matrix_lab = np.array([
    [np.cos(azim_angle_lab), np.sin(azim_angle_lab), 0],
    [-np.sin(azim_angle_lab), np.cos(azim_angle_lab), 0],
    [0, 0, 1]
])
pos_text_lab = np.dot(rot_matrix_lab, abs_pos_text_lab)

def animate(i):
    if i in time_list and show_time_list:
        index_in_time_list = list(time_list).index(i)
        new_ax.cla()
        new_ax.text(0, 0.5, f"t_{index_in_time_list:.0f}: t = {i}", fontsize=20, ha='center', va='center')
    if i <= t_0:
        new_alpha_sphere = interpolate_between(i, 0, t_0, 0, settings['initial_frame_alpha'])
        new_alpha_font = interpolate_between(i, 0, t_0, 0, settings['intial_font_alpha'])
        new_alpha_vectors = interpolate_between(i, 0, t_0, 0, 1)
        sphere_dict['bloch_lab'].sphere_alpha = new_alpha_sphere
        sphere_dict['bloch_lab'].frame_alpha = new_alpha_sphere
        sphere_dict['bloch_lab'].font_color = (0, 0, 0, new_alpha_font)
        sphere_dict['bloch_lab'].frame_width = new_alpha_vectors
        sphere_dict["bloch_lab"].vector_alpha = [new_alpha_vectors, new_alpha_vectors, new_alpha_vectors, 0]
        sphere_dict['bloch_lab'].make_sphere()
    if t_0 < i <= t_12:
        update_bloch_sphere_vectors(
            i, sphere_dict, ax_dict, B_zeeman_lab, B_drive_lab, B_drive_counter_lab, B_total_lab, B_drive_rot, B_drive_counter_rot, B_zeeman_rot, B_total_rot, azim_angle_rot_sphere, settings
        )
    if t_1 < i <= t_2:
        fade_in_texts(i, t_1, t_2, [B_lab_H_text, B_lab_H_zeeman_text, B_lab_H_drive_text])
    if t_3 < i <= t_4:
        fade_in_texts(i, t_3, t_4, [B_lab_H_zeeman_eq, B_lab_H_drive_eq])
    if t_5 < i <= t_6:
        fade_in_texts(i, t_5, t_6, [W_trans_arrow, W_trans_equation, omega_is_omega_L])
    if t_7 < i <= t_8:
        fade_in_texts(i, t_7, t_8, [B_rot_H_text, B_rot_H_zeeman_text, B_rot_H_drive_text])
    if t_9 < i <= t_10:
        if settings['detuning'] == 0:
            fade_in_texts(i, t_9, t_10, [B_rot_H_zeeman_eq, B_rot_H_drive_eq_1, B_rot_H_drive_eq_2])
        else:
            fade_in_texts(i, t_9, t_10, [B_rot_H_zeeman_eq, B_rot_H_drive_eq_1])
    if t_12 < i <= t_13:
        new_alpha_sphere = max(0, (t_13 - i) / (t_13 - t_12) * settings['initial_frame_alpha'])
        new_alpha_font = max(0, (t_13 - i) / (t_13 - t_12) * settings['intial_font_alpha'])

        sphere_dict['bloch_lab'].sphere_alpha = new_alpha_sphere
        sphere_dict['bloch_lab'].frame_alpha = new_alpha_sphere
        sphere_dict['bloch_lab'].font_color = (0, 0, 0, new_alpha_font)
        sphere_dict['bloch_lab'].frame_width *= (t_13 - i)/(t_13-t_12)
        sphere_dict["bloch_lab"].vector_alpha = [new_alpha_sphere, new_alpha_sphere, new_alpha_sphere, new_alpha_sphere]
        sphere_dict['bloch_lab'].make_sphere()

        if settings['detuning'] == 0:
            all_texts_to_fade = (
                [B_lab_H_text, B_lab_H_zeeman_text, B_lab_H_drive_text] +
                [B_lab_H_zeeman_eq, B_lab_H_drive_eq] +
                [W_trans_arrow, W_trans_equation, omega_is_omega_L] +
                [B_rot_H_text, B_rot_H_zeeman_text, B_rot_H_drive_text] +
                [B_rot_H_zeeman_eq, B_rot_H_drive_eq_1, B_rot_H_drive_eq_2]
            )
        else:
            all_texts_to_fade = (
                [B_lab_H_text, B_lab_H_zeeman_text, B_lab_H_drive_text] +
                [B_lab_H_zeeman_eq, B_lab_H_drive_eq] +
                [W_trans_arrow, W_trans_equation, omega_is_omega_L] +
                [B_rot_H_text, B_rot_H_zeeman_text, B_rot_H_drive_text] +
                [B_rot_H_zeeman_eq, B_rot_H_drive_eq_1]
            )
        fade_out_texts(i, t_12, t_13, all_texts_to_fade)
        if i == t_13:
            global lab_sphere_position
            lab_sphere_position = ax_dict['bloch_lab'].get_position()

    if t_13 < i <= t_14:
        current_position = ax_dict['bloch_rot'].get_position()
        new_height = interpolate_between(i, t_13, t_14, current_position.height, 0.8)
        new_width = interpolate_between(i, t_13, t_14, current_position.width, 0.5)
        new_y_pos = interpolate_between(i, t_13, t_14, current_position.y0, 0.)
        new_x_pos = interpolate_between(i, t_13, t_14, current_position.x0, lab_sphere_position.x0)
        next_position = [new_x_pos, new_y_pos, new_width, new_height]
        ax_dict['bloch_rot'].set_position(next_position)

    if t_14 < i <= t_15:
        if i == t_14 + 1:
            ax_dict['spin_statistics'] = fig.add_axes((0.6, 0.2, 0.4, 0.6))
            ax_dict['spin_statistics'].set_axis_off()
            global pretty_axis_spin_statistics
            global pretty_axis_x_spin
            global pretty_axis_y_spin

            pretty_axis_spin_statistics = PrettyAxis(ax_dict['spin_statistics'], (0, 5, 4.5), (1, 8, 0), (0, 1), (-1.05, 1.05), alpha=1)
            pretty_axis_spin_statistics.add_line("spin_z", 1, 1, c='blue', alpha=1)
            pretty_axis_spin_statistics.add_label(r"$\langle S_{z'} \rangle$", "y", size=20)
            pretty_axis_spin_statistics.add_label(r"$t$", "x", size=20)

            pretty_axis_x_spin = PrettyAxis(ax_dict['spin_statistics'], (0, 2.25, -3.5), (-6, -1, 0), (0, 1), (-1.05, 1.05), alpha=1)
            pretty_axis_x_spin.add_line("spin_x", 0, 1, c='blue', alpha=1)
            pretty_axis_x_spin.add_label(r"$\langle S_{x'} \rangle$", "y", size=20)
            pretty_axis_x_spin.add_label(r"$t$", "x", size=20)

            pretty_axis_y_spin = PrettyAxis(ax_dict['spin_statistics'], (2.75, 5, -3.5), (-6, -1, 2.75), (0, 1), (-1.05, 1.05), alpha=1)
            pretty_axis_y_spin.add_line("spin_y", 0, 1, c='blue', alpha=1)
            pretty_axis_y_spin.add_label(r"$\langle S_{y'} \rangle$", "y", size=20)
            pretty_axis_y_spin.add_label(r"$t$", "x", size=20)

        fade_in_axes(i, t_14, t_15, [pretty_axis_spin_statistics, pretty_axis_x_spin, pretty_axis_y_spin])
        sphere_dict['bloch_rot'].vectors = []
        sphere_dict['bloch_rot'].vector_color = [settings['vector_colors'][0], settings["vector_colors"][1], settings["vector_colors"][2], settings["spin_vector_color"]]
        sphere_dict['bloch_rot'].add_vectors([B_zeeman_rot[i-t_14+1], B_drive_rot[i-t_14+1], B_total_rot[i - t_14 + 1], initial_bloch_vector])
        sphere_dict['bloch_rot'].vector_alpha = [0.4, 0.4, 1, (i-t_14)/(t_15-t_14)]
        sphere_dict['bloch_rot'].make_sphere()

    if t_15 < i <= t_16:
        all_spinx_vals = [
            vector[0] for vector in precessing_bloch_vector_rot[:i - t_15 + 1]
        ]
        all_spiny_vals = [
            vector[1] for vector in precessing_bloch_vector_rot[:i - t_15 + 1]
        ]
        all_spinz_vals = [
            vector[2] for vector in precessing_bloch_vector_rot[:i - t_15 + 1]
        ]
        tail_length = min(settings['tail_length'], len(all_spinz_vals))
        tail_points = precessing_bloch_vector_rot[i-t_15-tail_length+1:i-t_15+2]
        sphere_dict['bloch_rot'].vector_alpha = [0.4, 0.4, 1, 1]
        sphere_dict['bloch_rot'].vector_color = [settings['vector_colors'][0], settings["vector_colors"][1], settings["vector_colors"][2], settings["spin_vector_color"]]
        sphere_dict['bloch_rot'].vectors = []
        sphere_dict['bloch_rot'].add_vectors([B_zeeman_rot[i-t_14+1], B_drive_rot[i-t_14+1], B_total_rot[i - t_14 + 1], precessing_bloch_vector_rot[i-t_15+1]])
        sphere_dict['bloch_rot'].points = []
        sphere_dict['bloch_rot'].add_points([tail_points[:,0], tail_points[:,1], tail_points[:,2]], meth="l") if not DEBUG else None
        sphere_dict['bloch_rot'].make_sphere()
        pretty_axis_spin_statistics.update_line('spin_z', np.arange(len(all_spinz_vals))/(t_17-t_15), np.array(all_spinz_vals))
        pretty_axis_x_spin.update_line('spin_x', np.arange(len(all_spinx_vals))/(t_17-t_15), np.array(all_spinx_vals))
        pretty_axis_y_spin.update_line('spin_y', np.arange(len(all_spiny_vals))/(t_17-t_15), np.array(all_spiny_vals))

    if t_16 < i <= t_17:
        all_spinx_vals = [
            vector[0] for vector in precessing_bloch_vector_rot[:i - t_15 + 1]
        ]
        all_spiny_vals = [
            vector[1] for vector in precessing_bloch_vector_rot[:i - t_15 + 1]
        ]
        all_spinz_vals = [
            vector[2] for vector in precessing_bloch_vector_rot[:i - t_15 + 1]
        ]
        alpha_vectors = interpolate_between(i, t_16, t_17, 0.4, 0)
        tail_points = precessing_bloch_vector_rot[i-t_15-settings['tail_length']+1:i-t_15+2]
        sphere_dict['bloch_rot'].vectors = []
        sphere_dict["bloch_rot"].vector_alpha = [alpha_vectors, alpha_vectors, 1, 1]
        sphere_dict['bloch_rot'].add_vectors([B_zeeman_rot[i-t_14+1], B_drive_rot[i-t_14+1], B_total_rot[i - t_14 + 1], precessing_bloch_vector_rot[i-t_15+1]])
        sphere_dict['bloch_rot'].points = []
        sphere_dict['bloch_rot'].add_points([tail_points[:,0], tail_points[:,1], tail_points[:,2]], meth="l") if not DEBUG else None
        sphere_dict['bloch_rot'].make_sphere()
        pretty_axis_spin_statistics.update_line('spin_z', np.arange(len(all_spinz_vals))/(t_17-t_15), np.array(all_spinz_vals))
        pretty_axis_x_spin.update_line('spin_x', np.arange(len(all_spinx_vals))/(t_17-t_15), np.array(all_spinx_vals))
        pretty_axis_y_spin.update_line('spin_y', np.arange(len(all_spiny_vals))/(t_17-t_15), np.array(all_spiny_vals))
        for axis in [pretty_axis_spin_statistics, pretty_axis_x_spin, pretty_axis_y_spin]:
            axis.alpha = 1 - (i-t_16)/(t_17-t_16)

        if i == t_17:
            ax_dict["bloch_lab"].set_position([
                ax_dict["bloch_rot"].get_position().x0+0.5,
                ax_dict["bloch_rot"].get_position().y0,
                ax_dict["bloch_rot"].get_position().width,
                ax_dict["bloch_rot"].get_position().height
            ])
            fig.delaxes(ax_dict['spin_statistics'])

    if t_17 < i <= t_18:
        tail_points = r[max(0, i-t_17-settings['tail_length']+1):i-t_17+1]
        new_alpha_sphere = interpolate_between(i, t_17, t_18, 0, settings['initial_frame_alpha'])
        new_alpha_font = interpolate_between(i, t_17, t_18, 0, settings['intial_font_alpha'])
        new_alpha_vectors = interpolate_between(i, t_17, t_18, 0, 1)
        sphere_dict['bloch_lab'].sphere_alpha = new_alpha_sphere
        sphere_dict['bloch_lab'].frame_alpha = new_alpha_sphere
        sphere_dict['bloch_lab'].font_color = (0, 0, 0, new_alpha_font)
        sphere_dict["bloch_lab"].vector_color = [settings['vector_colors'][2], "blue"]
        sphere_dict["bloch_lab"].vectors = [B_total_lab[i - t_17 + 1] / np.linalg.norm(B_total_lab[i - t_17 + 1]), r[i - t_17 + 1]]
        sphere_dict["bloch_lab"].vector_alpha = [new_alpha_vectors, new_alpha_vectors]
        sphere_dict['bloch_lab'].frame_width = new_alpha_vectors
        sphere_dict['bloch_lab'].points = []
        sphere_dict['bloch_lab'].add_points([tail_points[:,0], tail_points[:,1], tail_points[:,2]], meth="l", alpha=1) if not DEBUG else None
        sphere_dict['bloch_lab'].make_sphere()

        tail_points = precessing_bloch_vector_rot[i-t_15-settings['tail_length']+1:i-t_15+2]
        sphere_dict['bloch_rot'].vectors = []
        sphere_dict["bloch_rot"].vector_alpha = [1, 1]
        sphere_dict["bloch_rot"].vector_color = [settings["spin_vector_color"], settings['vector_colors'][2]]
        sphere_dict['bloch_rot'].add_vectors([precessing_bloch_vector_rot[i-t_15+1], B_total_rot[i - t_15 + 1]])
        sphere_dict['bloch_rot'].points = []
        sphere_dict['bloch_rot'].add_points([tail_points[:,0], tail_points[:,1], tail_points[:,2]], meth="l") if not DEBUG else None
        sphere_dict['bloch_rot'].make_sphere()

    if i > t_18:
        tail_points = r[max(0, i-t_17-settings['tail_length']+1) : i-t_17+1]
        sphere_dict["bloch_lab"].vectors = [B_total_lab[i - t_17 + 1] / np.linalg.norm(B_total_lab[i - t_17 + 1]), r[i - t_17 + 1]]
        sphere_dict["bloch_lab"].vector_color = [settings['vector_colors'][2], "blue", "purple"]
        sphere_dict['bloch_lab'].points = []
        sphere_dict['bloch_lab'].add_points([tail_points[:,0], tail_points[:,1], tail_points[:,2]], meth="l") if not DEBUG else None #, colors=['blue']*tail_length)
        sphere_dict["bloch_lab"].vector_alpha = [1, 1, 1]
        sphere_dict["bloch_lab"].make_sphere()

        tail_points = precessing_bloch_vector_rot[i-t_15-settings['tail_length']+1:i-t_15+2]
        sphere_dict['bloch_rot'].vectors = []
        sphere_dict['bloch_rot'].add_vectors([precessing_bloch_vector_rot[i-t_15+1], B_total_rot[i - t_15 + 1]])
        sphere_dict['bloch_rot'].points = []
        sphere_dict['bloch_rot'].add_points([tail_points[:,0], tail_points[:,1], tail_points[:,2]], meth="l") if not DEBUG else None
        sphere_dict['bloch_rot'].make_sphere()

    if t_0 < i < t_12:
        text_lab = ax_dict["bloch_lab"].text(pos_text_lab[0], pos_text_lab[1], pos_text_lab[2], "Lab frame", color = "black", size=20, alpha=1, zorder=1000)
        if i > t_8:
            angle_rot = ax_dict["bloch_rot"].azim * np.pi / 180
            rot_matrix_rot = np.array([
                [np.cos(angle_rot), -np.sin(angle_rot), 0],
                [np.sin(angle_rot), np.cos(angle_rot), 0],
                [0, 0, 1]
            ])
            pos_text_rot = np.dot(rot_matrix_rot, abs_pos_text_rot)
            if i <= t_9:
                alpha = interpolate_between(i, t_8, t_9, 0, 1)
            else:
                alpha = 1
            text_rot = ax_dict["bloch_rot"].text(pos_text_rot[0], pos_text_rot[1], pos_text_rot[2], "Rotating frame", color = "black", size=20, alpha=alpha, zorder=1000)

    if t_17 < i:
        desired_pos_rot = np.array([-0.8, -0.8, 1.1])
        desired_pos_lab = np.array([-1, -0.8, 1.05])
        pos_text_lab_now = np.dot(rot_matrix_lab, desired_pos_lab)
        text_lab = ax_dict["bloch_lab"].text(-0.35, -0.35, 1.5, "Lab frame", color = "black", size=30, alpha=1, zorder=1000)

        angle_rot = ax_dict["bloch_rot"].azim * np.pi / 180
        rot_matrix_rot = np.array([
            [np.cos(angle_rot), -np.sin(angle_rot), 0],
            [np.sin(angle_rot), np.cos(angle_rot), 0],
            [0, 0, 1]
        ])
        pos_text_rot = np.dot(rot_matrix_rot, desired_pos_rot)
        text_rot = ax_dict["bloch_rot"].text(pos_text_rot[0], pos_text_rot[1], pos_text_rot[2], "Rotating frame", color = "black", size=30, alpha=1, zorder=1000)

    return list(ax_dict.values())

def init():
    return list(ax_dict.values())

ani = anim.FuncAnimation(
    fig,
    animate,
    tqdm(np.arange(N_time)),
    interval=50,
    init_func=init,
    blit=False,
    repeat=False
)
detuning_name = "zero" if settings["detuning"] == 0 else ("positive" if settings["detuning"] > 0 else "negative")
cache_then_save_funcanimation(ani, f'animations/test/{detuning_name}_detuning_new_final.{file_type}', fps=20)