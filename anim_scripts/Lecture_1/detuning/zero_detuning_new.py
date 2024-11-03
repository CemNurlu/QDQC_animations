import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from anim_scripts.Lecture_1.detuning.utils import (
    initialize_ax_dict,
    fade_in_texts,
    fade_out_texts,
    fade_in_axes,
    fade_out_axes,
    interpolate_between,
    calculate_labframe_B_fields,
    calculate_rotating_frame_B_fields,
    update_bloch_sphere_vectors,
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
N_time = 800
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
t_10 = 384 # Show rot frame equation
t_11 = 438 # Let it sink in for a bit
t_12 = 500 # Show B_rot bloch sphere
t_13 = 548 # Show time evolution of B_rot
t_14 = 620 # Fade out the left side
t_15 = 650
t_16 = N_time

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
    t_15 //= 10
    t_16 //= 10

time_list = (t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10, t_11, t_12, t_13, t_14, t_15)
bloch_mosaic = [["bloch_lab", "plot_between", "bloch_rot"],
                ["plot", "plot", "plot"]]

settings = {
    "B_x_max": 0.7,
    "B_zeeman_lab_z": 1,
    "phi_0": 0,
    "phi_end_lab": 24*np.pi,
    "phi_end_rot": 500*np.pi,
    "arrow_length": 0.2,
    "time_list": time_list,
    "vector_colors": ["maroon", "hotpink", "red"],
    "intial_sphere_alpha": 0.2,
    "initial_frame_alpha": 0.2,
    "intial_font_alpha": 1,
    "vector_rotation_speed_rad": 24*np.pi/528,
    "rabi_frequency": 5/528 * np.pi,#48*np.pi/528,
    "spin_vector_color": "blue",
    "tail_length": 6,
}

bloch_kwargs = [{
    "vector_color": settings['vector_colors'],
    "vector_alpha" : [1,1,0],
    "vector_width": 6,
    },
    {
    "vector_color": settings['vector_colors'][:],
    "vector_width": 6,
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
B_drive_lab, B_zeeman_lab, B_total_lab = calculate_labframe_B_fields(phi, settings)
B_drive_rot = calculate_rotating_frame_B_fields(phi_rot, settings)
sphere_dict["bloch_lab"].add_vectors([B_zeeman_lab[0], B_drive_lab[0]])
theta_spin = np.arange(0, np.pi*500, settings['rabi_frequency'])
precessing_bloch_vector = bloch_vector(theta_spin, np.pi/2)
initial_bloch_vector = bloch_vector(0, 0)

B_lab_texts, B_rot_texts, transformation_texts = initialize_ax_dict(ax_dict, settings)
B_lab_H_text, B_lab_H_zeeman_text, B_lab_H_drive_text, B_lab_H_zeeman_eq, B_lab_H_drive_eq = B_lab_texts
B_rot_H_text, B_rot_H_zeeman_text, B_rot_H_drive_text, B_rot_H_zeeman_eq, B_rot_H_drive_eq_1, B_rot_H_drive_eq_2 = B_rot_texts
W_trans_equation, W_trans_arrow, omega_is_omega_L = transformation_texts

def animate(i):
    first = True
    if i <= t_0:
        sphere_dict["bloch_lab"].make_sphere()
    if t_0 < i <= t_14:
        update_bloch_sphere_vectors(
            i, sphere_dict, ax_dict, B_zeeman_lab, B_drive_lab, B_total_lab, B_drive_rot, azim_angle_rot_sphere, settings
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
        fade_in_texts(i, t_9, t_10, [B_rot_H_zeeman_eq, B_rot_H_drive_eq_1, B_rot_H_drive_eq_2])
    if t_13 < i <= t_14:
        if first:
            current_position = ax_dict['bloch_rot'].get_position()
            first = False

        new_alpha_sphere = max(0, (t_14 - i) / (t_14 - t_13) * settings['initial_frame_alpha'])
        new_alpha_font = max(0, (t_14 - i) / (t_14 - t_13) * settings['intial_font_alpha'])

        new_height = interpolate_between(i, t_13, t_14, current_position.height, 0.8)
        new_width = interpolate_between(i, t_13, t_14, current_position.width, 0.5)
        new_y_pos = interpolate_between(i, t_13, t_14, current_position.y0, 0.1)

        next_position = [
            current_position.x0 - (current_position.x0 - ax_dict['bloch_lab'].get_position().x0) * (i - t_13) / (t_14 - t_13),
            new_y_pos,
            new_width,
            new_height
        ]
        ax_dict['bloch_rot'].set_position(next_position)
        sphere_dict['bloch_lab'].sphere_alpha = new_alpha_sphere
        sphere_dict['bloch_lab'].frame_alpha = new_alpha_sphere
        sphere_dict['bloch_lab'].font_color = (0, 0, 0, new_alpha_font)
        sphere_dict['bloch_lab'].frame_width *= (t_14 - i)/(t_14-t_13)
        sphere_dict["bloch_lab"].vector_alpha = [new_alpha_sphere, new_alpha_sphere, new_alpha_sphere]
        sphere_dict['bloch_lab'].make_sphere()

        all_texts_to_fade = (
            [B_lab_H_text, B_lab_H_zeeman_text, B_lab_H_drive_text] +
            [B_lab_H_zeeman_eq, B_lab_H_drive_eq] +
            [W_trans_arrow, W_trans_equation, omega_is_omega_L] +
            [B_rot_H_text, B_rot_H_zeeman_text, B_rot_H_drive_text] +
            [B_rot_H_zeeman_eq, B_rot_H_drive_eq_1, B_rot_H_drive_eq_2]
        )
        fade_out_texts(i, t_13, t_14, all_texts_to_fade)
        if i == t_14:
            fig.delaxes(ax_dict['bloch_lab'])
            ax_dict['spin_statistics'] = fig.add_axes((0.6, 0.2, 0.4, 0.6))
            ax_dict['spin_statistics'].set_axis_off()
            global pretty_axis_spin_statistics
            global pretty_axis_x_spin
            global pretty_axis_y_spin
            pretty_axis_spin_statistics = PrettyAxis(ax_dict['spin_statistics'], (0, 5, 4.5), (1, 8, 0), (0, 1), (-1., 1), alpha=1)
            pretty_axis_spin_statistics.add_line("spin_z", 1, 1, c='blue', alpha=1)
            pretty_axis_spin_statistics.add_label(r"$\langle \sigma_z \rangle$", "y", size=20)
            pretty_axis_spin_statistics.add_label(r"$t$", "x", size=20)

            pretty_axis_x_spin = PrettyAxis(ax_dict['spin_statistics'], (0, 2.25, -3.5), (-6, -1, 0), (0, 1), (-1., 1), alpha=1)
            pretty_axis_x_spin.add_line("spin_x", 0, 1, c='blue', alpha=1)
            pretty_axis_x_spin.add_label(r"$\langle \sigma_x \rangle$", "y", size=20)
            pretty_axis_x_spin.add_label(r"$t$", "x", size=20)

            pretty_axis_y_spin = PrettyAxis(ax_dict['spin_statistics'], (2.75, 5, -3.5), (-6, -1, 2.75), (0, 1), (-1., 1), alpha=1)
            pretty_axis_y_spin.add_line("spin_y", 0, 1, c='blue', alpha=1)
            pretty_axis_y_spin.add_label(r"$\langle \sigma_y \rangle$", "y", size=20)
            pretty_axis_y_spin.add_label(r"$t$", "x", size=20)

    if t_14 < i <= t_15:
        fade_in_axes(i, t_14, t_15, [pretty_axis_spin_statistics])
        sphere_dict['bloch_rot'].vectors = []
        sphere_dict['bloch_rot'].add_vectors([initial_bloch_vector, B_drive_rot[i - t_14 + 1]])
        sphere_dict['bloch_rot'].vector_alpha = [(i-t_14)/(t_15-t_14), 1]
        sphere_dict['bloch_rot'].vector_color = [settings['spin_vector_color'], 'red']
        sphere_dict['bloch_rot'].make_sphere()

    if t_15 < i <= t_16:
        all_spinx_vals = [
            vector[0] for vector in precessing_bloch_vector[:i - t_15 + 1]
        ]
        all_spiny_vals = [
            vector[1] for vector in precessing_bloch_vector[:i - t_15 + 1]
        ]
        all_spinz_vals = [
            vector[2] for vector in precessing_bloch_vector[:i - t_15 + 1]
        ]
        tail_length = min(settings['tail_length'], len(all_spinz_vals))
        tail_points = precessing_bloch_vector[i-t_15-tail_length+1:i-t_15+2]
        sphere_dict['bloch_rot'].vectors = []
        sphere_dict['bloch_rot'].add_vectors([precessing_bloch_vector[i-t_15+1], B_drive_rot[i - t_14 + 1]])
        sphere_dict['bloch_rot'].points = []
        sphere_dict['bloch_rot'].add_points([tail_points[:,0], tail_points[:,1], tail_points[:,2]], meth="l")
        sphere_dict['bloch_rot'].make_sphere()
        pretty_axis_spin_statistics.update_line('spin_z', np.arange(len(all_spinz_vals))/(t_16-t_15), np.array(all_spinz_vals))
        pretty_axis_x_spin.update_line('spin_x', np.arange(len(all_spinx_vals))/(t_16-t_15), np.array(all_spinx_vals))
        pretty_axis_y_spin.update_line('spin_y', np.arange(len(all_spiny_vals))/(t_16-t_15), np.array(all_spiny_vals))

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
cache_then_save_funcanimation(ani, f'animations/test/zero_detuning_new.{file_type}', fps=20)