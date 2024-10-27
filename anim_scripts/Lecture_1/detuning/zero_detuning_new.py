import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from anim_scripts.Lecture_1.detuning.utils import (
    initialize_ax_dict,
    fade_in_texts,
    calculate_labframe_B_fields,
    calculate_rotating_frame_B_fields,
    update_bloch_sphere_vectors,
)
from anim_base import (
    cache_then_save_funcanimation,
    prepare_bloch_mosaic,
    file_type
)

##########################
# LONGER DURATIONS FOR ACTUAL ANIMATION
##########################
N_time = 570
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
t_14 = 570 # Fade out the left side
t_15 = N_time

##########################
# SHORTER DURATIONS FOR DEBUGGING
##########################
DEBUG = True
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

time_list = (t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10, t_11, t_12, t_13, t_14, t_15)
bloch_mosaic = [["bloch_lab", "plot_between", "bloch_rot"],
                ["plot", "plot", "plot"]]

settings = {
    "B_x_max": 0.7,
    "B_zeeman_lab_z": 1,
    "phi_0": 0,
    "phi_end": 24*np.pi,
    "arrow_length": 0.2,
    "time_list": time_list,
    "vector_colors": ["maroon", "hotpink", "red"],
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

phi = np.linspace(settings["phi_0"], settings['phi_end'], t_13-t_0)
azim_angle_rot_sphere = (-60 - phi[:] * 180 / np.pi) % 360
B_drive_lab, B_zeeman_lab, B_total_lab = calculate_labframe_B_fields(phi, settings)
B_drive_rot = calculate_rotating_frame_B_fields(phi, settings)
sphere_dict["bloch_lab"].add_vectors([B_zeeman_lab[0], B_drive_lab[0]])

B_lab_texts, B_rot_texts, transformation_texts = initialize_ax_dict(ax_dict, settings)
B_lab_H_text, B_lab_H_zeeman_text, B_lab_H_drive_text, B_lab_H_zeeman_eq, B_lab_H_drive_eq = B_lab_texts
B_rot_H_text, B_rot_H_zeeman_text, B_rot_H_drive_text, B_rot_H_zeeman_eq, B_rot_H_drive_eq_1, B_rot_H_drive_eq_2 = B_rot_texts
W_trans_equation, W_trans_arrow, omega_is_omega_L = transformation_texts

def animate(i):
    if i <= t_0:
        sphere_dict["bloch_lab"].make_sphere()
    if t_0 < i <= t_13:
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
        sphere_dict["bloch_rot"].upda

    return [ax for key, ax in ax_dict.items()]

def init():
    return [ax for key, ax in ax_dict.items()]

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