from anim_base import (cache_then_save_funcanimation, bloch_vector, PrettyAxis, file_type,
                       prepare_bloch_mosaic, math_fontfamily, fit_damped_cosine)
from anim_scripts.Lecture_0_new.hom_broadening_L0_utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from tqdm import tqdm
import random
from matplotlib.patches import FancyArrowPatch

def init():
    sphere_dict["bloch"].make_sphere()
    return [ax for key, ax in ax_dict.items()]


def animate_spins(i):
    """Scene 1 - Show the spin evolution with varying magnetic fields"""
    frames_per_spin = (t_1 - t_0) // n_spins
    spin_idx = (i - t_0 - 1) // frames_per_spin
    time_idx_this_spin = (i - t_0 - 1) % frames_per_spin
    tail = frames_per_spin // 10
    sphere_obj = sphere_dict["bloch"]

    if i == t_1 - 1:
        reset_sphere(sphere_obj)

    # For each spin, let the vectors fade in, then move according to B(t), then fade out
    if time_idx_this_spin <= 0.1 * frames_per_spin:
        update_alpha(pretty_axes_B, pretty_axes_spins, spin_idx, time_idx_this_spin, frames_per_spin)
    elif time_idx_this_spin <= 0.9 * frames_per_spin:
        update_spin_animation(
            spin_idx, time_idx_this_spin, frames_per_spin, tail, sphere_obj, pretty_axes_B, pretty_axes_spins, **settings
            )
    else:
        move_axis(
            spin_idx, time_idx_this_spin, frames_per_spin, pretty_axes_B, **settings
            )

def remove_B_plot(i):
    """Scene 2 - Remove B(t) plot"""
    y_lim_max = 1.08/n_spins + 0.2
    linear_interpolation = (t_2 - i) / (t_2 - t_1)
    for idx in range(n_spins):
        pretty_axes_B[idx].alpha = linear_interpolation

    new_y_pos = (
        y_lim_max - 2.1  + (-3.8 - (y_lim_max - 2.1) ) * linear_interpolation,
        y_lim_max - 0.1  + (-1.8 - (y_lim_max - 0.1) ) * linear_interpolation,
        0
    )
    new_x_pos = (0, 2, (new_y_pos[0] + new_y_pos[1])/2)

    pretty_axes_spins.update_x_y_pos(new_x_pos, new_y_pos)

def show_avg_eq(i):
    """Scene 3 - Show the average equation"""
    new_alpha = (i - t_2) / (t_3 - t_2)
    eq_text.set_alpha(new_alpha)

def show_avg_eq_and_axis(i):
    # Logic for displaying average equation and axis
    pass

def show_spin_avg_plot(i):
    # Logic for displaying spin average plot
    pass

def remove_S_x_and_show_fourier(i):
    # Logic for removing S_x and showing Fourier plot
    pass

def show_fourier_transform(i):
    # Logic for Fourier transform visualization
    pass

# Constants and parameters
DEBUG = False
N_time, t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9 = (850, 10, 510, 540, 560, 610, 640, 740, 770, 790, 850)
n_spins, n_omegas = 5, 5
B_time_start, B_time_end, B_0, B_fluctuation_multiplier = 0, 25, 1.5, 2
SEED, theta, gridspec_kw = 12378, np.pi/3, {"height_ratios": [1], "width_ratios": [1, 1.8]}

# Debug mode
if DEBUG:
    N_time = N_time // 5
    t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8 = [x // 5 for x in (t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8)]

# Initialize
vector_colors = ["red", "blue", "green", "purple", "orange", "cyan", "pink", "yellow"]
bloch_kwargs = {"vector_color": vector_colors, "vector_width": 3}
fig, ax_dict, sphere_dict = prepare_bloch_mosaic([["bloch", "plot"]], (14, 8), bloch_kwargs, gridspec_kw=gridspec_kw)
ax_dict["plot"].set_axis_off()
np.random.seed(SEED)

# Create a dictionary containing all settings of the animation
settings = {
    "n_spins": n_spins,
    "n_omegas": n_omegas,
    "B_time_start": B_time_start,
    "B_time_end": B_time_end,
    "B_0": B_0,
    "B_fluctuation_multiplier": B_fluctuation_multiplier,
    "t_1": t_1,
    "t_0": t_0,
    "theta": theta,
}

# Initialization steps for setting up the components of the animation
B_time, B, B_min, B_max, phi = create_random_B_fields(**settings)
fitted_spin_avg, spin_vectors = fit_spin_evolutions(ax_dict, B_time, phi, **settings)
pretty_axes_B = create_B_field_axes(ax_dict, B_time, B, B_min, B_max, **settings)
pretty_axis_spin_avg, pretty_axes_spins = create_spin_evolution_axes(ax_dict, B_time, np.sin(theta), fitted_spin_avg, **settings)
pretty_axis_fourier, omega_L_text, T2_arrow, T2_text, T2_inv_arrow, T2_inv_text = create_fourier_visualization(ax_dict)
eq_text = create_equation_text(ax_dict)

settings.update({
    "B_time": B_time,
    "B": B,
    "B_min": B_min,
    "B_max": B_max,
    "vector_colors": vector_colors,
    "spin_vectors": spin_vectors,
    "fitted_spin_avg": fitted_spin_avg,
    "S_x_max": np.sin(theta),
})

# Define all scenes and their corresponding logic and timing
animation_phases = {
    (0, t_0): lambda i: None,
    (t_0, t_1): animate_spins,
    (t_1, t_2): remove_B_plot,
    (t_2, t_3): show_avg_eq,
    (t_3, t_4): lambda i: None,
    (t_4, t_5): show_avg_eq_and_axis,
    (t_5, t_6): show_spin_avg_plot,
    (t_6, t_7): remove_S_x_and_show_fourier,
    (t_7, t_8): show_fourier_transform,
    (t_8, t_9): lambda i: None,
}

def animate(i):
    for (start, end), func in animation_phases.items():
        if start <= i < end:
            func(i)
            return [ax for key, ax in ax_dict.items()]

# Create and save the animation
anim_func = anim.FuncAnimation(fig, animate, frames=tqdm(range(N_time)), interval=50, init_func=init, blit=True)
cache_then_save_funcanimation(anim_func, f'animations/test/hom_broadening_L0_timo.{file_type}', fps=20)