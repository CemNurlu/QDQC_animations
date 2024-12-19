import numpy as np
from matplotlib.patches import FancyArrowPatch
from anim_base import math_fontfamily
from qutip import *

def linear(x, a, b):
    return a * x + b

def linear_between(x, x_start, x_end, y_start, y_end):
    return  (x - x_start)/(x_end - x_start) * (y_end - y_start) + y_start

def interpolate_between(x, x_start, x_end, y_start, y_end, interpolation_function = linear_between):
    assert x_start <= x <= x_end, f"Value of the variable {x:.2f} must stay between x_start and x_end [{x_start:.1f}, {x_end:.1f}]"
    if isinstance(interpolation_function, str):
        if interpolation_function == "sigmoid":
            interpolation_function = sigmoid_between
        elif interpolate_between == "linear":
            interpolation_function = linear_between
        else:
            raise NotImplementedError(f"Unknown interpolation function {interpolation_function}")
    # if y_end > y_start:
    #     assert y_start <= interpolation_function(x, x_start, x_end, y_start, y_end) <= y_end, f"Value of the interpolation function for x={x} (f(x) = {interpolation_function(x, x_start, x_end, y_start, y_end)})exceeds the inputted range [{y_start}, {y_end}]"
    # elif y_start > y_end:
    #     assert y_end <= interpolation_function(x, x_start, x_end, y_start, y_end) <= y_start, f"Value of the interpolation function for x={x} (f(x) = {interpolation_function(x, x_start, x_end, y_start, y_end)})exceeds the inputted range [{y_end}, {y_start}]"

    return interpolation_function(x, x_start, x_end, y_start, y_end)

def sigmoid_between(x, x_start, x_end, y_start, y_end):
    return y_start + (y_end - y_start) / (1 + np.exp(-0.1*(x - (x_start + x_end)/2)))

def initialize_ax_dict(ax_dict, settings):
    zeeman_color, driving_color, total_color = settings['vector_colors'][0], settings['vector_colors'][1], settings['vector_colors'][2]
    common_kwargs = {"math_fontfamily": math_fontfamily, "size": 20, "alpha": 0}\

    B_lab_H_text = ax_dict["plot"].text(-1.35, 0.2, r'$H \; =$', color="red", **common_kwargs)
    B_lab_H_zeeman_text = ax_dict["plot"].text(-1.13, 0.2, r'$H_{\mathrm{Zeeman}} + $', color=zeeman_color, **common_kwargs)
    B_lab_H_drive_text = ax_dict["plot"].text(-0.7, 0.2, r"$H_{\mathrm{driving}}$", color = driving_color, **common_kwargs)
    B_lab_H_zeeman_eq = ax_dict["plot"].text(-1.55, 0.1, r'$=\omega_L S_z + $', color=zeeman_color, **common_kwargs)
    B_lab_H_drive_eq = ax_dict["plot"].text(-1.10, 0.1, r"$h [ S_x \: \mathrm{cos}\,\omega t + S_y \: \mathrm{sin}\,\omega t ]$", color=driving_color, **common_kwargs)
    B_lab_texts = (B_lab_H_text, B_lab_H_zeeman_text, B_lab_H_drive_text, B_lab_H_zeeman_eq, B_lab_H_drive_eq)

    B_rot_H_text = ax_dict["plot"].text(0.9, 0.2, r"$H \prime \; =$", color = "red", **common_kwargs)
    B_rot_H_zeeman_text = ax_dict["plot"].text(1.15, 0.2, r"$H \prime _{\mathrm{Zeeman}} + $", color = zeeman_color, **common_kwargs)
    B_rot_H_drive_text = ax_dict["plot"].text(1.6, 0.2, r"$H \prime _{\mathrm{driving}}$", color = driving_color, **common_kwargs)
    B_rot_H_zeeman_eq = ax_dict["plot"].text(0.74, 0.1, r"$=( \omega_L - \omega )  S \prime _z + $", color = zeeman_color, **common_kwargs)
    B_rot_H_drive_eq_1 = ax_dict["plot"].text(1.45, 0.1, r"$h (S \prime _x + S \prime _y)$", color = driving_color, **common_kwargs)
    B_rot_H_drive_eq_2 = ax_dict["plot"].text(1.18, 0.0, r"$=h (S \prime _x + S \prime _y) $", color = total_color, **common_kwargs)
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
    if settings['detuning'] == 0:
        omega_is_omega_L = ax_dict["plot_between"].text(0.25, 0.75, r'$\omega = \omega_L$', color = "black", alpha = 0, size = 30, math_fontfamily = math_fontfamily)
    elif settings['detuning'] > 0:
        omega_is_omega_L = ax_dict["plot_between"].text(0.25, 0.75, r'$\omega < \omega_L$', color = "black", alpha = 0, size = 30, math_fontfamily = math_fontfamily)
    elif settings['detuning'] < 0:
        omega_is_omega_L = ax_dict["plot_between"].text(0.25, 0.75, r'$\omega > \omega_L$', color = "black", alpha = 0, size = 30, math_fontfamily = math_fontfamily)
    else:
        raise ValueError(f"Detuning must be a positive or negative number, not {settings['detuning']}")
    
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

def fade_in_axes(i, t_start, t_end, axes_list):
    alpha = (i-t_start)/(t_end-t_start)
    for axis in axes_list:
        axis.alpha = alpha

def fade_out_axes(i, t_start, t_end, axes_list):
    alpha = 1 - (i-t_start)/(t_end-t_start)
    for axis in axes_list:
        axis.alpha = alpha

def fade_out_texts(i, t_start, t_end, text_list):
    alpha = 1 - (i-t_start)/(t_end-t_start)
    for text in text_list:
        text.set_alpha(alpha)

def calculate_labframe_B_fields(phi, settings):
    B_zeeman = np.zeros((len(phi), 3))
    B_drive = np.zeros((len(phi), 3))
    B_drive_counter = np.zeros((len(phi), 3))
    B_total = np.zeros((len(phi), 3))

    B_zeeman[:,2] = settings['B_zeeman_lab_z']

    B_drive[:,0] = settings['B_x_max'] * np.cos(phi)
    B_drive[:,1] = settings['B_x_max'] * np.sin(phi)

    B_drive_counter[:,0] = settings['B_x_max'] * np.cos(-phi)
    B_drive_counter[:,1] = settings['B_x_max'] * np.sin(-phi)

    B_total = B_zeeman + B_drive

    return B_drive, B_drive_counter, B_zeeman, B_total

def calculate_rotating_frame_B_fields(phi, settings):
    B_drive = np.zeros((len(phi), 3))
    B_drive[:,0] = settings['B_x_max'] * np.cos(settings['phi_0'])
    B_drive[:,1] = settings['B_x_max'] * np.sin(settings['phi_0'])

    B_drive_counter_rot = np.zeros((len(phi), 3))
    B_drive_counter_rot[:,0] = settings['B_x_max'] * np.cos(settings['phi_0'] - 5 * phi)
    B_drive_counter_rot[:,1] = settings['B_x_max'] * np.sin(settings['phi_0'] - 5 * phi)

    B_zeeman = np.zeros((len(phi), 3))
    if settings['detuning'] > 0:
        B_zeeman[:,2] = settings['detuning_plot_size']
    elif settings['detuning'] < 0:
        B_zeeman[:,2] = settings['detuning_plot_size']

    B_total = B_zeeman + B_drive

    return B_drive, B_drive_counter_rot, B_zeeman, B_total

def update_bloch_sphere_vectors(
        i, sphere_dict, ax_dict,
        B_zeeman_lab, B_drive_lab, B_drive_counter_lab, B_total_lab,
        B_drive_rot, B_drive_counter_rot, B_zeeman_rot, B_total_rot,
        azim_angle_rot_sphere, settings
    ):
    t_0, t_1, t_2, t_8, t_9, t_10, t_11, t_12 = [settings['time_list'][i] for i in [0, 1, 2, 8, 9, 10, 11, 12]]
    if i <= t_1:
        sphere_dict["bloch_lab"].vectors = []
        sphere_dict["bloch_lab"].add_vectors([B_zeeman_lab[i-t_0-1], B_drive_lab[i-t_0-1], B_drive_counter_lab[i-t_0-1], B_total_lab[i-t_0-1]])
        sphere_dict["bloch_lab"].make_sphere()

    if t_1 < i <= t_2:
        new_alpha = (i-t_1)/(t_2-t_1)
        new_alpha_2 = min(0.4, (i-t_1)/(20))
        sphere_dict["bloch_lab"].vector_color = [settings['vector_colors'][0], settings['vector_colors'][1], settings['counter_rotating_vector_color'], settings['vector_colors'][2]]
        sphere_dict["bloch_lab"].vector_alpha = [1,1,new_alpha_2,new_alpha]
        sphere_dict["bloch_lab"].vectors = []
        sphere_dict["bloch_lab"].add_vectors([B_zeeman_lab[i-t_0-1], B_drive_lab[i-t_0-1], B_drive_counter_lab[i-t_0-1], B_total_lab[i-t_0-1]])
        sphere_dict["bloch_lab"].make_sphere()


    if t_2 < i < t_12:
        B_time_index = i - t_0 - 1
        sphere_dict["bloch_lab"].vectors = []
        sphere_dict["bloch_lab"].add_vectors([B_zeeman_lab[i-t_0-1], B_drive_lab[i-t_0-1], B_drive_counter_lab[i-t_0-1], B_total_lab[i-t_0-1]])
        sphere_dict["bloch_lab"].make_sphere()



    if t_8 < i <= t_9:
        B_time_index = i - t_0 - 1
        new_alpha_sphere = interpolate_between(i, t_8, t_9, 0, settings['initial_frame_alpha'])
        new_alpha_font = interpolate_between(i, t_8, t_9, 0, settings['intial_font_alpha'])
        new_alpha_vectors = interpolate_between(i, t_8, t_9, 0, 1)
        sphere_dict['bloch_rot'].sphere_alpha = new_alpha_sphere
        sphere_dict['bloch_rot'].frame_alpha = new_alpha_sphere
        sphere_dict['bloch_rot'].font_color = (0, 0, 0, new_alpha_font)
        sphere_dict['bloch_rot'].frame_width = new_alpha_vectors
        sphere_dict['bloch_rot'].vector_color = [settings['vector_colors'][0], settings['vector_colors'][1], settings['vector_colors'][2], settings['counter_rotating_vector_color']]
        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors([B_zeeman_lab[i], B_drive_rot[B_time_index], B_total_lab[0], B_drive_counter_rot[B_time_index]])
        sphere_dict["bloch_rot"].vector_alpha = [new_alpha_vectors, new_alpha_vectors, new_alpha_vectors, new_alpha_vectors]
        ax_dict["bloch_rot"].azim = azim_angle_rot_sphere[i]
        sphere_dict["bloch_rot"].make_sphere()
    if t_9 < i <= t_10:
        B_time_index = i - t_0 - 1
        alpha = interpolate_between(i, t_9, t_10, 1, 0.3)
        # alpha_counter = interpolate_between(i, t_9, t_10, 1, 0)
        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors([B_zeeman_lab[i], B_drive_rot[B_time_index], B_total_lab[0], B_drive_counter_rot[B_time_index]])
        sphere_dict["bloch_rot"].vector_alpha = [alpha, alpha, alpha, alpha]
        ax_dict["bloch_rot"].azim = azim_angle_rot_sphere[i]
        sphere_dict["bloch_rot"].make_sphere()

    if t_10 < i <= t_11:
        B_time_index = i - t_0 - 1
        alpha = interpolate_between(i, t_10, t_11, 0, 1)
        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors([B_zeeman_rot[i], B_drive_rot[B_time_index], B_total_rot[i], B_drive_counter_rot[B_time_index]])
        sphere_dict["bloch_rot"].vector_alpha = [alpha, alpha, alpha, 0.3]
        ax_dict["bloch_rot"].azim = azim_angle_rot_sphere[i]
        sphere_dict["bloch_rot"].make_sphere()

    if i > t_11:
        B_time_index = i - t_0 - 1
        alpha_vectors = interpolate_between(i, t_11, t_12, 1, 0.4) #, interpolation_function = "linear")
        alpha_counter = interpolate_between(i, t_11, t_12, 1, 0)
        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors([B_zeeman_rot[i], B_drive_rot[B_time_index], B_total_rot[i], B_drive_counter_rot[B_time_index]])
        # sphere_dict["bloch_rot"].vector_color = [settings['vector_colors'][2]]
        sphere_dict["bloch_rot"].vector_alpha = [alpha_vectors, alpha_vectors, 1, alpha_counter]

        if 290 <= ax_dict["bloch_rot"].azim <= 310:
            if t_12 - i <= 2*np.pi/settings['vector_rotation_speed_rad']:
                ax_dict["bloch_rot"].azim = 300
                # print("A")

            else:
                ax_dict["bloch_rot"].azim = azim_angle_rot_sphere[i]
        else:
            ax_dict["bloch_rot"].azim = azim_angle_rot_sphere[i]

        sphere_dict["bloch_rot"].make_sphere()


# Function to define the Hamiltonian in the lab frame
def hamiltonian_lab(omega_0, omega_d, Omega):
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    H_static = 0.5 * omega_0 * sz  # Static field in z-direction
    H_drive_x = [0.5 * Omega * sx, lambda t, args: np.cos(omega_d * t - 3*np.pi/4)]  # Driving in x
    H_drive_y = [0.5 * Omega * sy, lambda t, args: np.sin(omega_d * t - 3*np.pi/4)]  # Driving in y
    return [H_static, H_drive_x, H_drive_y]

# Simulate the dynamics
def simulate_dynamics(omega_d, omega_0, Omega, t, psi0):
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    H = hamiltonian_lab(omega_0, omega_d, Omega)
    result = mesolve(H, psi0, t, [], [sx, sy, sz])
    bloch_vecs = np.array(result.expect)  # Extract <X>, <Y>, <Z>
    return bloch_vecs

# Rotating frame transformation
def transform_to_rotating_frame(bloch_vecs, omega_rot, t):
    x, y, z = bloch_vecs
    x_rot = x * np.cos(omega_rot * t) + y * np.sin(omega_rot * t)
    y_rot = -x * np.sin(omega_rot * t) + y * np.cos(omega_rot * t)
    z_rot = z  # z-component is invariant in rotation about z-axis
    return np.array([x_rot, y_rot, z_rot])

def calculate_bloch_vectors(omega_d, omega_0, Omega, t, psi_0):
    bloch_vecs = simulate_dynamics(omega_d, omega_0, Omega, t, psi_0)
    bloch_vecs_rot = transform_to_rotating_frame(bloch_vecs, omega_0, t)

    return bloch_vecs.T, bloch_vecs_rot.T

def rotate_vector(vector, axis, angle):
    """
    Rotate a vector by a given angle around a specified axis using Rodrigues' rotation formula.

    Parameters:
    vector (np.ndarray): The vector to be rotated.
    axis (np.ndarray): The axis around which to rotate the vector.
    angle (float): The angle by which to rotate the vector (in radians).

    Returns:
    np.ndarray: The rotated vector.
    """
    axis = axis / np.linalg.norm(axis)  # Normalize the rotation axis
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotated_vector = (vector * cos_angle +
                      np.cross(axis, vector) * sin_angle +
                      axis * np.dot(axis, vector) * (1 - cos_angle))
    return rotated_vector

def calculate_bloch_vectors(rabi_freq, times, initial_state, driving_field_rot, driving_fields_lab):
    angles = np.linspace(0, rabi_freq * times[-1], len(times))
    bloch_vecs_rot = np.array([rotate_vector(initial_state, driving_field_rot, angle) for angle in angles])
    bloch_vecs_lab = np.array([
        rotate_vector(initial_state, driving_field/np.linalg.norm(driving_field), angle)
        for angle, driving_field in zip(angles, driving_fields_lab)
    ])

    return bloch_vecs_lab, bloch_vecs_rot
