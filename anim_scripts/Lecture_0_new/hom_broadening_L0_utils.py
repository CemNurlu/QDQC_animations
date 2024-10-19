from anim_base import PrettyAxis, bloch_vector, fit_damped_cosine, math_fontfamily
import numpy as np
from matplotlib.patches import FancyArrowPatch

def create_equation_text(ax_dict, **kwargs):
    avg_eq_string = r"$\langle S^{avg}_x (t) \rangle = \frac{1}{N} \sum_{i=1}^N \langle S_x^{(i)} (t) \rangle $"
    avg_eq_x_y_start = np.array([0.1, -3.])
    avg_eq_x_y_end = np.array([1.4, -1.9])
    avg_eq_start_size, avg_eq_end_size = 50, 20
    avg_eq = ax_dict["plot"].text(*avg_eq_x_y_start, avg_eq_string, size = avg_eq_start_size, alpha = 0, math_fontfamily = math_fontfamily)

    return avg_eq

def create_fourier_visualization(ax_dict, **kwargs):
    pretty_axis_fourier = PrettyAxis(ax_dict["plot"], (0, 2, -4), (-4, -2, 0), data_x_lim=(0, 1), data_y_lim=(0, 1), alpha=0)
    omega_fourier_vec = np.linspace(0, 1, 500)
    F_S_x = np.exp(-0.5 * ((omega_fourier_vec - 0.5) / 0.06) ** 2)
    pretty_axis_fourier.add_line("F_S_x", omega_fourier_vec, F_S_x, "black", alpha=0)
    pretty_axis_fourier.add_label(r'$\mathcal{F} \; [ \langle S_x^{avg} \rangle ] $  ', "y", size=20)
    pretty_axis_fourier.add_label(r'  $\omega$', "x", size=20)

    omega_L_text = ax_dict["plot"].text(0.95, -4.1, r'$\omega_L$', size=18, alpha=0, math_fontfamily=math_fontfamily)

    # T2 and inverse T2 arrows
    T2_arrow = FancyArrowPatch((0.03, 0.23), (1, 0.12), arrowstyle='<->', mutation_scale=20, lw=1.5, color="black", alpha=0)
    T2_text = ax_dict["plot"].text(0.34, 0.34, r'$ \sim T_2$', size=30, alpha=0, math_fontfamily=math_fontfamily)
    T2_inv_arrow = FancyArrowPatch((0.86, -3.13), (1.14, -3.13), arrowstyle='<->', mutation_scale=20, lw=1.5, color="black", alpha=0)
    T2_inv_text = ax_dict["plot"].text(0.83, -3.43, r'$ \sim 1 / \, T_2$', size=30, alpha=0, math_fontfamily=math_fontfamily)

    return pretty_axis_fourier, omega_L_text, T2_arrow, T2_text, T2_inv_arrow, T2_inv_text

def create_spin_evolution_axes(ax_dict, B_time, S_x_max, fitted_spin_avg, B_time_start, B_time_end, **kwargs):
    pretty_axis_spins = PrettyAxis(ax_dict["plot"], (0, 2, -2.8), (-3.8, -1.8, 0),
                                data_x_lim=(B_time_start, B_time_end + 0.01), 
                                data_y_lim=(-S_x_max - 0.1, S_x_max + 0.1), alpha=0)
    pretty_axis_spins.add_label(r'$\langle S_x^{(i)}(t) \rangle $  ', "y", size=20)
    pretty_axis_spins.add_label(r' $t$', "x", size=20)

    pretty_axis_spin_avg = PrettyAxis(ax_dict["plot"], (0, 2, -3), (-4, -2, 0),
                                    data_x_lim=(B_time_start, B_time_end + 0.01),
                                    data_y_lim=(-S_x_max - 0.1, S_x_max + 0.1), alpha=0)
    pretty_axis_spin_avg.add_label(r'$\langle S_x^{avg}(t) \rangle$  ', "y", size=20)
    pretty_axis_spin_avg.add_label(r' $t$', "x", size=15)
    pretty_axis_spin_avg.add_line("S_x_avg", B_time[0], fitted_spin_avg[0, 0], c="black", alpha=0, lw=3.5)

    return pretty_axis_spin_avg, pretty_axis_spins

def create_B_field_axes(ax_dict, B_time, B, B_min, B_max, n_spins, B_time_start, B_time_end, **kwargs):
    pretty_axises_B = [PrettyAxis(ax_dict["plot"], (0, 2, -1.4), (-1.4, -0.2, 0),
                                data_x_lim=(B_time_start, B_time_end + 0.01), data_y_lim=(B_min, B_max), alpha=0)
                    for _ in range(n_spins)]

    # Add labels and setup for each axis
    for s_i, pretty_axis_B in enumerate(pretty_axises_B):
        pretty_axis_B.add_line(f"B_{s_i}", B_time[0], B[s_i, 0], c="red", alpha=0, lw=2.5)
        pretty_axis_B.add_label(fr'$B_{s_i+1}(t) \;$', "y", size=18)
        pretty_axis_B.add_label(r'$\; t$', "x", size=18)

    return pretty_axises_B

def fit_spin_evolutions(ax_dict, B_time, phi, theta, n_spins, **kwargs):
    spin_vectors = np.array([bloch_vector(theta, phi[s_i, :]) for s_i in range(n_spins)])
    actual_spin_avg = np.mean(spin_vectors, axis=0)
    _, (A_fit, omega_fit, phi_fit, tau_fit) = fit_damped_cosine(actual_spin_avg[:, 0], B_time)

    fitted_spin_avg_x = np.sin(theta) * np.cos(omega_fit * B_time + phi_fit) * np.exp(-B_time / tau_fit)
    fitted_spin_avg_y = np.sin(theta) * np.sin(omega_fit * B_time + phi_fit) * np.exp(-B_time / tau_fit)
    fitted_spin_avg_z = np.cos(theta) * np.ones_like(B_time)
    fitted_spin_avg = np.stack((fitted_spin_avg_x, fitted_spin_avg_y, fitted_spin_avg_z), axis=1)
    fitted_exp = np.sin(theta) * np.exp(-B_time / tau_fit)
    return fitted_spin_avg, spin_vectors

def create_random_B_fields(n_spins, n_omegas, B_time_start, B_time_end, B_0, B_fluctuation_multiplier, t_1, t_0, **kwargs):
    w_array = np.random.rand(n_spins, n_omegas) * 2 + 0.5
    B_time = np.linspace(B_time_start, B_time_end, t_1 - t_0)
    delta_t_B = B_time[1] - B_time[0]
    B_offsets = np.linspace(0, 2 * np.pi / B_time_end, n_spins, endpoint=False)
    np.random.shuffle(B_offsets)

    B = np.ones((n_spins, t_1 - t_0)) * B_0 + B_offsets.reshape(-1, 1)
    for s_i in range(n_spins):
        for w_i in range(n_omegas):
            B[s_i, :] += np.cos(w_array[s_i, w_i] * B_time) / n_omegas * B_fluctuation_multiplier

    B_min, B_max = np.min(B) - 0.1, np.max(B) + 0.1
    phi = np.cumsum(B, axis=1) * delta_t_B
    return B_time,B,B_min,B_max,phi


def reset_sphere(sphere_obj):
    """Reset the sphere with black vectors and points."""
    sphere_obj.vectors = []
    sphere_obj.points = []
    sphere_obj.vector_color = ["black"]
    sphere_obj.point_color = ["black"]
    sphere_obj.make_sphere()


def update_alpha(pretty_axes_B, pretty_axes_spins, s_i, local_t_i, frames_per_spin):
    """Update the transparency (alpha) for the plot."""
    new_alpha = local_t_i / (0.1 * frames_per_spin)
    pretty_axes_B[s_i].alpha = new_alpha
    if s_i == 0:
        pretty_axes_spins.alpha = new_alpha


def update_spin_animation(s_i, local_t_i, frames_per_spin, tail, sphere_obj, pretty_axes_B, pretty_axis_spins, **kwargs):
    """Update the spin vectors and plot them on the Bloch sphere."""
    B_time = kwargs.get("B_time", 10)
    vector_colors = kwargs.get("vector_colors", ["black", "blue", "green", "orange", "purple"])
    spin_vectors = kwargs.get("spin_vectors", np.zeros((5, 10, 3)))
    B = kwargs.get("B", np.zeros((5, 10)))
    B_max = kwargs.get("B_max", 1)
    B_min = kwargs.get("B_min", 0)

    B_time_index = int((len(B_time) - 1) * (local_t_i - 0.1 * frames_per_spin) / (0.8 * frames_per_spin))

    sphere_obj.vectors = []
    sphere_obj.points = []
    sphere_obj.vector_color = ["red", vector_colors[s_i + 1]]
    sphere_obj.point_color = [vector_colors[s_i + 1]]

    # Add vectors and points for the current spin
    sphere_obj.add_vectors([[0, 0, B[s_i, B_time_index] / (B_max - B_min)], spin_vectors[s_i, B_time_index]])

    if tail > B_time_index:
        sphere_obj.add_points(spin_vectors[s_i, 0:B_time_index + 1].T, meth="l")
    else:
        sphere_obj.add_points(spin_vectors[s_i, B_time_index - tail + 1:B_time_index + 1].T, meth="l")

    sphere_obj.make_sphere()

    pretty_axes_B[s_i].update_line(f"B_{s_i}", B_time[:B_time_index + 1], B[s_i, :B_time_index + 1])

    if local_t_i - 0.1 * frames_per_spin <= 1:
        pretty_axis_spins.add_line(f"spin_{s_i}_x", B_time[:B_time_index + 1], spin_vectors[s_i, :B_time_index + 1, 0], c=vector_colors[s_i + 1], alpha=1, lw=2.5)
    else:
        pretty_axis_spins.update_line(f"spin_{s_i}_x", B_time[:B_time_index + 1], spin_vectors[s_i, :B_time_index + 1, 0])


def move_axis(s_i, local_t_i, frames_per_spin, pretty_axes_B, **kwargs):
    """Move the axis to the new position."""
    n_spins = kwargs.get("n_spins", 5)

    new_alpha = (local_t_i - 0.9 * frames_per_spin) / (0.1 * frames_per_spin) if local_t_i < frames_per_spin - 1 else 1

    x_0_target = s_i * 2 / n_spins
    x1_target = (s_i + 0.9) * 2 / n_spins
    y_0_target = 0.1
    y_1_target = 0.1 + 1.08 / n_spins

    new_x_pos = (0 + (x_0_target - 0) * new_alpha, 2 + (x1_target - 2) * new_alpha, -1.4 + (y_0_target + 1.4) * new_alpha)
    new_y_pos = (-1.4 + (y_0_target + 1.4) * new_alpha, -0.2 + (y_1_target + 0.2) * new_alpha, new_x_pos[0])

    pretty_axes_B[s_i].update_x_y_pos(new_x_pos, new_y_pos)