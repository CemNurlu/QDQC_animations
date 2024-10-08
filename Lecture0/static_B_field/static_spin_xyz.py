from anim_base import cache_then_save_funcanimation, bloch_vector, PrettyAxis, prepare_bloch_mosaic, math_fontfamily, file_type
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from tqdm import tqdm


N_time = 300

t_0 = 30 # Show bloch sphere
t_1 = 70 # Show time axises
t_2 = 200 # Show time evolution
t_3 = 220 # Show dm equation
t_4 = N_time # Do nothing

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



plot_S_xyz = {"S_x":True, "S_y":True, "S_z":True}
# plot_S_xyz = {"S_x":True, "S_y":False, "S_z":False}
bloch_kwargs = {
    "vector_color": ["blue"],
    "point_color": ["blue"],
    "point_marker": ["o"]
}

bloch_mosaic = [["bloch_0", "plot_0"]]
fig, ax_dict, sphere_dict = prepare_bloch_mosaic(bloch_mosaic, (12,6), bloch_kwargs)

# fig, ax_dict, sphere_dict = prepare_plots_dict(2, True, (12,6), bloch_kwargs)

n_component_plots = sum([int(val) for val in plot_S_xyz.values()])

B_time = np.linspace(0,1, t_2-t_1)

theta_spin = np.pi/3
phi_spin = np.pi/3
bloch_points_spin = bloch_vector(theta_spin, phi_spin)




sphere_dict["bloch_0"].add_vectors(bloch_points_spin)
sphere_dict["bloch_0"].make_sphere()

# ax_plot = ax_dict["plot_0"]

if n_component_plots > 0:

    pretty_axis_B = PrettyAxis(ax_dict["plot_0"], (0, 1, -2.4*n_component_plots/4 ), (-2.4*n_component_plots/2 + 0.2, -0.2, 0), 
                            data_x_lim=(0, 1), data_y_lim=(-1, 1), alpha=0)
    pretty_axis_B.add_line("B", 0, 0, "red", alpha = 0)
    pretty_axis_B.add_label(r'$B(t)$  ', "y", size = 15)


plot_index = 0
pretty_axises = []
for coord, include_plot in plot_S_xyz.items():
    if include_plot:
        x_pos = (1.4, 2.4, -1.2 - 2.4*plot_index)
        y_pos = ( -2.2 - 2.4*plot_index, - 0.2 - 2.4*plot_index, 1.4)
        data_x_lim = (0, 1)
        data_y_lim = (-1, 1)

        pretty_axis = PrettyAxis(ax_dict["plot_0"], x_pos, y_pos, 
                                data_x_lim=data_x_lim, data_y_lim=data_y_lim, alpha=0)
        
        if coord == "S_x":
            component_value = bloch_points_spin[0]
        elif coord == "S_y":
            component_value = bloch_points_spin[1]
        elif coord == "S_z":
            component_value = bloch_points_spin[2]
        
        pretty_axis.add_line(coord, 0, component_value, "blue", alpha = 0)
        pretty_axis.add_label(fr'$\langle {coord}(t) \rangle$  ', "y", size = 15)
        pretty_axis.add_label(r'  $t$', "x", size = 15)

        pretty_axises.append(pretty_axis)
        plot_index += 1


if n_component_plots > 0:
    ax_dict["plot_0"].set_xlim(-0.2, 2.5)
    ax_dict["plot_0"].set_ylim(0 - 2.4*n_component_plots, 0)
    ax_dict["plot_0"].set_axis_off()
# fig.tight_layout()
dm_eq_string_z = r'$ \langle S_z \rangle = \frac{1}{2} ( \rho_{00} - \rho_{11} )$'
dm_eq_string_x = r'$ \langle S_x \rangle = \frac{1}{2} ( \rho_{01} + \rho_{10} )$'
dm_eq_string_y = r'$ \langle S_y \rangle = \frac{1}{2} ( \rho_{01} - \rho_{10} ) $'
dm_equation_z = ax_dict["plot_0"].text(0.1, -4.4, dm_eq_string_z, fontsize=15, color="black", alpha=0, math_fontfamily = math_fontfamily)
dm_equation_x = ax_dict["plot_0"].text(0.1, -5, dm_eq_string_x, fontsize=15, color="black", alpha=0, math_fontfamily = math_fontfamily)
dm_equation_y = ax_dict["plot_0"].text(0.1, -5.6, dm_eq_string_y, fontsize=15, color="black", alpha=0, math_fontfamily = math_fontfamily)


def animate(i):
    if i <= t_0:
        pass

    elif i <= t_1:
        new_alpha = (i-t_0)/(t_1-t_0)
        if n_component_plots > 0:
            for pretty_axis in pretty_axises:
                pretty_axis.alpha = new_alpha
            pretty_axis_B.alpha = new_alpha

    elif i <= t_2:
        S_index = i - t_1 - 1
        plot_index = 0
        for coord, include_plot in plot_S_xyz.items():
            if include_plot:

                if coord == "S_x":
                    component_values = bloch_points_spin[0] * np.ones(S_index+1)
                elif coord == "S_y":
                    component_values = bloch_points_spin[1] * np.ones(S_index+1)
                elif coord == "S_z":
                    component_values = bloch_points_spin[2] * np.ones(S_index+1)

                pretty_axises[plot_index].update_line(coord, B_time[:S_index+1], component_values)

                plot_index += 1
        pretty_axis_B.update_line("B", B_time[:S_index+1], np.zeros(S_index+1))

    elif i <= t_3:
        new_alpha = (i-t_2)/(t_3-t_2)
        for eq in [dm_equation_z, dm_equation_x, dm_equation_y]:
            eq.set_alpha(new_alpha)


    if n_component_plots > 0:
        return ax_dict["bloch_0"], ax_dict["plot_0"]
    else:
        return ax_dict["bloch_0"]
    
    e
    

def init():
    if n_component_plots > 0:
        return ax_dict["bloch_0"], ax_dict["plot_0"]
    else:
        return ax_dict["bloch_0"]

ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init, blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/static_spin.{file_type}', fps = 20 )