from anim_base import cache_then_save_funcanimation, bloch_vector, PrettyAxis, prepare_bloch_mosaic, math_fontfamily, file_type
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm

##########################
# LONGER DURATIONS FOR ACTUAL ANIMATION
##########################

N_time = 750
t_0 = 10 # Show bloch spheres
t_1 = 50 # Show S_x_super axis
t_2 = 250 # Show time evolution of many spins
t_3 = 280 # Do nothing
t_4 = 310 # Show equaiton for S_x_avg 
t_5 = 330 # show N=7 annotation
t_6 = 350 # Do nothing
t_7 = 380 # Reset super lines, Shrink and move equation for S_x_avg, remove annotation and show S_x_avg axis
t_8 = 580 # Show time evolution of S_x_super and S_x_avg
t_9 = 620 # Do nothing
t_10 = 650 # Move S_x_avg plot up and remove S_x_super
t_11 = 680 # Show fourier transform and T2 and 1/T2 arrows
t_12 = N_time

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
   

# Create mosaic plot
bloch_mosaic = [["bloch_super", "plot"],
                ["bloch_avg", "plot"]]
spin_colors = ["red", "blue", "green", "purple", "orange", "pink", "yellow"]
bloch_kwargs = [{
    "vector_color": spin_colors,
    "vector_width": 6},
    {
    "vector_color": ["black"],
    "vector_width": 6}
]

gridspec_kw = {"height_ratios":[1,1], "width_ratios":[1,2.6]}
fig, ax_dict, sphere_dict = prepare_bloch_mosaic(bloch_mosaic, (15.6,8), bloch_kwargs, gridspec_kw=gridspec_kw)
ax_dict["plot"].set_axis_off()


# Some random angular frequencies
w_vec = np.array(
    [11.0859516652828848, 10.48751575588186447,
    10.8744352926749055,  10.6277185865151992,
    11.364233965544978,   11.708209673453967,
    12.415826542789627]
)

# Polar angle of spins and S_z
theta = np.pi/5
S_z_value = np.cos(theta)

# Time vector
B_time = np.linspace(0, 5, t_2-t_1)

# Phases are simply time * angular frequency
phi_super = w_vec.reshape(-1,1)*B_time.reshape(1,-1)

# Allocate memory for all different spin vectors
bloch_super = np.zeros((len(w_vec), t_2-t_1, 3))

# The corresponding bloch vectors for each spin are computed
for w_i in range(len(w_vec)):
    bloch_super[w_i,:,:] = bloch_vector(theta, phi_super[w_i])

# The average of all spins
bloch_avg = np.mean(bloch_super, axis=0)

# Initialize a PrettyAxis for the S_x component of multiple spins
pretty_axis_super_x = PrettyAxis(ax_dict["plot"], (0,1, -1.2), (-2.2, -0.2, 0),
                        data_x_lim=(0,5),
                        data_y_lim=(-1, 1),
                        alpha = 0)

# For every spin, add a line to the pretty axis
for w_i in range(len(w_vec)):
    pretty_axis_super_x.add_line(f"spin_{w_i}_x", B_time[0], bloch_super[w_i,0,0], c = spin_colors[w_i], alpha=0) 
# Add labels to the multi-spin plot
pretty_axis_super_x.add_label(r'$\langle S_x^{(i)}(t) \rangle $  ', "y", size = 20)
pretty_axis_super_x.add_label(r'  $t$', "x", size = 20)

# Create a PrettyAxis for the z-value of all spins
pretty_axis_super_z = PrettyAxis(ax_dict["plot"], (1.2, 1.8, -1.2), (-2.2, -0.2, 1.2),
                        data_x_lim=(0,5),
                        data_y_lim=(-1, 1),
                        alpha = 0)

pretty_axis_super_z.add_line("spin_z", B_time[0], S_z_value, c = "black", alpha=0, lw = 3.5)
pretty_axis_super_z.add_label(r'$\langle S_z^{(i)}(t) \rangle $', "y", size = 20)
pretty_axis_super_z.add_label(r'  $t$', "x", size = 20)

# Create a pretty axis for the S_x value of the average over all spins
pretty_axis_avg_x = PrettyAxis(ax_dict["plot"], (0,1, -3.6), (-4.6, -2.6, 0),
                        data_x_lim=(0,5),
                        data_y_lim=(-1, 1),
                        alpha = 0)
pretty_axis_avg_x.add_line("spin_avg_x", B_time[0], bloch_avg[0,0], c = "black", alpha=0, lw = 3.5)
pretty_axis_avg_x.add_label(r"$\langle S^{avg}_x (t) \rangle $", "y", size = 20)
pretty_axis_avg_x.add_label(r'  $t$', "x", size = 20)

# Create a pretty axis for the S_z value of the average over all spins
pretty_axis_avg_z = PrettyAxis(ax_dict["plot"], (1.2, 1.8, -3.6), (-4.6, -2.6, 1.2),
                        data_x_lim=(0,5),
                        data_y_lim=(-1, 1),
                        alpha = 0)
pretty_axis_avg_z.add_line("spin_avg_z", B_time[0], S_z_value, c = "black", alpha=0, lw = 3.5)
pretty_axis_avg_z.add_label(r'$\langle S_z^{avg}(t) \rangle $ ' , "y", size = 20)
pretty_axis_avg_z.add_label(r'  $t$', "x", size = 20)

# Set ax limits
ax_dict["plot"].set_xlim(-0.2, 2.0)
ax_dict["plot"].set_ylim(-4.8, 0)


# Create a PrettyAxis object for the fourier transform
pretty_axis_fourier = PrettyAxis(ax_dict["plot"], (0,1.8, -4.6), (-4.6, -2.6, 0),
                           data_x_lim=(0, 1), data_y_lim=(0, 1), alpha=0)
pretty_axis_fourier.add_label(r'$\mathcal{F} \; [ \langle S^{avg}_x (t) \rangle ] $  ', "y", size = 20)
pretty_axis_fourier.add_label(r'  $\omega$', "x", size = 15)

# Create a gaussian to add to the fourier transform plot
omega_fourier_vec = np.linspace(0, 1, 500)
mu = 0.5
sigma = 0.6
F_S_x = np.exp(-0.5*((omega_fourier_vec - mu)/sigma)**2)
pretty_axis_fourier.add_line("F_S_x", 
                    omega_fourier_vec, F_S_x,
                    "black", lw=3.5, alpha = 0)

# Add the omega Larmor label
omega_L_text = ax_dict["plot"].text(0.85, -4.8, r'$\omega_L$', size = 20, alpha = 0, math_fontfamily = math_fontfamily)

# Some parameters for the average spin equation
avg_eq_string = r"$\langle S^{avg}_x (t) \rangle = \frac{1}{N} \sum_{i=1}^N \langle S_x^{(i)} (t) \rangle $"
avg_eq_x_y_start = np.array([0.3, -3.2])
avg_eq_x_y_end = np.array([0.3, -2.8])
avg_eq_start_size, avg_eq_end_size = 50, 25

# Add the average spin equation to the plot
avg_eq = ax_dict["plot"].text(*avg_eq_x_y_start, avg_eq_string, size = avg_eq_start_size, alpha = 0, math_fontfamily = math_fontfamily)

# Number of spins text annotation
N_annotation_xy = avg_eq_x_y_start + np.array([0.4, -1])
N_annotation = ax_dict["plot"].text(*N_annotation_xy, fr"$(here \;\; N = {len(w_vec)})$", size = 30, alpha = 0, math_fontfamily = math_fontfamily)

# Arrow indicating the T2 relaxation time
T2_arrow = FancyArrowPatch((0.0, -0.15), (1, -0.15), 
        arrowstyle='<->', mutation_scale=20,
        lw = 1.5, color = "black",
        alpha = 0
    )
# Text explaining the T2 arrow
T2_text = ax_dict["plot"].text(0.4, -0.05, r'$ \sim T_2^*$', size = 30, alpha = 0, math_fontfamily = math_fontfamily)

# Arrow indicating the inverse of the T2 time in the fourier spectra
T2_inv_arrow = FancyArrowPatch((0.78, -3.6), (1.01, -3.6), 
        arrowstyle='<->', mutation_scale=20,
        lw = 1.5, color = "black",
        alpha = 0
    )
# Text explaining the inverted T2 arrow
T2_inv_text = ax_dict["plot"].text(0.76, -3.92, r'$ \sim 1 / \, T_2^*$', size = 30, alpha = 0, math_fontfamily = math_fontfamily)

for arrow in [T2_arrow, T2_inv_arrow]:
    ax_dict["plot"].add_patch(arrow)
    arrow.set_zorder(10)

# Add title to plot
ax_dict["bloch_super"].set_title(r"$Individual \; Spins$", size = 22, math_fontfamily = math_fontfamily)

def animate(i):
    
    if i <= t_0:
        pass

    if t_0 < i <= t_1:
        # Fade in the PrettyAxis objects by changing their alpha value
        new_alpha = (i-t_0)/(t_1-t_0)
        pretty_axis_super_x.alpha = new_alpha
        pretty_axis_super_z.alpha = new_alpha
        
    
    if t_1 < i <= t_2:
        # This index tells us in which time step of the spin vector we are
        S_index = i - t_1 - 1

        # Update all spin lines in the PrettyAxis
        for w_i in range(len(w_vec)):
            pretty_axis_super_x.update_line(f"spin_{w_i}_x", B_time[:S_index+1], bloch_super[w_i,:S_index+1,0])
        pretty_axis_super_z.update_line("spin_z", B_time[:S_index+1], S_z_value*np.ones(S_index+1))

        # Reset vectors in bloch sphere
        sphere_dict["bloch_super"].vectors = []
        # Add new vectors from the current timestep
        sphere_dict["bloch_super"].add_vectors(bloch_super[:,S_index,:])
        # Render the sphere
        sphere_dict["bloch_super"].make_sphere()
        # Rendering the sphere resets the title, so we need to impose it again
        ax_dict["bloch_super"].set_title(r"$Individual \; Spins$", size = 22, math_fontfamily = math_fontfamily)

    if t_2 < i <= t_3:
        pass

    if t_3 < i <= t_4:
        # Fade in equation for the average spin by changing the alpha
        new_alpha = (i - t_3)/(t_4-t_3)
        avg_eq.set_alpha(new_alpha)
    
    if t_4 < i <= t_5:
        # Fade in the number-of-spins annotation by changing its alpha
        new_alpha = (i - t_4)/(t_5-t_4)
        N_annotation.set_alpha(new_alpha)
    
    if t_5 < i <= t_6:
        pass

    if t_6 < i <= t_7:
        if i == t_6 + 1:
            # If we are in the first frame of this iteration, reset the lines
            for w_i in range(len(w_vec)):
                pretty_axis_super_x.update_line(f"spin_{w_i}_x", B_time[:1], bloch_super[w_i,:1,0])
            pretty_axis_super_z.update_line("spin_z", B_time[:1], S_z_value*np.ones(1))

            # Reset the vectors on the bloch sphere
            sphere_dict["bloch_avg"].vectors = []
            sphere_dict["bloch_avg"].add_vectors(bloch_avg[0])
            sphere_dict["bloch_avg"].make_sphere()
            ax_dict["bloch_avg"].set_title(r"$Average \; of \; Spins$", size = 22, math_fontfamily = math_fontfamily)

        # Fade in the pretty axis for the average of the spins by changing their alphas
        new_alpha = (i - t_6)/(t_7-t_6)
        pretty_axis_avg_x.alpha = new_alpha
        pretty_axis_avg_z.alpha = new_alpha

        # Move and change size of the spin average equation based on alpha
        new_avg_eq_x, new_avg_eq_y = avg_eq_x_y_start + (avg_eq_x_y_end - avg_eq_x_y_start) * new_alpha
        new_size = avg_eq_start_size + (avg_eq_end_size - avg_eq_start_size) * new_alpha
        avg_eq.set(x = new_avg_eq_x, y = new_avg_eq_y, fontsize = new_size)
        # fade out the number-of-spins annotation
        N_annotation.set_alpha(1-new_alpha)

    if t_7 < i <= t_8:
        # This if-clause repeats the same process os the 'if t_1 < i <= t_2' but also
        # updating the average spin plot lines and vectors

        S_index = i - t_7 - 1

        for w_i in range(len(w_vec)):
            pretty_axis_super_x.update_line(f"spin_{w_i}_x", B_time[:S_index+1], bloch_super[w_i,:S_index+1,0])

        pretty_axis_super_z.update_line("spin_z", B_time[:S_index+1], S_z_value*np.ones(S_index+1))

        pretty_axis_avg_x.update_line("spin_avg_x", B_time[:S_index+1], bloch_avg[:S_index+1,0])
        pretty_axis_avg_z.update_line("spin_avg_z", B_time[:S_index+1], S_z_value*np.ones(S_index+1))

        sphere_dict["bloch_super"].vectors = []
        # print(bloch_super.shape)
        sphere_dict["bloch_super"].add_vectors(bloch_super[:,S_index,:])
        sphere_dict["bloch_super"].make_sphere()

        sphere_dict["bloch_avg"].vectors = []
        sphere_dict["bloch_avg"].add_vectors(bloch_avg[S_index])
        sphere_dict["bloch_avg"].make_sphere()

        ax_dict["bloch_super"].set_title(r"$Individual \; Spins$", size = 22, math_fontfamily = math_fontfamily)
        ax_dict["bloch_avg"].set_title(r"$Average \; of \; Spins$", size = 22, math_fontfamily = math_fontfamily)

    if t_8 < i <= t_9:
        pass

    if t_9 < i <= t_10:
        new_alpha = (t_10 - i)/(t_10-t_9)
        # Fade out the multi-spin plots, the z component plot of
        # the average spin and the equation for the average spin
        pretty_axis_super_x.alpha = new_alpha
        pretty_axis_super_z.alpha = new_alpha
        pretty_axis_avg_z.alpha = new_alpha
        avg_eq.set_alpha(new_alpha)

        # Move the PrettyAxis of the average spin 
        new_x_pos = (0, 1.8 - 0.8*new_alpha, -1.2 - 2.4 *new_alpha)
        new_y_pos = (-2.2 - 2.4 *new_alpha, -0.2 - 2.4 *new_alpha, 0)
        pretty_axis_avg_x.update_x_y_pos(new_x_pos, new_y_pos)
    
    if t_10 < i <= t_11:
        # Fade in the fourier transform PrettyAxis, the T2 arrows and the associated
        # text annotations
        new_alpha = (i-t_10)/(t_11-t_10)
        pretty_axis_fourier.alpha = new_alpha
        for plot_obj in (omega_L_text, T2_arrow, T2_text, T2_inv_arrow, T2_inv_text):
            plot_obj.set_alpha(new_alpha)


    return [ax for key, ax in ax_dict.items()]

def init():
    sphere_dict["bloch_super"].add_vectors(bloch_avg[0])
    # sphere_dict["bloch_avg"].add_vectors(bloch_avg[0])

    sphere_dict["bloch_super"].make_sphere()
    ax_dict["bloch_super"].set_title(r"$Individual \; Spins$", size = 22, math_fontfamily = math_fontfamily)

    # sphere_dict["bloch_avg"].make_sphere()
    return [ax for key, ax in ax_dict.items()]
    


ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init, 
                              blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/inhom_broadening.{file_type}', fps = 20 )