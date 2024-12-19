from anim_base import  cache_then_save_funcanimation, Spin2D, PrettySlider, random_walk, math_fontfamily, cache_then_save_funcanimation, file_type
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tqdm import tqdm
import numpy as np

N_time = 480

t_0 = 20 # Show the balls, spin labels and S1 frame text
t_1 = 60  # Do nothing
t_2 = 80 # Fade in axises
t_3 = 120 # Do nothing
t_4 = 140 # Fade in spin2 field lines
t_5 = 180 # Do nothing
t_6 = 200 # Show magnetic field sliders and text
t_7 = 260 # Do nothing
t_8 = 460 # Move the spin2 ball and update the sliders

DEBUG = False

if DEBUG:
    N_time //= 5
    t_0 //= 5
    t_1 //= 5
    t_2 //= 5
    t_3 //= 5
    t_4 //= 5
    t_5 //= 5
    t_6 //= 5
    t_7 //= 5
    t_8 //= 5

# These constants are used for the random walk of the balls and the spin axis
RANDOM_FORCE_VAR_POS = 0.01
SPRING_CONST_POS = 0.3
DAMPING_CONST_POS = 0.2
# The maximum alpha value of the particle balls
# Has to be between 0 and 1
BALL_MAX_ALPHA = 0.7 


fig, ax = plt.subplots(figsize=(15,10))
ax.set_axis_off()
ax.set(xlim=(-1.5, 1.5), ylim=(-1, 1))
fig.set_layout_engine("tight")


spin_1_pos_0 = np.array([-0.8,0])
spin_2_pos_0 = spin_1_pos_0 * -1

spin1_kwargs = {"ax_plot": ax,
                "position" : spin_1_pos_0,
                "layout_params" : {
                    "spin_color":"blue",
                    # "eq_distances": (0.3, 0.5, 0.9),
                    "arrow_width": 20,
                    "arrow_length": 0.7,
                    "arrow_mutation": 50,
                    "ball_radius": 0.15,
                    "field_line_width": 3,
                },
                # "ball_alpha": 0,
                # "arrow_alpha": 0,
                "line_alpha": 0,
                "ball_alpha": 0,
                "arrow_alpha": 0,
                # "line_alpha": 1,
                
                }

spin2_kwargs = {"ax_plot": ax,
                "position" : spin_2_pos_0,
                "layout_params" : {
                    "spin_color":"deepskyblue",
                    "eq_distances": (0.6, 1, 1.8),
                    "arrow_width": 20,
                    "arrow_length": 0.7,
                    "arrow_mutation": 50,
                    "ball_radius": 0.15,
                    "field_line_width": 3,
                },
                # "ball_alpha": 0,
                # "arrow_alpha": 0,
                "line_alpha": 0,
                "ball_alpha": 0.,
                "arrow_alpha": 0,
                # "line_alpha": 1,
                
                }
                
spin1 = Spin2D(**spin1_kwargs)
spin2 = Spin2D(**spin2_kwargs)


# lab_frame_text = ax.text(0, 0.8, r"$\mathrm{Lab \; \; Frame}$", ha = "center", fontsize=50, math_fontfamily=math_fontfamily, alpha = 1)
spin1_frame_text = ax.text(0, 0.8, r"$\mathrm{Frame \; \; of} \; \; S_1$", ha = "center", fontsize=50, math_fontfamily=math_fontfamily, alpha = 1)

spin1_annotation = ax.text(spin_1_pos_0[0], 0.6, r"$\mathrm{Spin} \; \; S_1$", ha = "center", fontsize=40, math_fontfamily=math_fontfamily, alpha = 0)
spin2_annotation = ax.text(spin_2_pos_0[0], 0.6, r"$\mathrm{Spin} \; \; S_2$", ha = "center", fontsize=40, math_fontfamily=math_fontfamily, alpha = 0)


spin2_position = random_walk(N_t = t_8 - t_7,
                                            dim = 2, 
                                            random_force_var = 2*RANDOM_FORCE_VAR_POS, 
                                            k = SPRING_CONST_POS,
                                            c = DAMPING_CONST_POS,
                                            center = spin_2_pos_0) 

mag_field_on_spin_1 = np.zeros((t_8 - t_7, 2))

for i in range(t_8 - t_7):
    spin2.position = spin2_position[i]
    mag_field_on_spin_1[i] = spin1.get_mag_field_from_other(spin2)

spin2.position = spin_2_pos_0
max_perp_field = np.max(np.abs(mag_field_on_spin_1[:,0]))
max_par_field = np.max(np.abs(mag_field_on_spin_1[:,1]))

B_perp_slider = PrettySlider(
    ax_plot = ax,
    x_pos=(-1.1, -0.5),
    y_pos=(-0.8, -0.8),
    data_lim=(-max_perp_field*1.2, max_perp_field*1.2),
    arrow_style="<->",
    slider_dot_data=mag_field_on_spin_1[0,0],
    horizontal=True,
    alpha = 0,
    c = ("black", "red"),
    labels=(None, None, r"$\mathrm{B_{\perp}}$"),
    arrow_lw=4,
    ball_markersize= 25,
    label_size=35,
    center_label_offset=-0.15
)

B_par_slider = PrettySlider(
    ax_plot = ax,
    x_pos=(-1.3, -1.3),
    y_pos=(-0.4, 0.4),
    data_lim=(-max_par_field*1.2, max_par_field*1.2),
    arrow_style="<->",
    slider_dot_data=mag_field_on_spin_1[0,1],
    horizontal=False,
    alpha = 0,
    c = ("black", "red"),
    labels=(None, None, r"$\mathrm{B_{\parallel}}$"),
    arrow_lw=4,
    ball_markersize= 25,
    label_size=35,
    center_label_offset=-0.2
)

def animate(i):
   
    if i <= t_0:
        alpha = i / t_0
        spin1.ball_alpha = alpha*BALL_MAX_ALPHA
        spin2.ball_alpha = alpha*BALL_MAX_ALPHA
        spin1_annotation.set_alpha(alpha)
        spin2_annotation.set_alpha(alpha)
        spin1.generate_plot_objects()
        spin2.generate_plot_objects()


    if t_0 < i <= t_1:
        pass

    if t_1 < i <= t_2:
        new_alpha = (i - t_1 ) / (t_2 - t_1)
        spin1.arrow_alpha = new_alpha
        spin2.arrow_alpha = new_alpha
        spin1.generate_plot_objects()
        spin2.generate_plot_objects()
    
    if t_2 < i <= t_3:
        pass

    if t_3 < i <= t_4:
        new_alpha = (i - t_3 ) / (t_4 - t_3)
        spin2.line_alpha = new_alpha
        spin2.generate_plot_objects()
    
    if t_4 < i <= t_5:
        pass

    if t_5 < i <= t_6:
        new_alpha = (i - t_5 ) / (t_6 - t_5)
        B_par_slider.alpha = new_alpha
        B_perp_slider.alpha = new_alpha

    if t_6 < i <= t_7:
        pass

    if t_7 < i <= t_8:
        mag_field_index = i - t_7 - 1
        B_perp_slider.update_slider_dot( mag_field_on_spin_1[mag_field_index, 0] )
        B_par_slider.update_slider_dot( mag_field_on_spin_1[mag_field_index, 1] )
        spin2.position = spin2_position[mag_field_index]
        spin2.generate_plot_objects()    
    
    return ax

ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50)
                            # init_func=init,
                            # blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/local_fluctuations2D.{file_type}', fps = 30)
