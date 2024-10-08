from anim_base import  cache_then_save_funcanimation, Spin2D,PrettySlider, random_walk, math_fontfamily, cache_then_save_funcanimation, file_type
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tqdm import tqdm
import numpy as np

N_time = 400

t_0 = 20 # Show the balls, spin labels and lab frame text
t_1 = 80 # Let the balls move
t_2 = 100 # Fade out everything except spin labels
t_3 = 250 # Do nothing
t_4 = 140 # Show the spin1-frame text and both balls
t_5 = 200 # Let the spin2 ball move
t_6 = 220 # Fade in quantization axis of spin2 ( keep it static)
t_7 = 260 # Continue moving the spin2 ball
t_8 = 280 # Fade in the magnetic field lines from the spin2 ball
t_9 = 320 # continue moving the spin2 ball
t_10 = 340 # Fade in the quantization axis of the spin1 ball


##########################
# SHORTER DURATIONS FOR DEBUGGING
##########################
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
    t_9 //= 5
    t_10 //= 5

# These constants are used for the random walk of the balls and the spin axis
RANDOM_FORCE_VAR_POS = 0.005
SPRING_CONST_POS = 0.5
DAMPING_CONST_POS = 0.5

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
                    "spin_color":"red",
                    # "eq_distances": (0.3, 0.5, 0.9),
                    "arrow_width": 20,
                    "arrow_length": 0.7,
                    "arrow_mutation": 50,
                    "ball_radius": 0.15,
                    "field_line_width": 3,
                    "mag_line_color": "red",
                    "eq_distances": (0.5,1,1.5,1.625,1.75),
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
                    "mag_line_color": "deepskyblue",
                    "eq_distances": (0.5,1,1.5,1.625,1.75),
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

lab_frame_text = ax.text(0, 0.8, r"$\mathrm{Spin \; \; Coupling}$", ha = "center", fontsize=50, math_fontfamily=math_fontfamily, alpha = 1)
spin1_frame_text = ax.text(0, 0.8, r"$\mathrm{Frame \; \; of} \; \; S_1$", ha = "center", fontsize=50, math_fontfamily=math_fontfamily, alpha = 0)

spin1_annotation = ax.text(spin_1_pos_0[0], 0.6, r"$\mathrm{Spin} \; \; S_1$", ha = "center", fontsize=40, math_fontfamily=math_fontfamily, alpha = 0)
spin2_annotation = ax.text(spin_2_pos_0[0], 0.6, r"$\mathrm{Spin} \; \; S_2$", ha = "center", fontsize=40, math_fontfamily=math_fontfamily, alpha = 0)

# The random walk for both balls in the lab frame
shaking_position = np.array([
    random_walk(N_t = N_time, 
                dim = 2, 
                random_force_var = RANDOM_FORCE_VAR_POS, 
                k = SPRING_CONST_POS,
                c = DAMPING_CONST_POS,
                center = c) 
                for c in [spin_1_pos_0, spin_2_pos_0]
                ])

#Pseudo-random spin flipping
N_angle_points = N_time
angle_target = np.concatenate(
    [
    np.linspace(0, np.pi, N_angle_points-N_angle_points//5).reshape(-1,1),
    np.ones((N_angle_points//5, 1))*np.pi
    ],
    axis=0
)
SPRING_CONST_ANG = 0.03
DAMPING_CONST_ANG = 0.03
RANDOM_FORCE_VAR_ANG = 0.05
spin_angle = np.array([
    random_walk(N_t = N_angle_points,
                dim = 1,
                random_force_var=RANDOM_FORCE_VAR_ANG,
                k = SPRING_CONST_ANG,
                c = DAMPING_CONST_ANG,
                center = c).flatten()
                for c in [0,0]
                ])

"""
# The random walk for the spin1 axis in the lab frame
spin2_in_spin1_frame_position = random_walk(N_t = t_11 - t_3, 
                                            dim = 2, 
                                            random_force_var = 2*RANDOM_FORCE_VAR_POS, 
                                            k = SPRING_CONST_POS,
                                            c = DAMPING_CONST_POS,
                                            center = spin_2_pos_0)
""" 


#Energy Levels:
Spin_1_level_up = PrettySlider(
    ax_plot=ax,
    x_pos=(-1.3,-1),
    y_pos=(0.1,0.1),
    data_lim = (0,0),
    arrow_style = "-",
    #slider_dot_data=mag_field_on_spin_1[0,1],
    horizontal=True,
    alpha = 0,
    c = ("red", "red"),
    labels=(None, None, r"$|e_{1}>$"),
    arrow_lw=4,
    ball_markersize= 0,
    label_size=30,
    center_label_offset=0.055,
)
Spin_1_level_down = PrettySlider(
    ax_plot=ax,
    x_pos=(-1.3,-1),
    y_pos=(-0.1,-0.1),
    data_lim = (0,0),
    arrow_style = "-",
    #slider_dot_data=mag_field_on_spin_1[0,1],
    horizontal=True,
    alpha = 0,
    c = ("red", "red"),
    labels=(None, None, r"$|g_{1}>$"),
    arrow_lw=4,
    ball_markersize= 0,
    label_size=30,
    center_label_offset=0.055,
)

Spin_2_level_up = PrettySlider(
    ax_plot=ax,
    x_pos=(1.3,1),
    y_pos=(0.1,0.1),
    data_lim = (0,0),
    arrow_style = "-",
    #slider_dot_data=mag_field_on_spin_1[0,1],
    horizontal=True,
    alpha = 0,
    c = ("deepskyblue", "deepskyblue"),
    labels=(None, None, r"$|e_{2}>$"),
    arrow_lw=4,
    ball_markersize= 0,
    label_size=30,
    center_label_offset=0.055,
)

Spin_2_level_down = PrettySlider(
    ax_plot=ax,
    x_pos=(1.3,1),
    y_pos=(-0.1,-0.1),
    data_lim = (0,0),
    arrow_style = "-",
    #slider_dot_data=mag_field_on_spin_1[0,1],
    horizontal=True,
    alpha = 0,
    c = ("deepskyblue", "deepskyblue"),
    labels=(None, None, r"$|g_{2}>$"),
    arrow_lw=4,
    ball_markersize= 0,
    label_size=30,
    center_label_offset=0.055,
)

def animate(i):
   
    
    if  i <= N_time:
        pos_index = i -1
        spin1.position = shaking_position[0][pos_index]
        spin2.position = shaking_position[1][pos_index]
        spin1.rotation = spin_angle[0][pos_index]
        spin2.rotation = spin_angle[1][pos_index]

    if i <= t_0:
        alpha = i / t_0
        spin1.ball_alpha = alpha*BALL_MAX_ALPHA
        spin2.ball_alpha = alpha*BALL_MAX_ALPHA

    if t_1 < i <= t_2:
        alpha = (i - t_1 ) / (t_2 - t_1)
        spin1_annotation.set_alpha(alpha)
        spin2_annotation.set_alpha(alpha)
        spin2.arrow_alpha = alpha
        spin1.arrow_alpha = alpha

    if t_2 < i <= t_3:
        pass
    spin1.generate_ball_patch()
    spin2.generate_ball_patch()
    spin1.generate_plot_objects()
    spin2.generate_plot_objects()

    return ax

ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50)
                            # init_func=init,
                            # blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/shaking_whole_picture.{file_type}', fps = 30)

# cache_then_save_funcanimation(ani, f'animations/test/spin_coupling_2D_full_story.{file_type}', fps = 30 )