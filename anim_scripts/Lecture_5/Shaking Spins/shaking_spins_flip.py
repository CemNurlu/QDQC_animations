from anim_base import  cache_then_save_funcanimation, Spin2D, PrettySlider, random_walk, math_fontfamily, cache_then_save_funcanimation, file_type
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tqdm import tqdm
import numpy as np

"""
The setting is similar to the animation from Lecture 2: two atoms with spins (e.g. colorcoded, 1st red and 2nd blue), with two level pairs (same colors as the atoms/spins). First 
show the magnetic field created by the spin 1 on the spin 2, then the field created by the 
spin 2 on spin 1. First clip: the atom 1 shakes, the levels of the spin 2 change the splitting, 
then vice versa. Then the second clip: atom 1 randomly shakes, and we see the spin 2 rotate, 
eventually reaching the opposite direction. Same with atoms/spins changing roles. Third clip: 
both atoms shake and both spins rotate.
"""


increment = 50
#N_time = 650 + 11*increment
t_0 = 20 # Show the balls, spin labels and S1 frame text
t_1 = 60  # Do nothing
t_2 = 80 # Fade in axises
t_3 = 120 # Do nothing
t_4 = 140 # Fade in spin2 field lines
t_5 = 145 # Do nothing
t_6 = 149 # Show spin 1 levels
t_7 = 190 # Do nothing
t_8 = 240 # Move the spin2 ball and update the sliders
t_9 = 280 # Fade out initial levels and mag field and flip S2
t_10= 320 #Do nothing
t_11= 340 #Flip spin2
t_12= 380 #Do nothing
t_13= 430 #Fade in spin2 magnetic field lines
t_14= 440 #Do nothing
t_15= 460 #Show spin1 levels
t_16= 500 #Do nothing
t_17= 550 #Change level splitting
t_18= t_17 + 1
t_19= t_18 + 1
t_20= t_19 + increment
t_21= t_20 + increment
t_22= t_21 + increment
t_23= t_22 + increment
t_24= t_23 + increment
t_25= t_24 + increment
t_26= t_25 + increment
t_27= t_26 + increment
t_28= t_27 + increment
t_29= t_28 + increment

N_time = t_29 + increment

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
    t_11//= 5
    t_12 //= 5
    t_13 //= 5
    t_14 //=5
    t_15//=5
    t_16//=5
    t_17//=5
    t_18 //= 5
    t_19 //= 5
    t_20 //= 5
    t_21//= 5
    t_22 //= 5
    t_23 //= 5
    t_24 //=5
    t_25//=5
    t_26//=5
    t_27//=5
    t_28//=5
    t_29//=5

# These constants are used for the random walk of the balls and the spin axis
RANDOM_FORCE_VAR_POS = 0.005
SPRING_CONST_POS = 0.5
DAMPING_CONST_POS = 0.5
RANDOM_FORCE_VAR_ANG = 0
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
lab_frame_position = np.array([
    random_walk(N_t = N_time, 
                dim = 2, 
                random_force_var = 0, 
                k = 0,
                c = 0,
                center = c) 
                for c in [spin_1_pos_0, spin_2_pos_0]
                ])



# Shaking Spins postions
shaking_position = np.array([
    random_walk(N_t = N_time,
                dim = 2,
                random_force_var = 2*RANDOM_FORCE_VAR_POS,
                k = SPRING_CONST_POS,
                c = DAMPING_CONST_POS,
                center = c)
                for c in [spin_1_pos_0, spin_2_pos_0]
                ])

#Pseudo-random spin flipping
N_angle_points = increment
angle_target = np.concatenate(
    [
    np.linspace(0, np.pi, N_angle_points-N_angle_points//5).reshape(-1,1),
    np.ones((N_angle_points//5, 1))*np.pi
    ],
    axis=0
)
SPRING_CONST_ANG = 0.3
DAMPING_CONST_ANG = 0.2
RANDOM_FORCE_VAR_ANG = 0.05
spin_angle = random_walk(
    N_t = N_angle_points,
    dim = 1,
    random_force_var=RANDOM_FORCE_VAR_ANG,
    k = SPRING_CONST_ANG,
    c = DAMPING_CONST_ANG,
    center = angle_target,
).flatten()



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

New_1_level_up = PrettySlider(
    ax_plot=ax,
    x_pos=(-1.3,-1),
    y_pos=(0.3,0.3),
    data_lim = (0,0),
    arrow_style = "-",
    #slider_dot_data=mag_field_on_spin_1[0,1],
    horizontal=True,
    alpha = 0,
    c = ("red", "red"),
    labels=(None, None, r"$|e'_{1}>$"),
    arrow_lw=4,
    ball_markersize= 0,
    label_size=30,
    center_label_offset=0.055,
)

New_1_level_down = PrettySlider(
    ax_plot=ax,
    x_pos=(-1.3,-1),
    y_pos=(-0.3,-0.3),
    data_lim = (0,0),
    arrow_style = "-",
    #slider_dot_data=mag_field_on_spin_1[0,1],
    horizontal=True,
    alpha = 0,
    c = ("red", "red"),
    labels=(None, None, r"$|g'_{1}>$"),
    arrow_lw=4,
    ball_markersize= 0,
    label_size=30,
    center_label_offset=0.055,
)

New_2_level_up = PrettySlider(
    ax_plot=ax,
    x_pos=(1.3,1),
    y_pos=(0.3,0.3),
    data_lim = (0,0),
    arrow_style = "-",
    #slider_dot_data=mag_field_on_spin_1[0,1],
    horizontal=True,
    alpha = 0,
    c = ("deepskyblue", "deepskyblue"),
    labels=(None, None, r"$|e'_{2}>$"),
    arrow_lw=4,
    ball_markersize= 0,
    label_size=30,
    center_label_offset=0.055,
)
New_2_level_down = PrettySlider(
    ax_plot=ax,
    x_pos=(1.3,1),
    y_pos=(-0.3,-0.3),
    data_lim = (0,0),
    arrow_style = "-",
    #slider_dot_data=mag_field_on_spin_1[0,1],
    horizontal=True,
    alpha = 0,
    c = ("deepskyblue", "deepskyblue"),
    labels=(None, None, r"$|g'_{2}>$"),
    arrow_lw=4,
    ball_markersize= 0,
    label_size=30,
    center_label_offset=0.055,
)

"""
mag_field_on_spin_1 = np.zeros((t_8 - t_7, 2))

for i in range(t_8 - t_7):
    spin2.position = spin2_position[i]
    mag_field_on_spin_1[i] = spin1.get_mag_field_from_other(spin2)

spin2.position = spin_2_pos_0
max_perp_field = np.max(np.abs(mag_field_on_spin_1[:,0]))
max_par_field = np.max(np.abs(mag_field_on_spin_1[:,1]))

spin_level_up_pos = np.linspace(0.1,0.2,t_17-t_16)
spin_level_down_pos = np.linspace(-0.1,-0.2,t_17-t_16)



B_perp_slider = PrettySlider(
    ax_plot = ax,
    x_pos=(-1.1, -0.5),
    y_pos=(-0.8, -0.8),
    data_lim=(-max_perp_field*1.2, max_perp_field*1.2),
    arrow_style="<->",
    slider_dot_data=mag_field_on_spin_1[0,0],
    horizontal=True,
    alpha = 0,
    c = ("red", "red"),
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
"""
def animate(i):

    if  i <= N_time: #Set tiny brownian motion
        pos_index = i -1

   
    if i <= t_0: #Fade in balls
        alpha = i / t_0
        spin1.ball_alpha = alpha*BALL_MAX_ALPHA
        spin2.ball_alpha = alpha*BALL_MAX_ALPHA
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]

    if t_1 < i <= t_2: #Fade in Spins and anotations
        alpha = (i - t_1 ) / (t_2 - t_1)
        spin1_annotation.set_alpha(alpha)
        spin2_annotation.set_alpha(alpha)
        spin2.arrow_alpha = alpha
        spin1.arrow_alpha = alpha
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]
    
    if t_2 < i <= t_3:
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]

    if t_3 < i <= t_4: #Fade in S2 field lines
        new_alpha = (i - t_3 ) / (t_4 - t_3)
        spin2.line_alpha = new_alpha
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]
    
    if t_4 < i <= t_5:
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]

    if t_5 < i <= t_6: #Fade in S1 Original Energy Levels
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]
        new_alpha = (i - t_5 ) / (t_6 - t_5)
        """
        B_par_slider.alpha = new_alpha
        B_perp_slider.alpha = new_alpha
        
        """
        """
        Spin_1_level_down.alpha = new_alpha
        Spin_1_level_up.alpha = new_alpha
        """

    if t_6 < i <= t_7:
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]


    if t_7 < i <= t_8: #Fade out S2 mag lines
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]
        new_alpha = 1 - (i-t_7)/(t_8-t_7)
        spin2.line_alpha = new_alpha
    
    if t_8 < i <= t_9:
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]
    
    if t_9 < i <= t_10:#Shake S2
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = shaking_position[1][pos_index]

    if t_10 < i <= t_11: #Fade in S2 Field lines
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = shaking_position[1][pos_index]
        new_alpha = (i-t_10)/(t_11-t_10)
        spin2.line_alpha = new_alpha
    
    if t_11 < i <= t_12:
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = shaking_position[1][pos_index]
    
    if t_12 < i <= t_13: #Flip S1
        index = i - t_12 - 1
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = shaking_position[1][pos_index]
        spin1.rotation = spin_angle[index]
    
    if t_13 < i <= t_14:#Fade out S2 field lines
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = shaking_position[1][pos_index]
        new_alpha = 1 - (i-t_13)/(t_14-t_13)
        spin2.line_alpha = new_alpha


    if t_14 < i <= t_15: #Rotate S1 back
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = shaking_position[1][pos_index]
        new_rotation = np.pi*(1-(i-t_14)/(t_15-t_14))
        spin1.rotation = new_rotation
    
    if t_15 < i <= t_16:
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]

    #Now the same with S2


    if t_16 < i <= t_17: #Fade in S1 field lines
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]
        new_alpha = (i - t_16 ) / (t_17 - t_16)
        spin1.line_alpha = new_alpha
    
    if t_17 < i <= t_18:
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]

    if t_18 < i <= t_19: #Pass
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]
        new_alpha = (i - t_18 ) / (t_19 - t_18)
        """
        B_par_slider.alpha = new_alpha
        B_perp_slider.alpha = new_alpha
        
        
        Spin_2_level_down.alpha = new_alpha
        Spin_2_level_up.alpha = new_alpha
        """

    if t_19 < i <= t_20:
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]

    if t_20 < i <= t_21: #Fade out S1 field lines
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]
        new_alpha = 1 - (i - t_20)/(t_21 - t_20)
        spin1.line_alpha = new_alpha
    
    if t_21 < i <= t_22:
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]
    
    if t_22 < i <= t_23: #Shake S1
        spin1.position = shaking_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]
    
    if t_23 < i <= t_24:#Fade in S1 Field lines
        spin1.position = shaking_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]
        new_alpha = (i-t_23)/(t_24-t_23)
        spin1.line_alpha = new_alpha
    
    if t_24 < i <= t_25:
        spin1.position = shaking_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]
    
    if t_25 < i <= t_26: #Flip S2
        index = i - t_25 - 1
        spin1.position = shaking_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]
        spin2.rotation = spin_angle[index]
    
    if t_26 < i <= t_27:#Fade out S1 field lines
        spin1.position = shaking_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]
        new_alpha = 1 - (i - t_26)/(t_27 - t_26)
        spin1.line_alpha = new_alpha


    if t_27 < i <= t_28: #Rotate S2 back
        spin1.position = shaking_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]
        new_rotation = np.pi*(1-(i-t_27)/(t_28-t_27))
        spin2.rotation = new_rotation
        
    
    if t_28 < i <= t_29:
        spin1.position = lab_frame_position[0][pos_index]
        spin2.position = lab_frame_position[1][pos_index]



    spin1.generate_ball_patch()
    spin2.generate_ball_patch()
    spin1.generate_plot_objects()
    spin2.generate_plot_objects()

    return ax

ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50)
                            # init_func=init,
                            # blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/shaking_spins_flip.{file_type}', fps = 30)
