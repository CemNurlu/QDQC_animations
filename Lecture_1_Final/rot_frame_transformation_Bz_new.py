from anim_base import  cache_then_save_funcanimation, PrettySlider, prepare_bloch_mosaic, math_fontfamily, file_type
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib import animation as anim
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm

##########################
# LONGER DURATIONS FOR ACTUAL ANIMATION
##########################

N_time = 1020
t_0 = 30 # Show static bloch sphere and B_z
t_1 = 50 # Show B_z equation
t_2 = 90 # Let it sink in for a bit
t_3 = 110 # Show W transformation equation and omega slider
t_4 = 140 # Let it sink in for a bit
t_5 = 160 # Show rotating bloch sphere and B_z equation
t_6 = 200 # Let it sink in for a bit
t_7 = 400 # Slide omega from 0 to omega < omega_L, shrink B_z and rotate sphere
t_8 = 440 #Let it sink in
t_9 = 640 # Slide omega from 0 to omega = omega_L, shrink B_z and rotate sphere
t_10= 680 # Let it sink in
t_11= 880 # Slide omega from 0 to omega > omega_L, shrink B_z and rotate sphere
t_12= 1000 # Let it sink in 
t_13 = N_time

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
    t_11//=10
    t_12//=10
    t_13//=10
   


bloch_mosaic = [["bloch_lab", "plot_between", "bloch_rot"],
                ["plot", "plot", "plot"]]

vector_colors = ["maroon","maroon"]#, "blue", "green", vector_colors[0], "orange", "pink", "yellow"]

bloch_kwargs = [{
    "vector_color": vector_colors,
    "vector_width": 3,
    },
    {
    "vector_color": vector_colors,
    "vector_width": 3,
    "vector_alpha" : [1,0],
    "xlabel": [r"$x ^\prime $", ''],
    "ylabel": [r"$y ^\prime $", '']
    }
]


gridspec_kw = {"height_ratios":[1,0.1], "width_ratios":[1, 0.7, 1]}
fig, ax_dict, sphere_dict = prepare_bloch_mosaic(bloch_mosaic, (12,8), bloch_kwargs, gridspec_kw=gridspec_kw)

ax_dict["plot"].set_axis_off()
ax_dict["plot_between"].set_axis_off()


B_lab = 0.8
B_min_rot = -0.8

assert B_lab > 0 and B_min_rot < 0, "B_lab and B_min_rot must be positive and negative respectively"

omega_max = np.pi/10
omega_L = omega_max *  B_lab / (B_lab - B_min_rot)




B_rot = np.zeros(t_9-t_6)
B_rot[0:t_7-t_6] = np.linspace(B_lab, 0, t_7-t_6)
B_rot[t_8-t_6:] = np.linspace(0, B_min_rot, t_9-t_8)

B_pos = 3/4*B_lab + 1/4*B_min_rot
B_zero = 0
B_neg = 1/4*B_lab + 3/4*B_min_rot

azim_angle_pos_rot_sphere = np.linspace(0,4*np.pi,t_7 - t_6)
azim_angle_pos_rot_sphere = -np.pi/3 + azim_angle_pos_rot_sphere
azim_angle_pos_rot_sphere = (azim_angle_pos_rot_sphere*180/np.pi)%360

azim_angle_zero_rot_sphere = np.linspace(0,8*np.pi,t_9 - t_8)
azim_angle_zero_rot_sphere = -np.pi/3 + azim_angle_zero_rot_sphere
azim_angle_zero_rot_sphere = (azim_angle_zero_rot_sphere*180/np.pi)%360


azim_angle_neg_rot_sphere = np.linspace(0,16*np.pi,t_11 - t_10)
azim_angle_neg_rot_sphere = -np.pi/3 + azim_angle_neg_rot_sphere
azim_angle_neg_rot_sphere = (azim_angle_neg_rot_sphere*180/np.pi)%360

# B_inds = int(len(B_rot)*0.4), int(len(B_rot)*0.6)


# B_rot[0:B_inds[0]] = np.linspace(B_lab, 0, B_inds[0])
# B_rot[B_inds[1]:] = np.linspace(0, B_min_rot, len(B_rot)-B_inds[1])

omega = (B_rot - B_lab) / (B_min_rot - B_lab) * omega_max
omega_pos = (B_pos - B_lab) / (B_min_rot - B_lab) * omega_max
omega_zero = (B_zero - B_lab) / (B_min_rot - B_lab) * omega_max
omega_neg = (B_neg - B_lab) / (B_min_rot - B_lab) * omega_max


# print(omega)
# exit()

# B_rot = omega / omega_max * (B_min_rot - B_lab) + B_lab

vector_cutoff = 0.15
B_rot[np.abs(B_rot) < vector_cutoff] = 0


azim_angle_rot_sphere = (-60 - np.cumsum(omega)*(180/np.pi)) % 360
B_lab_equation = ax_dict["plot"].text(1.7, 2.5, r'$B_z = \frac{ \omega_L } { \gamma }$', color = vector_colors[0], alpha = 0, size = 30, math_fontfamily = math_fontfamily,ha = "center" )

B_rot_equation = ax_dict["plot"].text(8.3, 2.5, r'$B \prime _{\!\!\!z} = \frac{ \omega_L - \omega } { \gamma }$', color = vector_colors[0], alpha = 0, size = 30, math_fontfamily = math_fontfamily,ha = "center")

push_down = 0.15
positive_detuning_text = ax_dict["plot_between"].text(0.55, 0.9 - push_down, r'$\omega < \omega_L$', color = "black", alpha = 0, size = 30, math_fontfamily = math_fontfamily,ha = "center")
negative_detuning_text = ax_dict["plot_between"].text(0.55, 0.9 - push_down, r'$\omega > \omega_L$', color = "black", alpha = 0, size = 30, math_fontfamily = math_fontfamily,ha = "center")
zero_detuning_text = ax_dict["plot_between"].text(0.55, 0.9 - push_down, r'$\omega = \omega_L$', color = "black", alpha = 0, size = 30, math_fontfamily = math_fontfamily,ha = "center")

omega_slider = PrettySlider(ax_dict["plot_between"], 
                x_pos=(0.1, 1), y_pos=(0.7 - push_down, 0.7 - push_down), data_lim = (-omega_max*0.01, omega_max*1.01),
                arrow_style="->", slider_dot_data=0, alpha = 0, c = ("black", "aquamarine"), 
                labels = (r"$0$ ", r"$\omega$",None), arrow_lw=3,
                label_size = 25, 
                label_c = ("black", "black", "red"))

line_zero_detuning = lines.Line2D([0.55,0.55],[0.675- push_down,0.725- push_down], lw = 3, color = "red",alpha = 0)
ax_dict["plot_between"].add_line(line_zero_detuning)
text_zero_detuning = ax_dict["plot_between"].text(0.55, 0.75- push_down, r"$\omega_{L}$", color = "red", alpha = 0, size = 25, math_fontfamily = math_fontfamily, ha = "center")

W_trans_equation = ax_dict["plot_between"].text(0.55, 0.3- push_down, r'$W = \mathrm{exp}(-i \omega t \, S_z)$', color = "black", alpha = 0, size = 25, math_fontfamily = math_fontfamily, ha = "center")
W_trans_arrow = FancyArrowPatch((0.35, 0.5- push_down), (0.75, 0.5- push_down), 
        mutation_scale=110,
        lw = 2, 
        ec = "black",
        fc = "aquamarine",
        alpha = 0,
        mutation_aspect=1.
    )
ax_dict["plot_between"].add_patch(W_trans_arrow)
ax_dict["plot_between"].set_xlim(0, 1)
ax_dict["plot_between"].set_ylim(0, 1)

ax_dict["plot"].set_xlim(0, 10)
ax_dict["plot"].set_ylim(0,1)

def animate(i):
    # ax_dict["bloch_avg"].set_title("Average of Bloch vectors")
    # ax_dict["bloch_super"].set_title("Superposition of Bloch vectors")
    if i <= t_0:
        if i == 0: # Show static bloch sphere and B_z
            sphere_dict["bloch_lab"].add_vectors([0,0,B_lab])
            sphere_dict["bloch_lab"].make_sphere()
    
    elif t_0 < i <= t_1: # Show B_z equation
        new_alpha = (i-t_0)/(t_1-t_0)
        B_lab_equation.set_alpha(new_alpha)
    
    elif t_1 < i <= t_2: # Let it sink in for a bit
        pass

    elif t_2 < i <= t_3: # Show W transformation equation and omega slider
        new_alpha = (i-t_2)/(t_3-t_2)
        omega_slider.alpha = new_alpha
        W_trans_arrow.set_alpha(new_alpha)
        W_trans_equation.set_alpha(new_alpha)
        line_zero_detuning.set_alpha(new_alpha)
        text_zero_detuning.set_alpha(new_alpha)
    
    elif t_3 < i <= t_4: # Let it sink in for a bit
        pass

    elif t_4 < i <= t_5: # Show rotating bloch sphere and B_z equation
        if i == t_4+1:
            sphere_dict["bloch_rot"].add_vectors([0,0,B_lab])
            sphere_dict["bloch_rot"].make_sphere()
        new_alpha = (i-t_4)/(t_5-t_4)
        B_rot_equation.set_alpha(new_alpha)
    
    elif t_5 < i <= t_6:# Let it sink in for a bit
        pass

    elif t_6 < i <= t_7: # Slide omega from 0 to omega < omega_L, shrink B_z and rotate sphere   
        t_trans = (4/5*t_6 + 1/5*t_7)
        temp_index = i - t_6 - 1

        ax_dict["bloch_rot"].azim = azim_angle_pos_rot_sphere[temp_index]
        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors([[0,0,B_lab],[0,0,B_pos]])

        if t_6 < i <= t_trans:
            new_alpha = (i-t_6)/(t_trans-t_6)
            positive_detuning_text.set_alpha(new_alpha)
            omega_slider.update_slider_dot(new_alpha*omega_pos)
            sphere_dict["bloch_rot"].vector_alpha = [1-new_alpha,new_alpha]
        
        sphere_dict["bloch_rot"].make_sphere()
        

            

    elif t_7 < i <= t_8: # Let it sink in for a bit
        t_trans = (4/5*t_7 + 1/5*t_8)

        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors([[0,0,B_pos],[0,0,B_lab]])

        if t_7 < i <= t_trans:
            new_alpha = (i-t_7)/(t_trans-t_7)
            positive_detuning_text.set_alpha(1 - new_alpha)
            omega_slider.update_slider_dot((1-new_alpha)*omega_pos)
            sphere_dict["bloch_rot"].vector_alpha = [1-new_alpha,new_alpha]
        
        sphere_dict["bloch_rot"].make_sphere()



    elif t_8 < i <= t_9: # Slide omega to omega = omega_L, shrink B_z and rotate sphere
        t_trans = (4/5*t_8 + 1/5*t_9)
        temp_index = i - t_8 - 1

        ax_dict["bloch_rot"].azim = azim_angle_zero_rot_sphere[temp_index]
        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors([[0,0,B_lab],[0,0,B_zero]])

        if t_8 < i <= t_trans:
            new_alpha = (i-t_8)/(t_trans-t_8)
            zero_detuning_text.set_alpha(new_alpha)
            omega_slider.update_slider_dot(new_alpha*omega_L)
            sphere_dict["bloch_rot"].vector_alpha = [1-new_alpha,new_alpha]
        
        sphere_dict["bloch_rot"].make_sphere() 

    elif t_9 < i <= t_10:# Let it sink in for a bit
        t_trans = (4/5*t_9 + 1/5*t_10)

        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors([[0,0,B_zero],[0,0,B_lab]])

        if t_9 < i <= t_trans:
            new_alpha = (i-t_9)/(t_trans-t_9)
            zero_detuning_text.set_alpha(1 - new_alpha)
            omega_slider.update_slider_dot((1-new_alpha)*omega_zero)
            sphere_dict["bloch_rot"].vector_alpha = [1-new_alpha,new_alpha]
        
        sphere_dict["bloch_rot"].make_sphere()

    elif t_10 < i <= t_11: # Slide to omega > omega_L, shrink B_z and rotate sphere
        t_trans = (4/5*t_10 + 1/5*t_11)
        temp_index = i - t_10 - 1

        ax_dict["bloch_rot"].azim = azim_angle_neg_rot_sphere[temp_index]
        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors([[0,0,B_lab],[0,0,B_neg]])

        if t_10 < i <= t_trans:
            new_alpha = (i-t_10)/(t_trans-t_10)
            negative_detuning_text.set_alpha(new_alpha)
            omega_slider.update_slider_dot(new_alpha*omega_neg)
            sphere_dict["bloch_rot"].vector_alpha = [1-new_alpha,new_alpha]
        
        sphere_dict["bloch_rot"].make_sphere()
        

    elif t_11 < i <= t_12:# Let it sink in for a bit
        pass
    """
    elif t_6 < i <= t_9:
        arrowLength = 0.2
        temp_index = i-t_6-1
        ax_dict["bloch_rot"].azim = azim_angle_rot_sphere[temp_index]
        sphere_dict["bloch_rot"].vectors = []
        B_auxvector = np.sign(B_rot[temp_index])*np.maximum(np.abs(B_rot[temp_index]),arrowLength)
        if B_rot[temp_index] != 0:
            sphere_dict["bloch_rot"].add_vectors([0,0,B_auxvector])
        sphere_dict["bloch_rot"].make_sphere()
        omega_slider.update_slider_dot(omega[temp_index])
    """

    return [ax for key, ax in ax_dict.items()] 

def init():
    return [ax for key, ax in ax_dict.items()]
    
ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init, 
                              blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/rot_frame_transformation_Bz_new.{file_type}', fps = 20 )