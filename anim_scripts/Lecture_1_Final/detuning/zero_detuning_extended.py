from anim_base import  cache_then_save_funcanimation,  prepare_bloch_mosaic, math_fontfamily, file_type
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as anim
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm

##########################
# LONGER DURATIONS FOR ACTUAL ANIMATION
##########################

N_time = 1350

# Prelude
t_0 = 50
#Total lab field
t_1 = 150 
#Driving decomposition
t_2 = 250 
#Time evolution of just clockwise driving
t_3 = 350
#Time evolution of total lab field
t_4 = 450
#Rotating frame before transforming B_zeeman
t_5 = 550
#Rotating frame after transforming B_zeeman
t_6 = 650 # Show W transformation equation
#Fade in rotating total field
t_7 = 750 # Let it sink in for a bit
t_8 = 850 # Show rot frame text
t_9 =  950# Let it sink in for a bit
t_10 = 1050 # Show rot frame equation
t_11 = 1150 # Let it sink in for a bit
t_12 = 1250 # Show B_rot bloch sphere
t_13 = 1350 # Show time evolution of B_rot
t_14 = N_time

##########################
# SHORTER DURATIONS FOR DEBUGGING
##########################

DEBUG = False
#DEBUG = True
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
    


bloch_mosaic = [["bloch_lab", "plot_between", "bloch_rot"],
                ["plot", "plot", "plot"]]

vector_colours_lab = ["maroon","purple","deeppink","deepskyblue","red"]
vector_colours_rot = ["maroon", "deeppink", "red","maroon","red"]

bloch_kwargs = [{
    "vector_color": vector_colours_lab,
    "vector_alpha" : [1,1,0,0,0],
    "vector_width": 3,
    },
    {
    "vector_color": vector_colours_rot,
    "vector_width": 3,
    "xlabel": [r"$x^\prime$", ''],
    "ylabel": [r"$y^\prime$", '']
    }
]


gridspec_kw = {"height_ratios":[1,0.4], "width_ratios":[1, 0.5, 1]}
fig, ax_dict, sphere_dict = prepare_bloch_mosaic(bloch_mosaic, (12,8), bloch_kwargs, gridspec_kw=gridspec_kw)

ax_dict["plot"].set_axis_off()
ax_dict["plot_between"].set_axis_off()

B_x_max = 0.7
B_zeeman_lab_z = 1
B_zeeman_rot_z = 0

phi_0 = 0
phi_end = 26*np.pi*2
phi = np.linspace(phi_0, phi_end, t_13-t_0)

azim_angle_rot_sphere = (-60 - (phi[:]*180/np.pi)) % 360

B_zeeman_lab = np.zeros((len(phi), 3))
B_zeeman_lab[:,2] = B_zeeman_lab_z


B_zeeman_rot = np.zeros((len(phi), 3))
B_zeeman_rot[:,2] = B_zeeman_rot_z


B_drive_lab = np.zeros((len(phi), 3))
B_drive_lab[:,0] = B_x_max*np.cos(phi)
B_drive_lab[:,1] = B_x_max * np.sin(phi)


B_drive_fast = np.zeros((len(phi), 3))
B_drive_fast[:,0] = B_x_max*np.cos(-phi)
B_drive_fast[:,1] = B_x_max * np.sin(-phi)

B_drive_total = 0.5*(B_drive_fast + B_drive_lab)
arrow_length = 0.2

for i in range(len(B_drive_total)):
    B_drive_total[i] = np.maximum(np.abs(B_drive_total[i]),arrow_length)*np.sign(B_drive_total[i])

B_drive_rot = np.zeros((len(phi), 3))
B_drive_rot[:,0] = B_x_max

B_total_lab = np.zeros((len(phi), 3))
B_total_lab = B_zeeman_lab + B_drive_total
B_total_simple = B_zeeman_lab + B_drive_lab

B_total_aux = np.zeros((len(phi), 3))
B_total_aux = B_zeeman_lab + B_drive_rot

B_total_rot = np.zeros((len(phi), 3))
B_total_rot = B_zeeman_rot + B_drive_rot


sphere_dict["bloch_lab"].add_vectors([B_zeeman_lab[0], B_drive_lab[0], B_total_lab[0]])


# B_x_dec_equation = ax_dict["plot"].text(-0.6, 1.4, r'$H_{\mathrm{driving}} \; = \; 2hS_x \mathrm{cos}\omega t$', color = vector_colors[0], alpha = 0, size = 18)
#Lab texts
B_lab_H_text = ax_dict["plot"].text(-1.3, 0.2, r'$H \; =$', color = "red", alpha = 0, size = 25, math_fontfamily= math_fontfamily)
B_lab_H_zeeman_text = ax_dict["plot"].text(-1.1, 0.2, r'$H_{\mathrm{Zeeman}} + $', color = vector_colours_lab[0], alpha = 0,  size = 25, math_fontfamily= math_fontfamily)
B_lab_H_drive_text = ax_dict["plot"].text(-0.7, 0.2, r"$H_{\mathrm{driving}} \; = \;$", color = vector_colours_lab[1], alpha = 0,  size = 25, math_fontfamily= math_fontfamily)
B_lab_H_drive_text_pink = ax_dict["plot"].text(-0.7, 0.2, r"$H_{\mathrm{driving}} \; = \;$", color = vector_colours_lab[2], alpha = 0,  size = 25, math_fontfamily= math_fontfamily)
B_lab_H_zeeman_eq = ax_dict["plot"].text(-1.45, 0.1, r'$=\omega_L S_z + $', color = vector_colours_lab[0], alpha = 0,  size = 25, math_fontfamily= math_fontfamily)
B_lab_H_drive_eq = ax_dict["plot"].text(-1.05, 0.1, r"$h \: [ S_x \: \mathrm{cos}\omega t + S_y \: \mathrm{sin}\omega t ]$", color = vector_colours_lab[2], alpha = 0,  size = 25, math_fontfamily= math_fontfamily)
B_lab_H_drive_eq_2 = ax_dict["plot"].text(-0.80, 0.0, r"$ + h \:[ S_x \: \mathrm{cos}\omega t - S_y \: \mathrm{sin}\omega t ]$", color = vector_colours_lab[3], alpha = 0,  size = 25, math_fontfamily= math_fontfamily, ha = "center")

#Rotation Texts
B_rot_H_text = ax_dict["plot"].text(0.82, 0.2, r"$H \prime \; =$", color = "red", alpha = 0,  size = 25, math_fontfamily= math_fontfamily)
B_rot_H_zeeman_text = ax_dict["plot"].text(1.07, 0.2, r"$H \prime _{\!\!\mathrm{Zeeman}} + $", color = vector_colours_rot[0], alpha = 0,  size = 25, math_fontfamily= math_fontfamily)
B_rot_H_drive_text = ax_dict["plot"].text(1.50, 0.2, r"$H \prime _{\!\!\!\mathrm{driving}} \; = \;$", color = vector_colours_rot[1], alpha = 0,  size = 25, math_fontfamily= math_fontfamily)
B_rot_H_zeeman_eq = ax_dict["plot"].text(0.95, 0.1, r"$= ( \omega_L - \omega ) \:  S \prime _{\!\!\!\!z} + $", color = vector_colours_rot[0], alpha = 0,  size = 25, math_fontfamily= math_fontfamily)
B_rot_H_drive_eq = ax_dict["plot"].text(1.65, 0.1, r"$h \, S \prime _{\!\!\!\!x} \;$", color = vector_colours_rot[1], alpha = 0,  size = 25, math_fontfamily= math_fontfamily)
B_rot_H_drive_eq_2 = ax_dict["plot"].text(1.38, 0.0, r"$=h \, S \prime _{\!\!\!\!x}$", color = vector_colours_rot[2], alpha = 0, size = 25, math_fontfamily = math_fontfamily, ha = "center")

W_trans_equation = ax_dict["plot_between"].text(0.1, 0.3, r'$W = \mathrm{exp}(-i \omega t \: S_z)$', color = "black", alpha = 0, size = 25, math_fontfamily = math_fontfamily)
W_trans_arrow = FancyArrowPatch((0.3, 0.5), (0.9, 0.5), 
        # arrowstyle='->',
        mutation_scale=120,
        lw = 2, 
        ec = "black",
        fc = "aquamarine",
        alpha = 0
    )

omega_gt_omega_L = ax_dict["plot_between"].text(0.55, 0.7, r'$\omega = \omega_L$', color = "black", alpha = 0, size = 30, math_fontfamily = math_fontfamily, ha = "center")

ax_dict["plot_between"].add_patch(W_trans_arrow)
ax_dict["plot_between"].set_xlim(0, 1)
ax_dict["plot_between"].set_ylim(0, 1)


ax_dict["plot"].set_xlim(-1.5, 2.)
ax_dict["plot"].set_ylim(-0.1, 0.25)

def animate(i):
    trans_fraction = 1/8
    B_time_index = 0
    if i <= t_0: #Prelude
        new_alpha = i/t_0
        sphere_dict["bloch_lab"].make_sphere()
    
    if i > t_0: #Set time index definition
        B_time_index = i - t_0 - 1
    
    if t_0 < i <= t_1:  #Total lab field 
        sphere_dict["bloch_lab"].vectors = []
        sphere_dict["bloch_lab"].add_vectors([B_zeeman_lab[B_time_index], B_drive_total[B_time_index],[0,0,0],[0,0,0], B_total_lab[B_time_index]])
        t_trans = trans_fraction*(t_1-t_0) + t_0
        if i <= t_trans: #Fade in total lab field
            new_alpha = (i-t_0)/(t_trans-t_0)
            sphere_dict["bloch_lab"].vector_alpha = [1,1,0,0,new_alpha]

            B_lab_H_text.set_alpha(new_alpha)
            B_lab_H_zeeman_text.set_alpha(new_alpha)
            B_lab_H_drive_text.set_alpha(new_alpha)
        
        if i >= t_1 - t_trans: #Fade out total lab field and driving
            new_alpha = (i-(t_1 - t_trans))/(t_trans)
            sphere_dict["bloch_lab"].vector_alpha = [1,1-new_alpha,0,0,1-new_alpha]
        
        sphere_dict["bloch_lab"].make_sphere()

    if t_1 < i <= t_2: #Driving decomposition
        sphere_dict["bloch_lab"].vectors = []
        sphere_dict["bloch_lab"].add_vectors([B_zeeman_lab[B_time_index],[0,0,0], B_drive_lab[B_time_index], B_drive_fast[B_time_index]])
        t_trans = trans_fraction*(t_2-t_1) + t_1
        t_trans_2 = t_2 - (t_trans-t_1)
        if i <= t_trans: #Fade in clockwise and counter-clockwise
            new_alpha = (i-t_1)/(t_trans-t_1)
            sphere_dict["bloch_lab"].vector_alpha = [1,0,new_alpha,new_alpha,0]

            B_lab_H_zeeman_eq.set_alpha(new_alpha)
            B_lab_H_drive_eq.set_alpha(new_alpha)
            B_lab_H_drive_eq_2.set_alpha(new_alpha)
        
        if i >= t_trans_2: #Fade out counter-clockwise
            new_alpha = (i - t_trans_2)/(t_2-t_trans_2)
            sphere_dict["bloch_lab"].vector_alpha = [1,0,1,1-new_alpha,0]

            B_lab_H_drive_eq_2.set_alpha(1-new_alpha)
            B_lab_H_drive_text.set_alpha(1-new_alpha)
            B_lab_H_drive_text_pink.set_alpha(new_alpha)
        
        sphere_dict["bloch_lab"].make_sphere()
    

    if t_2 < i <= t_3: #Time evolution of just clockwise driving
        sphere_dict["bloch_lab"].vectors = []
        sphere_dict["bloch_lab"].add_vectors([B_zeeman_lab[B_time_index], [0,0,0],B_drive_lab[B_time_index]])
        sphere_dict["bloch_lab"].make_sphere()

    if t_3 < i <= t_4: #Time evolution of total lab field
        sphere_dict["bloch_lab"].vectors = []
        sphere_dict["bloch_lab"].add_vectors([B_zeeman_lab[B_time_index], [0,0,0],B_drive_lab[B_time_index], [0,0,0],B_total_simple[B_time_index]])
        
        t_trans = trans_fraction*(t_4-t_3) + t_3
        t_trans_2 = t_4 - (t_trans - t_3)
        if i <= t_trans:
            new_alpha = (i-t_3)/(t_trans-t_3)
            sphere_dict["bloch_lab"].vector_alpha = [1,0,1,0,new_alpha]
            sphere_dict["bloch_lab"].make_sphere()

        if i >= t_trans_2: #Fade in transformation arrow
            new_alpha = (i-t_trans_2)/(t_4-t_trans_2)
            W_trans_arrow.set_alpha(new_alpha)
            W_trans_equation.set_alpha(new_alpha)
            omega_gt_omega_L.set_alpha(new_alpha)
        
        sphere_dict["bloch_lab"].make_sphere()
    
    if t_4 < i <= t_13: #Keep bloch_lab the same for the rest of the anim
        sphere_dict["bloch_lab"].vectors = []
        sphere_dict["bloch_lab"].add_vectors([B_zeeman_lab[B_time_index], [0,0,0],B_drive_lab[B_time_index], [0,0,0],B_total_simple[B_time_index]])
        sphere_dict["bloch_lab"].make_sphere()
    

    if t_4 < i <= t_5: #Rotating frame before transforming B_zeeman
        new_alpha_vector = [1,1,1]
        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors([B_zeeman_lab[B_time_index],
                                                  B_drive_rot[B_time_index]]
                                                  )
        ax_dict["bloch_rot"].azim = azim_angle_rot_sphere[B_time_index]
        sphere_dict["bloch_rot"].vector_alpha = new_alpha_vector

        t_trans = trans_fraction*(t_5-t_4) + t_4
        t_trans_2 = t_5 - (t_trans - t_4)
        if i <= t_trans:
            new_alpha = (i-t_4)/(t_trans-t_4)
            B_rot_H_text.set_alpha(new_alpha)
            B_rot_H_zeeman_text.set_alpha(new_alpha)
            B_rot_H_drive_text.set_alpha(new_alpha)

        if i >= t_trans_2:
            new_alpha = (i- t_trans_2)/(t_5-t_trans_2)
            sphere_dict["bloch_rot"].vector_alpha = [1-new_alpha,1,0,0,0]
            B_rot_H_zeeman_eq.set_alpha(new_alpha)
            B_rot_H_drive_eq.set_alpha(new_alpha)

        
        sphere_dict["bloch_rot"].make_sphere()
        

    if t_5 < i <= t_6: #Rotating frame after transforming B_zeeman
        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors([B_zeeman_rot[B_time_index],
                                                  B_drive_rot[B_time_index]]
                                                  )
        ax_dict["bloch_rot"].azim = azim_angle_rot_sphere[B_time_index]
        sphere_dict["bloch_rot"].vector_alpha = [1,1,0,0,0]

        t_trans = trans_fraction*(t_6-t_5) + t_5
        t_trans = t_6 - (t_trans - t_5)
        if i >= t_trans:
            new_alpha = (i-t_trans)/(t_6-t_trans)
            sphere_dict["bloch_rot"].vector_alpha = [1-new_alpha,1,0,0,0]

        sphere_dict["bloch_rot"].make_sphere()
        
    
    if t_6 < i <= t_7: #Fade in rotating total field
        new_alpha = (i-t_6)/(t_7-t_6)
        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors([B_zeeman_rot[B_time_index],
                                                  B_drive_rot[B_time_index],
                                                  B_total_rot[B_time_index]]
                                                  )
        sphere_dict["bloch_rot"].vector_alpha = [1,1,new_alpha]
        sphere_dict["bloch_rot"].make_sphere()

        B_rot_H_drive_eq_2.set_alpha(new_alpha)

    
    if t_7 < i <= t_13: #Keep it for the rest of the animation
        sphere_dict["bloch_rot"].vectors = []
        sphere_dict["bloch_rot"].add_vectors([B_zeeman_rot[B_time_index],
                                                  B_drive_rot[B_time_index],
                                                  B_total_rot[B_time_index]]
                                                  )
        sphere_dict["bloch_rot"].make_sphere()

    """
    if t_1 < i <= t_2:
        new_alpha = (i-t_1)/(t_2-t_1)
        B_lab_H_text.set_alpha(new_alpha)
        B_lab_H_zeeman_text.set_alpha(new_alpha)
        B_lab_H_drive_text.set_alpha(new_alpha)

    if t_3 < i <= t_4:
        new_alpha = (i-t_3)/(t_4-t_3)
        B_lab_H_zeeman_eq.set_alpha(new_alpha)
        B_lab_H_drive_eq.set_alpha(new_alpha)
        B_lab_H_drive_eq_2.set_alpha(new_alpha)
    
    if t_5 < i <= t_6:
        new_alpha = (i-t_5)/(t_6-t_5)
        W_trans_arrow.set_alpha(new_alpha)
        W_trans_equation.set_alpha(new_alpha)
        omega_gt_omega_L.set_alpha(new_alpha)
    
    if t_7 < i <= t_8:
        new_alpha = (i-t_7)/(t_8-t_7)
        B_rot_H_text.set_alpha(new_alpha)
        B_rot_H_zeeman_text.set_alpha(new_alpha)
        B_rot_H_drive_text.set_alpha(new_alpha)
    
    if t_9 < i <= t_10:
        new_alpha = (i-t_9)/(t_10-t_9)
        B_rot_H_zeeman_eq.set_alpha(new_alpha)
        B_rot_H_drive_eq.set_alpha(new_alpha)
    
    if t_10 < i <= t_11:
        new_alpha = (i-t_10-1)/(t_11-t_10)
        B_rot_H_drive_eq_2.set_alpha(new_alpha)
    """
        
    return [ax for key, ax in ax_dict.items()] 

def init():
    return [ax for key, ax in ax_dict.items()]
    
ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50,
                              init_func=init, 
                              blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/zero_detuning_extended.{file_type}', fps = 20 )