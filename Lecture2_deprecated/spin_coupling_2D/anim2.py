from matplotlib import axis
from anim_base.spin2D import Spin, random_motion
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tqdm import tqdm
import numpy as np

N_time = 200

blue_spin_kwargs = {"N_t":N_time,
                "position" : random_motion(N_time), # + np.array([0.5,0]).reshape(1,2),
                "rotation": 0,
                "layout_params" : {
                    "eq_distances": (0.3, 0.5),
                    "color":"blue",
                    "arrow_width": 7,
                    "ball_radius": 0,
                },
                "mag": np.array([0,1])}
                

red_spin_kwargs = {"N_t":N_time,
                "position" : np.array([0.5,0]).reshape(1,2),
                "layout_params" : {
                    "color":"red",
                    "arrow_width": 7,
                    "ball_radius": 0,
                    "color": "darkcyan"
                }
                }

blue_spin = Spin(**blue_spin_kwargs)
red_spin = Spin(**red_spin_kwargs)

fig, ax = plt.subplots()
fig.set_layout_engine("tight")

def animate(t):
    ax.clear()
    ax.set(xlim=(-0.6, 0.7), ylim=(-0.35, 0.35))
    ax.set_axis_off()
    blue_patches, blue_lines = blue_spin.get_plot_objects(t)

    for p in blue_patches:
        ax.add_patch(p)
    for l in blue_lines:
        ax.add_line(l)
    
    red_patches, _ = red_spin.get_plot_objects(0)
    
    for p in red_patches:
        ax.add_patch(p)
    
    red_mag = red_spin.get_mag_arrow(blue_spin, t, loc = [0, -0.3])
    ax.add_patch(red_mag)

    return ax


ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50)
                            # init_func=init,
                            # blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/spin_coupling2.{file_type}', fps = 30 )


    