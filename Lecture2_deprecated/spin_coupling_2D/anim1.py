from matplotlib import axis
from anim_base.spin2D import Spin, random_motion
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tqdm import tqdm
import numpy as np

N_time = 200

blue_spin_kwargs = {"N_t":N_time,
                "position" : np.array([0,0]).reshape(1,2),
                "layout_params" : {
                    "color":"blue",
                    "arrow_width": 7,
                    "ball_radius": 0,
                }
                }

red_spin_kwargs = {"N_t":N_time,
                "position" : random_motion(N_time) + np.array([0.5,0]).reshape(1,2),
                "rotation": 0,
                "layout_params" : {
                    "eq_distances": (0.3, 0.5),
                    "arrow_width": 7,
                    "ball_radius": 0,
                    "color": "darkcyan",
                },
                "mag": np.array([0,1])}
                


red_spin = Spin(**red_spin_kwargs)
blue_spin = Spin(**blue_spin_kwargs)

# mag = blue_spin.get_mag_field_from_other(red_spin)

fig, ax = plt.subplots()
fig.set_layout_engine("tight")
# ax.set(xlim=(-2, 1.5), ylim=(-1.5, 1.5)


def animate(t):
    ax.clear()
    ax.set(xlim=(-0.2, 1.1), ylim=(-0.35, 0.35))
    ax.set_axis_off()
    red_patches, red_lines = red_spin.get_plot_objects(t)
    

    for p in red_patches:
        ax.add_patch(p)
    for l in red_lines:
        ax.add_line(l)
    
    blue_patches, _ = blue_spin.get_plot_objects(0)
    
    for p in blue_patches:
        ax.add_patch(p)
    
    blue_mag = blue_spin.get_mag_arrow(red_spin, t, loc = [0, -0.3])
    ax.add_patch(blue_mag)

    return ax

ani = anim.FuncAnimation(fig, animate, tqdm(np.arange(N_time)), interval=50)
                            # init_func=init,
                            # blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/spin_coupling1.{file_type}', fps = 30 )


    