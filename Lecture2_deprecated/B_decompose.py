from matplotlib import pyplot as plt
from matplotlib import animation as anim
import qutip
import numpy as np
from Scripts.util.bloch import bloch_vector, rot_matrix
from tqdm import tqdm
import matplotlib.patches as mpatches
# from matplotlib.gridspec import GridSpec




# ax_plot.add_patch(arrow)
   

size = 10

fig, ax = plt.subplots(figsize=(size*2, size))

ax.axis("off")
ax.set_xlim(-2,4)
ax.set_ylim(-1.5,3)


N_time = 300
t0 = 100
t1 = 150
# t2 = 75

t = np.arange(N_time)
phi = np.linspace(0, 6*np.pi, N_time)

big_Bx = np.cos(phi)

small_B1 = np.concatenate(
   (np.cos(phi).reshape(-1,1)/2, np.sin(phi).reshape(-1,1)/2),
   axis = 1
   )

small_B2 = np.concatenate(
   (np.cos(phi).reshape(-1,1)/2, - np.sin(phi).reshape(-1,1)/2), 
   axis = 1
   )

small_arrows = [
   mpatches.Arrow(0, 0, 1, 0, width=0.7, color = "blue"),
   mpatches.Arrow(0, 0, 0.5, 0, width=0.7, color = "blue")

]

big_arrow = mpatches.Arrow(0, 0, 1, 0, width=0.7, color = "red")
ax.add_patch(big_arrow)

red_str = r"$B_{drive} = B_0 \mathrm{cos}( \omega t ) \hat{x}$"
blue_str_a = r"$B_{drive} = \frac{B_0}{2} (  (\mathrm{cos}( \omega t ) \hat{x} + \mathrm{sin}( \omega t ) \hat{y})$"
blue_str_b = r"$ + \;\; (\mathrm{cos}( \omega t ) \hat{x} - \mathrm{sin}( \omega t ) \hat{y} ) )$"

blue_str = r"$ \, \, = \,\, \frac{B_0}{2} (  (\mathrm{cos}( \omega t ) \hat{x} + \mathrm{sin}( \omega t ) \hat{y}) + (\mathrm{cos}( \omega t ) \hat{x} - \mathrm{sin}( \omega t ) \hat{y} ) )$"


def init():   
   return ax

def animate(i):
   # big_arrow.remove()
   # ax.patches = []
   # big_arrow.remove()
   ax.clear()
   ax.axis("off")
   ax.set_xlim(-1, 1)
   ax.set_ylim(-0.8 , 1.1)
   text1 = ax.text(-0.5, 1, red_str, c = "red", fontsize = 25, ha = "center")
   
   big_arrow = mpatches.Arrow(0, 0, big_Bx[i], 0, width=0.7, color = "red")
   ax.add_patch(big_arrow)
   if i < t0:
      pass
   elif i < t1:
      alpha = (i - t0) / (t1-t0)

      # text2a = ax.text(2, 1, blue_str_a, c = "blue", fontsize = 25, ha = "right", alpha = alpha)
      # text2b = ax.text(2, 0.8, blue_str_b, c = "blue", fontsize = 25, ha = "right", alpha = alpha)
      text2 = ax.text(-0.3, 1, blue_str, c = "blue", fontsize = 25, ha = "left", alpha = alpha)

      small_arrow1 = mpatches.Arrow(0, 0, small_B1[i,0], small_B1[i,1], width=0.3, color = "blue", alpha = alpha)
      small_arrow2 = mpatches.Arrow(0, 0, small_B2[i,0], small_B2[i,1], width=0.3, color = "blue", alpha = alpha)
      ax.add_patch(small_arrow1)
      ax.add_patch(small_arrow2)
   else:
      # text2 = ax.text(2, 2, blue_str, c = "blue", fontsize = 25, ha = "center")

      # text2a = ax.text(2, 1, blue_str_a, c = "blue", fontsize = 25, ha = "right")
      # text2b = ax.text(2, 0.8, blue_str_b, c = "blue", fontsize = 25, ha = "right")

      text2 = ax.text(-0.3, 1, blue_str, c = "blue", fontsize = 25, ha = "left")

      small_arrow1 = mpatches.Arrow(0, 0, small_B1[i,0], small_B1[i,1], width=0.3, color = "blue")
      small_arrow2 = mpatches.Arrow(0, 0, small_B2[i,0], small_B2[i,1], width=0.3, color = "blue")
      ax.add_patch(small_arrow1)
      ax.add_patch(small_arrow2)

   return ax



ani = anim.FuncAnimation(fig, animate, tqdm(t), interval=50,
                              init_func=init, blit=False, repeat=False)

cache_then_save_funcanimation(ani, f'animations/test/B_decompose.{file_type}', fps = 20 )