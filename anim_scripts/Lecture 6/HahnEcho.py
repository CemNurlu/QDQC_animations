from matplotlib import pyplot as plt
from matplotlib import animation as anim
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import qutip
import numpy as np
from anim_base import  cache_then_save_funcanimation, file_type
size = 10
fig, axes = plt.subplots(2,1, figsize=(size*0.75, size), gridspec_kw={'height_ratios': [6, 1]})
axes[0].remove()
ax_sphere = fig.add_subplot(2,1,1, projection="3d", azim=-60, elev=30)

N_time = 1000

t_ax = axes[1]
t = np.linspace(0,N_time, N_time)
t0 = 0
t1 = int(N_time*0.10)
t2 = int(N_time*0.13)
t3 = int(N_time*0.40)
t4 = int(N_time*0.46)
t5 = int(N_time*0.73)
t6 = int(N_time*1)


pulse = np.zeros(len(t))
pulse[t1:t2] = 1
t_ax.text((t1+t2)/2, 1.2, r"$ R_x( \pi \, / \, 2) $",  ha = "center", size = "large")
pulse[t3:t4] = 1
t_ax.text((t3+t4)/2, 1.2, r"$ R_x ( \pi ) $",  ha = "center", size = "large")

mu_gauss = (t6+t4)/2 
sigma_gauss = (t6-t4)/30000
t_gauss = t[t4:t6]
exponent = - (t_gauss-mu_gauss)**2/2*sigma_gauss**2 
gauss = 0.8*np.exp(exponent)

t_ax.plot(t_gauss, gauss, c= "blue")
t_ax.text(mu_gauss, 1.2, "ECHO",  ha = "center", size = "large")



t_ax.plot(t,pulse)
t_line = axes[1].axvline(0,0,1.1, c = "red")
# axes[0].

axes[1].axis("off")


sphere = qutip.Bloch(axes=ax_sphere)
sphere.point_marker = ["o"]
sphere.point_color = ["r"]
sphere.vector_color = ['red', 'blue', 'yellow', 'blue', 'red']
# sphere.vector_color = ["red"]
sphere.point_marker = ["o"]
sphere.point_color = ["r"]
sphere.vector_width = 6
sphere.make_sphere()


if True:
   vectors = np.zeros((5,N_time,3))

   vectors[:,:t1,2] = 1

   vectors[:, t1:t2, 2] = np.cos(np.linspace(0, np.pi/2 , t2-t1))
   vectors[:, t1:t2, 1] = np.sin(np.linspace(0, np.pi/2 , t2-t1))

   vectors[0,t2:t3,0] = np.sin(np.linspace(0, -np.pi*4/5, t3-t2))
   vectors[0,t2:t3,1] = np.cos(np.linspace(0, np.pi*4/5, t3-t2))

   vectors[1,t2:t3,0] = np.sin(np.linspace(0, -np.pi*2/5, t3-t2))
   vectors[1,t2:t3,1] = np.cos(np.linspace(0, np.pi*2/5, t3-t2))

   vectors[2,t2:t3,1] = 1

   vectors[3,t2:t3,0] = np.sin(np.linspace(0, np.pi*2/5, t3-t2))
   vectors[3,t2:t3,1] = np.cos(np.linspace(0, np.pi*2/5, t3-t2))

   vectors[4,t2:t3,0] = np.sin(np.linspace(0, np.pi*4/5, t3-t2))
   vectors[4,t2:t3,1] = np.cos(np.linspace(0, np.pi*4/5, t3-t2))

   rot180_time = t4 - t3
   rot180 = np.zeros((rot180_time, 3, 3))
   theta = np.linspace(0, np.pi, rot180_time)
   rot180[:,0,0] = 1
   rot180[:,1,1] = np.cos(theta)
   rot180[:,2,2] = np.cos(theta)
   rot180[:,1,2] = - np.sin(theta)
   rot180[:,2,1] = np.sin(theta)

   non_rot_vecs = vectors[:,t3-1,:]
   # print(non_rot_vecs)
   rot_vecs = np.tensordot(rot180, non_rot_vecs, axes = ([2],[1]))
   rot_vecs = np.swapaxes(rot_vecs, 0, 2)
   rot_vecs = np.swapaxes(rot_vecs, 1, 2)
   # print(rot_vecs)
   # print(rot180)
   vectors[:, t3:t4, :] = rot_vecs

   vectors[0,t4:t5,0] = - np.sin(np.linspace(np.pi/5, np.pi, t5-t4))
   vectors[0,t4:t5,1] = np.cos(np.linspace(np.pi/5, np.pi, t5-t4))

   vectors[1,t4:t5,0] = - np.sin(np.linspace(np.pi*3/5, np.pi, t5-t4))
   vectors[1,t4:t5,1] = np.cos(np.linspace(np.pi*3/5, np.pi, t5-t4))

   vectors[2,t4:t5,1] = -1

   vectors[3,t4:t5,0] = np.sin(np.linspace(np.pi*3/5, np.pi, t5-t4))
   vectors[3,t4:t5,1] = np.cos(np.linspace(np.pi*3/5, np.pi, t5-t4))

   vectors[4,t4:t5,0] = np.sin(np.linspace(np.pi/5, np.pi, t5-t4))
   vectors[4,t4:t5,1] = np.cos(np.linspace(np.pi/5, np.pi, t5-t4))

   #  --------------------------------------------------------------

   vectors[0,t5:t6,0] = vectors[0,t4:t5,0][::-1]
   vectors[0,t5:t6,1] = vectors[0,t4:t5:,1][::-1]

   vectors[1,t5:t6,0] = vectors[1,t4:t5,0][::-1]
   vectors[1,t5:t6,1] = vectors[1,t4:t5,1][::-1]

   vectors[2,t5:t6,1] = -1

   vectors[3,t5:t6,0] = vectors[3,t4:t5,0][::-1]
   vectors[3,t5:t6,1] = vectors[3,t4:t5,1][::-1]

   vectors[4,t5:t6,0] = vectors[4,t4:t5,0][::-1]
   vectors[4,t5:t6,1] = vectors[4,t4:t5,1][::-1]

def animate(i):
   sphere.vectors = []
   sphere.add_vectors(vectors[:,i,:])
   t_line.set_xdata([i,i])
   sphere.make_sphere()
   if i%(N_time//10) == 0:
      print(f"Animation {(100*i)//N_time} % done")

   return axes

def init():
    
   sphere.make_sphere()
   return axes

ani = anim.FuncAnimation(fig, animate, np.arange(N_time), interval=50,
                              init_func=init, blit=False, repeat=False)
cache_then_save_funcanimation(ani, f'hahn_echo_time_dirac.{file_type}', fps = 40 )