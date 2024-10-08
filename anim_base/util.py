"""Utility functions that does not produce animations"""

from scipy.optimize import curve_fit
import time
import numpy as np
import matplotlib.pyplot as plt
import shutil
from moviepy.editor import VideoFileClip

def rot_matrix(axis, angle):
    """
    Creates a rotation matrix

    Arguments
    ---------
        axis: 1-dimensional np.array 
           The rotation axis. Should have 3 components and length 1
        
        angle: float or 1-dimensional np.array
            The rotation angle(s) 
    
    Returns
    -------
        rot_matrix: np.array
            the rotation matrix/matrices. Has shape (3,3) if angle is a float.
            Has shape (N,3,3) if angle has N elements
            
        
    """
    # Formula from https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
    assert len(axis) == 3 and np.linalg.norm(axis) == 1, "axis has to be 3 dimensional vector of length 1"

    u_x, u_y, u_z = axis

    if np.isscalar(angle):
        R = np.zeros((1,3,3))
    else:
        R = np.zeros((len(angle), 3,3))
    
    sin = np.sin(angle)
    cos = np.cos(angle)

    R[:,0,0] = cos + u_x**2*(1 - cos)
    R[:,0,1] = u_x*u_y*(1 - cos) - u_z*sin
    R[:,0,2] = u_x*u_z*(1 - cos) + u_y * sin
    R[:,1,0] = u_x*u_y*(1 - cos) + u_z*sin
    R[:,1,1] = cos + u_y**2*(1 - cos)
    R[:,1,2] = u_y*u_z*(1 - cos) - u_x*sin
    R[:,2,0] = u_z*u_x*(1 - cos) - u_y * sin
    R[:,2,1] = u_z*u_y*(1 - cos) + u_x*sin
    R[:,2,2] = cos + u_z**2*(1 - cos)

    # print(R)
    if np.isscalar(angle):
        R = R.reshape(3,3)
    
    return R

def bloch_vector(theta, phi, r=1):
    """Function computing the bloch vector of a qubit parametrized
    as cos(theta/2) |0> + sin(theta/2)*exp(i*phi) |1>
    
   

    Arguments
    -----------
        theta: float or 1-dimensional np.array

        phi: float or 1-dimensional np.array

        r: float or 1-dimensional np.array, default = 1
    
    Returns
    -------
        If all arguments are scalars, a single vector is returned as an array with shape (3,) Otherwise, several vectors
        are returned in an array on form (N,3)
    """


    assert all([np.isscalar(inp) or len(inp.shape) == 1 for inp in (theta, r, phi)]), "All inputs must be scalar or have a shape (N,)"
    input_lens = [1 if np.isscalar(inp) else len(inp) for inp in (theta, r, phi)]
    max_len = max(input_lens)
    
    # Check that dimensions match
    for i in range(3):
        for j in range(i+1, 3):
            if not ((input_lens[i] == 1) or (input_lens[j] == 1) or (input_lens[i] == input_lens[j])):
                raise ValueError("If 2 or more inputs to this function (theta, phi, r) are not scalar, they must all match in dimension")
    
    # Build the bloch vectors
    vec = np.zeros((max_len, 3))
    vec[:,0] = np.sin(theta)*np.cos(phi)*r
    vec[:,1] = np.sin(theta)*np.sin(phi)*r
    vec[:,2] = np.cos(theta)*r

    if max_len == 1:
        vec = vec.flatten()

    return vec

def flatten_list(l):
    """Function which recursively turns a list of lists of .... of lists into a flat list"""

    if l == []:
        return l
    if isinstance(l[0], list):
        return flatten_list(l[0]) + flatten_list(l[1:])
    return l[:1] + flatten_list(l[1:])

def damped_cosine(t, A, omega, phi, tau):
    return A*np.exp(-t/tau)*np.cos(omega*t + phi)

def damped_sine(t, A, omega, phi, tau):
    return A*np.exp(-t/tau)*np.sin(omega*t + phi)

def fit_damped_cosine(y_data, t_data=None):
    """Function which fits a damped cosine to datapoints
    
    Arguments
    ----------
        y_data: 1-dimensional np.array
            Datapoints to fit
        t_data: None or np.array, default = None
            Input values to the function we want to fit. If None, an integer range with the same
            length as y_data is used
    
    Returns
    -------
        functional values: np.array
            The values of the fitted function, evaluated in the points from t_date
        popt: tuple of floats
            The fitted parameters (A, omega, phi, tau). See function damped_coine above for explanantion

    """

    # jacobian of the function
    def jac(t, A, omega, phi, tau):
        return np.array([damped_cosine(t, 1, omega, phi, tau),
                        -t*damped_sine(t, A, omega, phi, tau),
                        -damped_sine(t, A, omega, phi, tau),
                        t/tau**2*damped_cosine(t, A, omega, phi, tau)
                        ]).T


    if t_data is None:
        t_data = np.arange(y_data.size)
    
    tic = time.time()
    popt, pcov = curve_fit(damped_cosine, t_data, y_data, 
                    p0=[1, 1, 0, 1],
                    jac=jac
                    )
    toc = time.time()
    print('Time to fit: ', toc - tic)
    return damped_cosine(t_data, *popt), popt


def random_walk(N_t, 
                dim,  
                random_force_var = 0.1,
                k = 1, 
                c = 1,
                center = None):
    """Generates a random walk in dim dimensions. The components of the acceleration is sampled randomly around mean=0 with variance var.
    There is a spring term that pulls the walker towrds a defined center, 
    and a friction term that slows the momenta.
    
    Parameters
    ----------
        N_t : int
            Number of time steps
        dim : int
            Dimension of the random walk
        var : float (default = 0.005)
            Variance of the sampled acceleration. Higher values lead to more erratic motion.
        k : float (default = 1)
            Spring constant which defines the force pulling the alker towards the center. 
        c: float (default = 1)
            Friction constant which defines the force slowing down the walker.
        center : float or np.array of shape (dim,) (default = None)
            Target position of the walker. Defines where the spring force will 
             pull towards If None, the center is set to the origin.
    
    Returns
    -------
        xy : np.array of shape (N_t, dim)
            Array of positions of the walker at each time step.
    """

    # If no center, set to origin
    if center is None:
        center = np.zeros((N_t,dim))
    # If center is a scalar, set to a vector of that value
    elif np.isscalar(center):
        center = np.ones((N_t,dim))*center
    # If center is a vector, check that it has the correct shape
    elif isinstance(center, np.ndarray):
        assert center.shape in ( (dim,), (N_t,dim)), "Target must be a scalar or an array of shape (dim,) or (N_t,dim)"
        if center.shape == (dim,):
            center = np.tile(center, (N_t,1))
    else:
        raise ValueError("Target must be None, a scalar or a vector of shape (dim,)")

    # Allocate positions and momenta
    # The initial position is the same as the initial center
    # The initial momenta is zero
    pos = np.zeros((N_t,dim))
    pos[0,:] = center[0]
    momenta = np.zeros_like(pos)
    random_force = np.random.normal(0, random_force_var, size=pos.shape)

    for i in range(1,N_t):
        spring_force = - k * (pos[i-1] - center[i])
        friction_force = - c * momenta[i-1]
        acceleration = random_force[i] + spring_force + friction_force
        momenta[i] = momenta[i-1] + acceleration
        pos[i] = pos[i-1] + momenta[i]
    
    return pos

def cache_then_save_funcanimation(func_animation, filename, fps = 30, temp_path = "animations/test/temp" ):
    """Generating a full animation can be slow. This function saves the animation to a temporary file,
    and then renames it to the desired filename. By using this function, you can still inspect the old 
    version of the animation while the new one is being generated
    
    Parameters
    ----------
        func_animation : matplotlib.animation.FuncAnimation
            The animation to save
        filename : str
            The filename to save the final animation to
        fps : int (default = 30)
            The number of frames per second to save the animation at
        temp_path : str (default = "animations/test/temp")
            The path to save the temporary animation to. The extension is added automatically.

    Returns
    -------
        None
    """
    # Extract the extension from the filename and add it to the temp_path
    extension = filename.split('.')[-1]
    temp_path = temp_path + '.' + extension
    # Save the animation to a temporary file
    func_animation.save(temp_path, fps=fps)
    # Rename the temporary file to the desired filename
    shutil.move(temp_path, filename)

def mp4_to_gif(mp4_path, gif_path=None):
    """Converts an mp4 file to a gif file
    
    Parameters
    ----------
        mp4_path : str
            The path to the mp4 file to convert
        gif_path : str (default = None)
            The path to save the gif file to. If None, the gif file is saved in the same directory as the mp4 file, with the same name but a .gif extension

    Returns
    -------
        None
    """
    video_name, ext  = mp4_path.split('.')
    assert ext == 'mp4', "The input file must be an mp4 file"

    if gif_path is None:
        gif_path = video_name + '.gif'
    else:
        assert gif_path.split('.')[-1] == 'gif', "The output file must be a gif file"

    # loading video
    clip = VideoFileClip(mp4_path)
    # saving video clip as gif
    clip.write_gif(gif_path)

   

if __name__ == "__main__":
    mp4_path = "hahn_echo_color.mp4"
    mp4_to_gif(mp4_path)