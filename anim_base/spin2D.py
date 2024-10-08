import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle, FancyArrowPatch
from anim_base.util import rot_matrix
from tqdm import tqdm

class Spin2D(object):
    """
    A class to represent a 2D spin object that can be added to plots/animations

    Attributes
    ----------
        ax_plot : matplotlib.axes.Axes
            The axes object toF which the spin object will be added

        position : np.array of shape (2,)
            The position of the center spin in the plot ((x,y) - coordinates)

        rotation : float ( default = 0 )
            The rotation of the spin in radians
        
        mag : float ( default = 1 )
            The magnitude of the magnetic moment of the spin
        
        line_alpha : float ( default = 1 )
            The alpha value of the field lines
        
        ball_alpha : float ( default = 1 )
            The alpha value of the ball
        
        arrow_alpha : float ( default = 1 )
            The alpha value of the magnetic moment arrow
        
        layout_params : dict ( default = {} )
            A dictionary of layout parameters for the spin object.
            - spin_color : str ( default = "red" )
                The color of the spin object ( ball and arrow)
            - mag_line_color : str ( default = "red" )
                The color of the field lines
            - eq_distances : tuple of floats ( default = (0.2, 0.5) )
                The distance(s) from the center of the spin to the intersection of the field line(s) with 
                the x-axis ( assuming no rotation)
            - ball_radius : float ( default = 0.05 )
                The radius of the ball representing the spin
            - arrow_length : float ( default = 0.25 )
                The length of the arrow representing the magnetic moment
            - arrow_width : float ( default = 5 )
                The width of the arrow representing the magnetic moment
            - field_line_resolution : int ( default = 100 )
                The number of points per field line that's used to draw them
            
    Methods
    -------

        see docstrings for each method

    """
    def __init__(self, ax_plot, position, rotation = 0, mag = 1, 
                 ball_alpha = 1, arrow_alpha = 1, 
                 line_alpha = 1, layout_params = dict(), 
                 ):
        
        self.ax_plot = ax_plot
        self.position = position
        self.rotation = rotation
        self.mag = mag

        self.lp = self.get_layout_params(layout_params)

        self.line_alpha = line_alpha
        self.ball_alpha = ball_alpha
        self.arrow_alpha = arrow_alpha

        self.generate_plot_objects(update=False)
    
    def get_layout_params(self, layout_params):
        """
        Returns a dictionary of layout parameters for the spin object.

        Parameters
        ----------
            layout_params : dict
                A dictionary of layout parameters for the spin object.
        """

        def_params = {  "spin_color":"blue",
                        "mag_line_color":"red",
                        "eq_distances":(0.2, 0.5), 
                        "ball_radius": 0.05, 
                        "arrow_length": 0.25,
                        "arrow_width":5,
                        "arrow_mutation": 1,
                        "field_line_resolution":100,
                        "field_line_width":1,
                        
                        # "mag_arrow_scale": 0.35, 
                        # "mag_arrow_width": 0.1
                        }
        
        for key in layout_params:
            if key not in def_params:
                raise ValueError("Invalid layout parameter: ", key)
            else:
                def_params[key] = layout_params[key]

        return def_params

    def get_mag_dipole_vector(self):
        """Calculates the magnetic moment vector based on the 
        rotation and magnitude of the spin"""

        mag_dipole_vector = np.array([0,self.mag])
        if not np.isclose(self.rotation, 0):
            axis = np.array([0,0,1])
            rot_mat = rot_matrix(axis, self.rotation)
            mag_dipole_vector = rot_mat[:2,:2] @ mag_dipole_vector

        return mag_dipole_vector

    def get_field_line_array(self):
        """Calculates the array representing the datapoints of the field lines based
        on the layout parameters 'field_line_resolution' and 'eq_distances', as well as the 
        spin's position and rotation

        Returns
        -------
            field_line_array : np.array of shape (2*eq_distances, 2, resolution)

                The array representing the datapoints of the field lines. The first axis is 
                the different field lines. It is 2*eq_distances since we have one field line
                on each side of the spin per distance. The second axis is the x and y positions
                of the datapoints. The third axis is the datapoints themselves.

        """
        resolution = self.lp["field_line_resolution"]

        # This is the parametrization of a field line emanating from a magnetic dipole,
        #  with the dipole at the origin, the magnetic moment pointing in the y direction,
        # and unit distance between the dipole and the intersection of the field line with
        #  the xy plane.
        # x = sin(theta)**3, 
        # y = sin(theta)**2 * cos(theta)
        # 0 <= theta <= pi

        phi = np.linspace(0, np.pi, resolution)
        x = np.sin(phi)**3 
        y = np.sin(phi)**2 * np.cos(phi) 
        xy = np.array([x, y]).reshape(1,2,resolution)
    

        # Find lines on the right side of the spin, Multiply the xy array with eq_distances
        # to create an array of shape (eq_distances,2,resolution) 
        field_lines_right = np.array(self.lp["eq_distances"]).reshape(-1,1,1) * xy

        # Find lines on the left side of the spin, Multiply x values with - 1
        # to create another array of shape (eq_distances,2,resolution)
        field_lines_left = (field_lines_right * np.array([-1,1]).reshape(1,2,1))

        # Concatenate the two arrays to get an array of shape (2*eq_distances,2,resolution)
        field_line_array = np.concatenate([field_lines_left, field_lines_right],axis = 0)

        # Rotate field lines
        if not np.isclose(self.rotation, 0):
            axis = np.array([0,0,1])
            rot_mat = rot_matrix(axis, self.rotation)[:2,:2]
            field_line_array = np.einsum("ij, kjl -> kil", rot_mat, field_line_array)
        
        # Translate field lines
        if not np.allclose(self.position, 0):
            field_line_array = field_line_array + self.position.reshape(1,2,1)
        
        return field_line_array

    def generate_arrow_patch(self, update = True):
        """Generates/updates a FancyArrowPatch object representing the magnetic moment of the spin.
        If generated, it also adds it to self.ax_plot and save it as an attribute of the spin object.

        Parameters 
        ----------
            update : bool (default = True)
                If True, self.arrow is updated 
                If False, a new FancyArrowPatch object is created, saved as self.arrow and added to self.ax_plot
        
        """
        dxdy = np.array([0, self.lp["arrow_length"]])
        if not np.isclose( self.rotation, 0):
            axis = np.array([0,0,1])
            rot_mat = rot_matrix(axis, self.rotation)[:2,:2]
            dxdy  = np.einsum("ij, j -> i", rot_mat, dxdy)
        
        x0y0 = self.position - dxdy/2
        
        if update:
            try:
                self.arrow.set_positions(x0y0, x0y0 + dxdy)
                # self.arrow.set_dx_dy(dxdy)
                self.arrow.set_alpha(self.arrow_alpha)
            except AttributeError:
                raise AttributeError("Arrow patch has not been created yet. Please set update = False")
        else:
            self.arrow = FancyArrowPatch(
                x0y0, x0y0 + dxdy,
                arrowstyle='-|>', 
                mutation_scale=self.lp["arrow_mutation"],
                lw = self.lp["arrow_width"], 
                color = self.lp["spin_color"],
                alpha = self.arrow_alpha)

            self.ax_plot.add_patch(self.arrow)
    
    def generate_ball_patch(self, update = True):
        """Generates/updates a Circle object representing the particle of the spin.
        If generated, it also adds it to self.ax_plot and save it as an attribute of the spin object.

        Parameters 
        ----------
            update : bool (default = True)
                If True, self.ball is updated 
                If False, a new FancyArrowPatch object is created, saved as self.ball and added to self.ax_plot
        
        """
        if update:
            try:
                self.ball.center = self.position
                self.ball.set_alpha(self.ball_alpha)
            except AttributeError:
                raise AttributeError("Ball patch has not been created yet. Please set update = False")
        else:
            self.ball = Circle(self.position, self.lp["ball_radius"], color = self.lp["spin_color"], alpha = self.ball_alpha)
            self.ax_plot.add_patch(self.ball)
    
    def generate_line2d_objects(self, update = True):
        
        """Generates/updates a list of Line2D objects representing the magnetic field of the spin.
        If generated, it also adds them to self.ax_plot and saves the list as an attribute of the spin object.

        Parameters 
        ----------
            update : bool (default = True)
                If True, self.lines is updated 
                If False, a new list of Line2D objects is created, saved as self.lines and added to self.ax_plot
        
        """
        field_line_array = self.get_field_line_array()
        if update:
            for i in range(field_line_array.shape[0]):
                self.lines[i].set_data(field_line_array[i,0,:], field_line_array[i,1,:])
                self.lines[i].set_alpha(self.line_alpha)
        else:
            self.lines = []
            for i in range(field_line_array.shape[0]):
                self.lines.append(matplotlib.lines.Line2D(field_line_array[i,0,:], 
                                                          field_line_array[i,1,:], 
                                                          color = self.lp["mag_line_color"], 
                                                          alpha = self.line_alpha,
                                                          lw = self.lp["field_line_width"]))

            for line in self.lines:
                self.ax_plot.add_line(line)

    def generate_plot_objects(self, update = True):
        """Calls all the methods that generate or update plot objects
        
        Parameters
        ----------
            update : bool (default = True)
                If True, the plot objects are updated with the current position, rotation and style parameters of the spin.
                If False, new plot objects are created and saved as attributes of the spin object.
        """
        self.generate_arrow_patch(update)
        self.generate_ball_patch(update)
        self.generate_line2d_objects(update)
        
    
    def get_mag_field_at_r1(self, r1):
        """Computes the magnetic field at a position r1, due to the magnetic moment of the spin
        
        Parameters
        ----------
            r1 : array-like
                Position vector of the point where the magnetic field is to be computed

        Returns
        -------
            mag_field : np.array of shape (2,)
                Magnetic field vector at r1
        
        """

        r0 = self.position
        r = r1 - r0
        r_abs = np.linalg.norm(r)

        if np.isclose(r_abs, 0):
            print("r_abs = ", r_abs)
            raise ZeroDivisionError("Cant avaluate magnetic field with r_abs = 0")
        
        mag_dipole_vector = self.get_mag_dipole_vector()

        mag_field = 3*r*np.dot(mag_dipole_vector, r)/(r_abs**5) - mag_dipole_vector/(r_abs**3)
        return mag_field
    
    def get_mag_field_from_other(self, other):
        """Computes the magnetic field at the position of the spin, due to the magnetic moment of another spin object"""

        assert isinstance(other, Spin2D), "other must be a Spin2D object"

        mag = other.get_mag_field_at_r1(self.position)
        return mag
    
    # def get_mag_arrow(self, other, t=0, loc = [0,-0.5], only_x = True):
    #     mag = self.get_mag_field_from_other(other, t) *self.lp["mag_arrow_scale"]
    #     if self.position.size == 2:
    #         t = 0

    #     if only_x:
    #         mag_arrow = Arrow( (self.position[t,0] - mag[0]*self.lp["mag_arrow_scale"]/2 + loc[0]),
    #                            self.position[t,1] + loc[1], 
    #                            mag[0]*self.lp["mag_arrow_scale"],
    #                            0, 
    #                            width = self.lp["mag_arrow_width"], 
    #                            color = self.lp["color"])
    #     return mag_arrow

if __name__ == "__main__":
    N_t = 100
    # xy = random_vibration(N_t)