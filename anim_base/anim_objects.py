import qutip
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import warnings 
from abc import ABC
from .util import flatten_list
from .config import math_fontfamily



def init_bloch_sphere(ax_sphere, **sphere_kwargs):
    """'
    Create a qutip.Bloch bloch sphere object and set its attributes.

    Arguments
    ---------
        ax_sphere: matplotlib.axes.Axes
            The axes object to use for the bloch sphere. Has to be a 3D axes object.
        sphere_kwargs: 
            Keyword arguments to pass to the qutip.Bloch object.
    
    Returns
    -------
        sphere: qutip.Bloch
            The bloch sphere object

    """
    
    sphere = qutip.Bloch(axes=ax_sphere)
    for key in sphere_kwargs:
        if key not in ("point_marker", "point_color", "vector_color", "vector_width", 
                    "vector_alpha", "sphere_alpha", "frame_alpha", "font_alpha", 
                    "ax_eq_alpha", "xlabel", "ylabel", "zlabel"):
            raise ValueError("Invalid keyword argument: {}".format(key)) 
        setattr(sphere, key, sphere_kwargs[key])

    return sphere

def prepare_bloch_mosaic(mosaic, size, bloch_kwargs, gridspec_kw=None):
    """Prepare a matplotlib subplot_mosaic for bloch spheres.
    
    Arguments
    ---------
        mosaic: list of lists of strings.
            See matplotlib.pyplot.subplot_mosaic for details on the mosaic argument.
            For the axes of the mosaic that have a string starting with 'bloch' we will 
            create a 3D axes object and a qutip.Bloch object. If the string does not start
            with 'bloch', we will create a normal 2D axes object.
        size: tuple of two ints
            size of the figure
        bloch_kwargs: list of dicts
            List of dictionaries containing keyword arguments to pass to each qutip.Bloch object
            If only one dictionary is given, it is used for all bloch spheres
        gridspec_kw: dict, default = None
            Keyword arguments to pass to the gridspec_kw argument of matplotlib.pyplot.subplot_mosaic
        
    Returns
    -------
        fig: matplotlib.figure.Figure
            The figure object
        axes: dict
            Dictionary containing the axes objects of the figure. keys are the strings in the mosaic argument.
        sphere_dict: dict
            Dictionary containing the qutip.Bloch objects. keys are the strings in the mosaic argument.    
    """


    # print(mosaic)
    assert len(set([len(row) for row in mosaic])) == 1, "Mosaic must be rectangular"

    n_cols = len(mosaic[0])
    n_rows = len(mosaic)
    
    # Default gridspec_kw
    if gridspec_kw is None:
        gridspec_kw = {"width_ratios": [1 for _ in range(n_cols)],
                    "height_ratios": [1 for _ in range(n_rows)]}

    n_blochs = 0
    n_plots = 0

    all_mosaic_keys = flatten_list(mosaic)
    bloch_subplot_indices = {}
    # This loop maps the bloch mosaic keys to the indices of each subplot
    # Probably a better way to do this
    for subplot_index, key in enumerate(all_mosaic_keys, 1):
        if key.startswith("bloch"):
            if key not in bloch_subplot_indices:
                bloch_subplot_indices[key] = subplot_index
                n_blochs += 1
    
    # Use same bloch_kwargs for all bloch spheres if a single dict is given
    if isinstance(bloch_kwargs, dict):
        bloch_kwargs = [bloch_kwargs for _ in range(n_blochs)]
    # If more than one dict is given, make sure the number of dicts matches the number of bloch spheres
    elif isinstance(bloch_kwargs, list):
        assert len(bloch_kwargs) == n_blochs, "Number of bloch_kwargs must match number of bloch spheres"
    else:
        raise ValueError("bloch_kwargs must be a dict or a list of dicts")

    # Create the figure and axes
    fig, ax_dict = plt.subplot_mosaic(mosaic,
                              figsize=size, layout="constrained", gridspec_kw=gridspec_kw)


    # Create the bloch spheres
    # For each bloch sphere, we remove the 2D axes object and create a 3D axes object
    # We then create a qutip.Bloch object and set its attributes
    sphere_dict = {}
    b_i = 0
    for ax_key in ax_dict.keys():
        if ax_key.startswith("bloch"):
            ax_dict[ax_key].remove()
            bloch_subplot_index = bloch_subplot_indices[ax_key]
            ax_dict[ax_key] = fig.add_subplot(n_rows, n_cols, bloch_subplot_index, projection="3d", azim=-60, elev=30)            
            ax_dict[ax_key].set_axis_off()
            sphere = init_bloch_sphere(ax_dict[ax_key], **bloch_kwargs[b_i])
            sphere_dict[ax_key] = sphere
            b_i += 1
    
    return fig, ax_dict, sphere_dict

class PrettyObject(ABC):
    """Abstract base class for objects that can be addad to a matplotlib plot
    This should not be used directly, but instead inherited from 
    """

    def __init__(self, ax_plot, x_pos, y_pos):
        self._ax_plot = ax_plot
        self._x_pos = x_pos
        self._y_pos = y_pos

    @property
    def ax_plot(self):
        return self._ax_plot
    
    @ax_plot.setter
    def ax_plot(self, ax_plot):
        raise AttributeError("ax_plot is not changeable, create a new PrettyAxis object instead")
    
    @property
    def x_pos(self):
        return self._x_pos
    
    @x_pos.setter
    def x_pos(self, x_pos):
        raise AttributeError("x_pos is not changeable, use update_x_y_pos instead")

    @property
    def y_pos(self):
        return self._y_pos
    
    @y_pos.setter
    def y_pos(self, y_pos):
        raise AttributeError("y_pos is not changeable, use update_x_y_pos instead")
  
class PrettyAxis(PrettyObject):
    """Class for creating a pretty axis object
    
    Methods
    -------
    Only use the methods listed here!!!
    See their individual docstrings for more information

        add_line : Adds a line to the axis

        update_line : Updates a line on the axis

        add_label : Adds a label to the axis

        update_x_y_pos : Updates the x_pos and y_pos and changes the lines and labels accordingly    
    """
    def __init__(self, ax_plot, x_pos, y_pos, data_x_lim = None, data_y_lim = None, c= "black", alpha=1, axis_ls = "-"):
        
        """
        Parameters
        ----------
            ax_plot: matplotlib.axes.Axes
                The axes object to add the axis to
            x_pos: tuple of 3 floats
                Position of the x-axis. The first two elements are x-coordinates of the start and end of the axis, the third is the y-position
            y_pos: tuple of 3 floats
                Position of the y-axis. The first two elements are y-coordinates of the start and end of the axis, the third is the x-position
            data_x_lim: tuple of 2 floats, default = None
                The x-limits of the data we plot o the axis. If None, the x-limits are set to the same as x_pos[:2]
            data_y_lim: tuple of 2 floats, default = None
                The y-limits of the data we plot o the axis. If None, the y-limits are set to the same as y_pos[:2]
            c: str, default = "black"
                Color of the axis
            alpha: float, default = 1
                Opaqueness of the axis. 0 is transparent, 1 is opaque
            axis_ls: str, default = "-"
                Line style of the axis. See matplotlib.pyplot.line2D linestyle for options
        """


        super().__init__(ax_plot, x_pos, y_pos)
        self._set_data_lims(data_x_lim, data_y_lim)
        self._c = c
        self._alpha = alpha
        self._ls = axis_ls
        self._x_axis, self._y_axis = self._get_axis()

        self._plot_lines = {}
        self.labels = {}

    def _get_axis(self):
        x_axis = self.ax_plot.plot(self.x_pos[:2], (self.x_pos[2], self.x_pos[2]), c = self.c, alpha = self.alpha, ls = self.ls, lw = 1)
        y_axis = self.ax_plot.plot((self.y_pos[2], self.y_pos[2]), self.y_pos[:2], c = self.c, alpha = self.alpha, ls = self.ls, lw = 1)
        
        return x_axis[0], y_axis[0]
    
    def _set_data_lims(self, data_x_lim, data_y_lim):
        if data_x_lim is not None:
            self._data_x_lim = data_x_lim
        else:
            self._data_x_lim = self._x_pos[:2]
        if data_y_lim is not None:
            self._data_y_lim = data_y_lim
        else:
            self._data_y_lim = self._y_pos[:2]

    def add_line(self, key, x, y, c, linestyle = "-", alpha = 1, lw = 3.5):
        """Adds a line to the axis
        
        Parameters
        ----------
            key: str
                String to identify the line. All lines must have a unique key
            x: array-like
                x-coordinates of the line. All values must be between data_x_lim[0] and data_x_lim[1]
            y: array-like
                y-coordinates of the line. All values must be between data_y_lim[0] and data_y_lim[1]
            c: str 
                Color of the line. See matplotlib colors for options
            linestyle: str, default = "-"
                Line style of the line. See matplotlib.pyplot.line2D linestyle for options
            alpha: float, default = 1
                Opaqueness of the line. 0 is transparent, 1 is opaque
            lw: float, default = 3.5
                Line width of the line        
        """
        assert not (key in self._plot_lines), "Key already exists, use update_line instead or choose a different key"

        transformed_x = self._transform_x(x)
        transformed_y = self._transform_y(y)
        new_line = self.ax_plot.plot(transformed_x, transformed_y, linestyle, c = c, alpha = alpha, lw = lw)


        self._plot_lines[key] = new_line[0]

    def update_line(self, key, x, y):
        """Updates a line on the axis

        Parameters
        ----------
            key: str
                String to identify the line. Use the same key as when the line was added
            x: array-like
                x-coordinates of the line. All values must be between data_x_lim[0] and data_x_lim[1]
            y: array-like
                y-coordinates of the line. All values must be between data_y_lim[0] and data_y_lim[1]
            
        """
        transformed_x = self._transform_x(x)
        transformed_y = self._transform_y(y)
        self._plot_lines[key].set_data(transformed_x, transformed_y)
    
    def add_label(self, label, axis, size = 14):
        """Adds a label to the axis

        Parameters
        ----------
            label: str
                The label to add
            axis: str
                Which axis to add the label to. Must be "x" or "y"
            size: int, default = 14
                Font size of the label
        """


        if axis == "x":
            text = self.ax_plot.text(self.x_pos[1], self.x_pos[2], label, size = size, color = self.c, ha = "left", va = "top", alpha = self.alpha, math_fontfamily = math_fontfamily)
            self.labels[axis] = text
        elif axis == "y":
            text = self.ax_plot.text(self.y_pos[2], self.y_pos[1], label, size = size, color = self.c, ha = "right", va = "bottom", alpha = self.alpha, math_fontfamily = math_fontfamily)
            self.labels[axis] = text
        else:
            raise ValueError("axis must be 'x' or 'y'")

    def update_x_y_pos(self, x_pos = None, y_pos = None):
        """Updates the position of the axis on the ax object self.ax_plot

        Parameters
        ----------
            x_pos: array-like, default = None
                New x-axis position. Must be of length 3. See class docstring for more info
            y_pos: array-like, default = None
                New y-axis position. Must be of length 3. See class docstring for more info
        """
    

        if x_pos is None and y_pos is None:
            raise ValueError("x_pos and y_pos cannot both be None")
        
        x_y_lines_original_data = self._get_lines_original_data()

        if x_pos is not None:
            self._x_pos = x_pos
            self._x_axis.set_data(self.x_pos[:2], (self.x_pos[2], self.x_pos[2]))
        if y_pos is not None:
            self._y_pos = y_pos
            self._y_axis.set_data((self.y_pos[2], self.y_pos[2]), self.y_pos[:2])
        
        if self.x_pos[0] != self.y_pos[2]:
            warnings.warn("x_pos[0] != y_pos[2], which means that y_axis is not in 0")
        
        
        for key, (x_data, y_data)  in x_y_lines_original_data.items():
            self.update_line(key, x_data, y_data)
        
        for key, lab in self.labels.items():
            if key == 'x':
                lab.set_position((self.x_pos[1], self.x_pos[2]))
            elif key == 'y':
                lab.set_position((self.y_pos[2], self.y_pos[1]))

    def _transform_x(self, x):
        if np.any(x < self.data_x_lim[0]) or np.any(x > self.data_x_lim[1]):
            raise ValueError("x-values must be within the data_x_lim")
        return (x - self.data_x_lim[0]) * (self.x_pos[1] - self.x_pos[0]) / (self.data_x_lim[1] - self.data_x_lim[0])  + self.x_pos[0]

    def _transform_y(self, y):
        if np.any(y < self.data_y_lim[0]) or np.any(y > self.data_y_lim[1]):
            raise ValueError("y-values must be within the data_y_lim")
        return (y - self.data_y_lim[0]) * (self.y_pos[1] - self.y_pos[0]) / (self.data_y_lim[1] - self.data_y_lim[0])  + self.y_pos[0]
    
    def _inverse_transform_x(self, x):
        """Transform from the coordinates of the global plot back to the original data coordinates."""
        return (x - self.x_pos[0]) * (self.data_x_lim[1] - self.data_x_lim[0]) / (self.x_pos[1] - self.x_pos[0])  + self.data_x_lim[0]
    
    def _inverse_transform_y(self, y):
        """Transform from the coordinates of the global plot back to the original data coordinates."""
        return (y - self.y_pos[0]) * (self.data_y_lim[1] - self.data_y_lim[0]) / (self.y_pos[1] - self.y_pos[0])  + self.data_y_lim[0]

    def _get_lines_original_data(self):
        """Compute the orginal data of the lines in the plot by back transforming the data in the plot to the original data."""
        x_y_lines_original_data = {}
        for key, line in self._plot_lines.items():
            x_data, y_data = self._inverse_transform_x(line.get_xdata()), self._inverse_transform_y(line.get_ydata())
            x_y_lines_original_data[key] = (x_data, y_data)
        
        return x_y_lines_original_data

    @property
    def data_x_lim(self):
        return self._data_x_lim
    
    @data_x_lim.setter
    def data_x_lim(self, data_x_lim):
        raise AttributeError("data_x_lim is not changeable, create a new PrettyAxis object instead")
    
    @property
    def data_y_lim(self):
        return self._data_y_lim
    
    @data_y_lim.setter
    def data_y_lim(self, data_y_lim):
        raise AttributeError("data_y_lim is not changeable, create a new PrettyAxis object instead")
    
    
    @property
    def c(self):
        return self._c
    
    @c.setter
    def c(self, c):
        self._c = c
        self._x_axis.set_color(c)
        self._y_axis.set_color(c)
        for lab in self.labels.values():
            lab.set_color(c)
        
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        self._x_axis.set_alpha(alpha)
        self._y_axis.set_alpha(alpha)

        for line in self._plot_lines.values():
            line.set_alpha(alpha)

        for lab in self.labels.values():
            lab.set_alpha(alpha)
    
    @property
    def ls(self):
        return self._ls
    
    @ls.setter
    def ls(self, ls):
        self._ls = ls
        self._x_axis.set_linestyle(ls)
        self._y_axis.set_linestyle(ls)
    
    @property
    def x_axis(self):
        return self._x_axis
    
    @x_axis.setter
    def x_axis(self, x_axis):
        raise AttributeError("x_axis is not changeable, create a new PrettyAxis object instead")
    
    @property
    def y_axis(self):
        return self._y_axis
    
    @y_axis.setter
    def y_axis(self, y_axis):
        raise AttributeError("y_axis is not changeable, create a new PrettyAxis object instead")
    
    @property
    def plot_lines(self):
        return self._plot_lines
    
    def remove_plot_lines(self):
        for line in self._plot_lines.values():
            self.ax.lines.remove(line)
        self._plot_lines = {}

class PrettySlider(PrettyObject):
    """A data slider that can be added to a plot. The slider is a line with a dot that can be moved along the line.
    
    Methods
    -------
    Only use the methods listed here!!!
    See their individual docstrings for more information

        update_slider_dot: Update the position of the slider dot.    
    """

    def __init__(self, ax_plot, x_pos, y_pos, data_lim, arrow_style = '|-|', slider_dot_data = None, 
                horizontal = True, alpha = 1, c = ("black", "blue"), labels = (None, None, None), 
                arrow_lw = 2, ball_markersize = 10, ball_marker = 'o',
                label_size = 15, center_label_offset = 0.2, label_c = (None, None, None) )-> None:
        """
        Parameters
        ----------
            ax_plot : matplotlib.axes.Axes
                The axes of the plot to which the slider is added.
            x_pos : tuple
                The x-coordinates of the start and end of the slider.
            y_pos : tuple
                The y-coordinates of the start and end of the slider.
            data_lim : tuple
                The data limits of the slider.
            arrow_style : str, optional
                The style of the arrow. The default is '|-|'.
            slider_dot_data : float, optional
                The position of the slider dot in the data coordinates. If not given, the slider dot is placed in the middle of the slider
            horizontal : bool, optional
                Whether the slider is oriented horizontally or vertically. The default is True.
            alpha : float, optional
                The alpha value of the slider. The default is 1.
            c : tuple, optional
                The color of the slider arrow and the slider ball. The default is ("black", "blue").
            labels : tuple of str or None, optional
                The labels of the slider. The first element is the label of the left or bottom end of the slider,
                the second element is the label of the right or top end of the slider, 
                the third element is the label of the center of the slider
            arrow_lw : float, optional
                The line width of the slider arrow. The default is 2.
            ball_markersize : float, optional
                The size of the slider ball. The default is 10.
            ball_marker : str, optional
                The marker of the slider ball. The default is 'o'.
            label_size : float, optional
                The size of the labels. The default is 15.
            center_label_offset : float, optional
                The distance between the center label and the center of the slider. The default is 0.2.
            label_c : tuple of str or None, optional
                The color of the labels. The default is (None, None, None) which means that the color of the labels 
                are black

            
        """
        super().__init__(ax_plot, x_pos, y_pos)
        
        self.data_lim = data_lim
        self.arrow_style = arrow_style
        self.slider_dot_data = slider_dot_data
        self.horizontal = horizontal
        self._alpha = alpha
        self.c = c
        self.arrow_lw = arrow_lw
        self.ball_markersize = ball_markersize
        self.ball_marker = ball_marker
        self.ax_objects = {}
        self.labels = labels
        self.label_size = label_size
        self.center_label_offset = center_label_offset
        self.label_c = label_c
        self._check_validity()
        self._build_slider()
    
    def update_slider_dot(self, slider_dot_data):
        """Update the position of the slider dot
        Parameters
        ----------
            slider_dot_data : float
                The new position of the slider dot in the data coordinates.
        """

        assert self.data_lim[0] <= slider_dot_data <= self.data_lim[1]
        self.slider_dot_data = slider_dot_data

        if self.horizontal:
            dot_x_pos = self.x_pos[0] + (self.slider_dot_data - self.data_lim[0]) / (self.data_lim[1] - self.data_lim[0]) * (self.x_pos[1] - self.x_pos[0])
            self.ax_objects['slider_dot'].set_xdata([dot_x_pos])
        else:
            dot_y_pos = self.y_pos[0] + (self.slider_dot_data - self.data_lim[0]) / (self.data_lim[1] - self.data_lim[0]) * (self.y_pos[1] - self.y_pos[0])
            self.ax_objects['slider_dot'].set_ydata(dot_y_pos)

    def _check_validity(self):
        if self.horizontal and self.y_pos[0] != self.y_pos[1]:
            raise ValueError("y_pos[0] != y_pos[1], which means that the slider is not horizontal")
        elif not self.horizontal and self.x_pos[0] != self.x_pos[1]:
            raise ValueError("x_pos[0] != x_pos[1], which means that the slider is not vertical")
        elif self.y_pos[0] != self.y_pos[1] and self.x_pos[0] != self.x_pos[1]:
            raise ValueError("x_pos[0] != x_pos[1] and y_pos[0] != y_pos[1], which means that the slider is tilted")

        if self.slider_dot_data is None:
            self.slider_dot_data = np.mean(self.data_lim)

        if self.slider_dot_data < self.data_lim[0] or self.slider_dot_data > self.data_lim[1]:
            raise ValueError("slider_dot_data is not in data_lim")
        
        new_label_c = []
        for c in self.label_c:
            if c is None:
                new_label_c.append(self.c[0])
            else:
                new_label_c.append(c)
        self.label_c = new_label_c

    def _build_slider(self):
        arrow = FancyArrowPatch(( self.x_pos[0], self.y_pos[0]), ( self.x_pos[1], self.y_pos[1]),
            arrowstyle=self.arrow_style, mutation_scale=10,
            lw = self.arrow_lw, color = self.c[0],
            alpha = self._alpha
            )
        self.ax_plot.add_patch(arrow)
        arrow.set_zorder(1)
        self.ax_objects['arrow'] = arrow

        if self.horizontal:
            dot_x_pos = self.x_pos[0] + (self.slider_dot_data - self.data_lim[0]) / (self.data_lim[1] - self.data_lim[0]) * (self.x_pos[1] - self.x_pos[0])
            slider_dot = self.ax_plot.plot(dot_x_pos, self.y_pos[0], self.ball_marker, color = self.c[1], alpha = self._alpha, markersize = self.ball_markersize)
        else:
            dot_y_pos = self.y_pos[0] + (self.slider_dot_data - self.data_lim[0]) / (self.data_lim[1] - self.data_lim[0]) * (self.y_pos[1] - self.y_pos[0])
            slider_dot = self.ax_plot.plot(self.x_pos[0], dot_y_pos, self.ball_marker, color = self.c[1], alpha = self._alpha, markersize = self.ball_markersize)
        
        self.ax_objects['slider_dot'] = slider_dot[0]
        if self.labels[0]:
            if self.horizontal:
                ha_0 = "right"
                va_0 = "center"
            else:
                ha_0 = "center"
                va_0 = "top"
            self.ax_objects['label_left'] = self.ax_plot.text(self.x_pos[0], self.y_pos[0], self.labels[0], 
                                            size = self.label_size, ha = ha_0, va = va_0, color = self.label_c[0], 
                                            alpha = self._alpha, math_fontfamily = math_fontfamily)
        
        if self.labels[1]:
            if self.horizontal:
                ha_1 = "left"
                va_1 = "center"
            else:
                ha_1 = "center"
                va_1 = "bottom"
            self.ax_objects['label_right'] = self.ax_plot.text(self.x_pos[1], self.y_pos[1], self.labels[1], 
                                            size = self.label_size, ha = ha_1, va = va_1, color = self.label_c[1], alpha = self._alpha, math_fontfamily = math_fontfamily)
            
        if self.labels[2]:
            if self.horizontal:
                self.ax_objects['label_center'] = self.ax_plot.text(np.mean(self.x_pos), self.y_pos[0] + self.center_label_offset, self.labels[2], size = self.label_size, ha = "center", color = self.label_c[2], alpha = self._alpha, math_fontfamily = math_fontfamily)
            else:
                label_side = "left" if self.center_label_offset < 0 else "right"
                self.ax_objects['label_center'] = self.ax_plot.text(self.x_pos[0] + self.center_label_offset, np.mean(self.y_pos), self.labels[2], size = self.label_size, ha = label_side, color = self.label_c[2], alpha = self._alpha, math_fontfamily = math_fontfamily)
    
        
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        for obj in self.ax_objects.values():
            obj.set_alpha(alpha)


if __name__ == "__main__":
    pass
