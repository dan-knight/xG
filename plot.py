import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import matplotlib.cm as cm

import numpy as np


class Pitch:
    def __init__(self, ax, title=None, bg=None, plot=True):
        self.ax = ax
        self.ax.set_title(title)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        if bg is not None:
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=bg,
                                       transform=ax.transAxes, zorder=-1))

        if plot:
            self.draw_pitch()

    def draw_pitch(self):
        def plot_sidelines():
            self._draw_line([0, 0], [0, pitch_length])
            self._draw_line([0, pitch_width], [pitch_length, pitch_length])
            self._draw_line([pitch_width, pitch_width], [pitch_length, 0])
            self._draw_line([pitch_width, 0], [0, 0])

        def plot_centre_circle():
            centre_x = centre_of_goal
            centre_y = pitch_length

            x = np.linspace(centre_circle_radius * -1, centre_circle_radius, 1000)
            y = np.sqrt(centre_circle_radius ** 2 - (x ** 2)) * -1
            self._draw_line(x + centre_x, y + centre_y)

        def plot_penalty_area():
            def plot_box(size):
                goal_line = 0
                edge_of_box = size

                width = goal_size + (size * 2)

                def get_y_coordinates():
                    split_box_width = width / 2
                    return centre_of_goal + split_box_width, centre_of_goal - split_box_width

                top, bottom = get_y_coordinates()

                self._draw_line([top, top], [goal_line, edge_of_box])
                self._draw_line([top, bottom], [edge_of_box, edge_of_box])
                self._draw_line([bottom, bottom], [goal_line, edge_of_box])

            plot_box(penalty_area_size)
            plot_box(six_yard_box_size)

        def plot_goal():
            size = (goal_size / 2)
            goal_left, goal_right = (centre_of_goal - size, centre_of_goal + size)
            net_depth = size * -0.5

            self._draw_line([goal_left, goal_left], [0, net_depth])
            self._draw_line([goal_left, goal_right], [net_depth, net_depth])
            self._draw_line([goal_right, goal_right], [net_depth, 0])

        plot_sidelines()
        plot_centre_circle()
        plot_penalty_area()
        plot_goal()

    def _draw_line(self, x, y):
        self.ax.plot(x, y, color='black', linewidth=1.2)

    def plot_histogram(self, histo):
        extent = [0, pitch_length, 0, pitch_width]
        self.ax.imshow(histo, extent=extent, cmap='Reds')


# Config
pitch_length = 116 / 2
pitch_width = 76

centre_circle_radius = 10

goal_size = 8
centre_of_goal = pitch_width / 2
goalposts = [centre_of_goal + (post * (goal_size * 0.5)) for post in [1, -1]]

penalty_area_size = 18
six_yard_box_size = 6


if __name__ == '__main__':
    fig, ax = plt.subplots()
    Pitch(ax, '17/18 EPL Shots By Location')
    plt.show()
