import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

color_code = {
    0 : 'w',
    1 : 'r',
    2 : 'g',
    3 : 'b',
    4 : 'y',
    5 : 'purple',
}

class Animation(object):
    def __init__(self, geometric_info):
        self.node_coordinates, self.node_pairs = geometric_info 

    def get_line_xy(self, node1, node2):
        spacing = 1
        x1, y1 = node1
        x2, y2 = node2

        if x1 < x2:
            xdata = [x1+spacing, x2-spacing]
        elif x1 > x2:
            xdata = [x1-spacing, x2+spacing]
        else:
            xdata = [x1, x2]

        if y1 < y2:
            ydata = [y1+spacing, y2-spacing]
        elif y1 > y2:
            ydata = [y1-spacing, y2+spacing]
        else:
            ydata = [y1, y2]

        return xdata, ydata

    def add_shapes(self, state, action):
        plt.cla()
        for id, xy in self.node_coordinates.items():
            circle = plt.Circle(xy, radius=1, fc=color_code[state[id]], ec='k')
            plt.gca().add_patch(circle)

        for id, pair in self.node_pairs.items():
            (s,d) = pair
            xdata, ydata = self.get_line_xy(self.node_coordinates[s],self.node_coordinates[d])
            if action[id] == 0:
                line = plt.Line2D(xdata, ydata, lw=1.5, color='k')
                plt.gca().add_line(line)
            elif action[id] == 1:
                arrow = plt.Arrow(xdata[0], ydata[0], xdata[1]-xdata[0], ydata[1]-ydata[0], color='black')
                plt.gca().add_patch(arrow)
            else:
                arrow = plt.Arrow(xdata[1], ydata[1], xdata[0]-xdata[1], ydata[0]-ydata[1], color='black')
                plt.gca().add_patch(arrow)


    def playback_animation(self, states, actions):
        self.time = 0
        self.max_time = len(states)
        self.states = states
        self.actions = actions

        fig, ax = plt.subplots()
        state = self.states[0]
        action = self.actions[0]
        self.add_shapes(state, action)

        plt.subplots_adjust(bottom=0.2) # buttons will be placed at buttom (so make some spaces)
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.play_forward)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(self.play_backward)

        plt.sca(ax)
        plt.title("Time step: " + str(self.time))
        plt.axis('scaled')
        plt.show()

    def play_forward(self, event):
        self.time += 1
        if self.time >= self.max_time:
            self.time = self.max_time - 1
            print("End of the play")
            return 

        state = self.states[self.time]
        action = self.actions[self.time]
        self.add_shapes(state, action)

        plt.title("Time step: " + str(self.time))
        plt.axis('scaled')
        plt.draw()

    def play_backward(self, event):
        self.time -= 1
        if self.time < 0:
            print("Start of the play")
            self.time = 0
            return

        state = self.states[self.time]
        action = self.actions[self.time]
        self.add_shapes(state, action)
        
        plt.title("Time step: " + str(self.time))
        plt.axis('scaled')
        plt.draw()    