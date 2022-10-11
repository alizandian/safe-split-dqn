import tkinter
import matplotlib.pyplot as plt
import networkx as nx
from typing import List
from .node import Node

colors = {-1: "red", 0: "grey", 1:"white", 2:"orange"}

tk_root_table = None
tk_root_graph = None

def draw_table(coord: List[List[int]] = [[1,0,0,1], [1,1,1,1], [1,0,0,1]]):
    root = tkinter.Tk()
    min_x, min_y = 0, 0
    max_x, max_y = 500, 500
    canvas = tkinter.Canvas(root, bg="white", height=max_x, width=max_y)

    for y, rows in enumerate(coord):
        for x, column in enumerate(rows):
            lx = max_x - min_x
            ly = max_y - min_y
            y_unit = lx / len(rows)
            x_unit = ly / len(coord)
            canvas.create_rectangle(x * x_unit, max_y - (y * y_unit), (x * x_unit) + x_unit, max_y -((y * y_unit) + y_unit), outline="#fb0", fill=colors[column])

    canvas.pack()
    root.mainloop()

def draw_graph(nodes: List[Node]):
    graph_edges = []
    graph_nodes = []
    edges_labels = {}
    formulas = []
    formulas_index = 0
    labels = {}
    for node in nodes:
        values = {1:"SAFE", -1:"UNSAFE", 0:"UNSURE", 2:"INCREASE RESOLUTION!"}
        index = nodes.index(node)
        graph_nodes.append(index)
        labels[index] = values[node.value]
        for n in node.nodes:
            i = nodes.index(n)
            graph_edges.append((index, i))
            formulas.append(n.region_formula_text())
            edges_labels[(index, i)] = str(formulas_index)
            formulas_index += 1

    G = nx.Graph()
    G.add_nodes_from(graph_nodes)
    G.add_edges_from(graph_edges)
    pos = nx.spring_layout(G)
    plt.figure()
    colors = {"SAFE": "white", "UNSURE": "grey", "UNSAFE": "red", "INCREASE RESOLUTION!": "orange"}
    color_values = [colors[label] for label in labels.values()]
    nx.draw(G, pos, edge_color='black', node_size=4000, width=1, linewidths=1, node_color=color_values, alpha=0.9, labels=labels)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edges_labels, font_color='red')
    plt.axis('off')
    plt.show()

def draw_graph_grid(nodes: List[Node]):
    root = tkinter.Tk()
    w, h = 500, 500
    canvas = tkinter.Canvas(root, bg="white", height=h, width=w)

    for node in nodes:
        for r in node.regions:
            x0, x1, y0, y1 = r.windowed_values(w, h)
            canvas.create_rectangle((x0, h - y0, x1, h - y1), fill=colors[node.value])

    canvas.pack()
    root.mainloop()
