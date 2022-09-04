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
    for index, node in enumerate(nodes):
        values = {1:"SAFE", -1:"UNSAFE", 0:"UNSURE", 2:"INCREASE RESOLUTION!"}
        graph_nodes.append(node.i)
        labels[index] = values[node.value]
        for n in node.nodes:
            graph_edges.append((node.i, n.i))
            formulas.append(n.region_formula_text())
            edges_labels[(node.i, n.i)] = str(formulas_index)
            formulas_index += 1

    G = nx.Graph()
    G.add_nodes_from(graph_nodes)
    G.add_edges_from(graph_edges)
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1, node_color='pink', alpha=0.9, labels=labels)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edges_labels, font_color='red')
    plt.axis('off')
    plt.show()

def draw_graph_grid(nodes: List[Node], grid_dimention):
    root = tkinter.Tk()
    min_x, min_y = 0, 0
    max_x, max_y = 500, 500
    node_size = 20
    canvas = tkinter.Canvas(root, bg="white", height=max_x, width=max_y)

    lx = max_x - min_x
    ly = max_y - min_y
    gx, gy = grid_dimention
    xu = lx / gx
    yu = ly / gy


    for index, node in enumerate(nodes):
        for r in node.regions:
            canvas.create_rectangle(r.x_min*xu, max_y- r.y_min*yu, r.x_max*xu, max_y - r.y_max*yu, fill=colors[node.value])

    canvas.pack()
    root.mainloop()
