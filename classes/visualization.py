import tkinter
import matplotlib.pyplot as plt
import networkx as nx
from typing import List
from .graph import Node

colors = {-1: "grey", 0: "black", 1:"red", 2:"orange"}

def draw_table(coord: List[List[int]] = [[1,0,0,1], [1,1,1,1], [1,0,0,1]]):
    root = tkinter.Tk()
    min_x, min_y = 0, 0
    max_x, max_y = 500, 500
    canvas = tkinter.Canvas(root, bg="white", height=max_x, width=max_y)

    for y, rows in enumerate(coord):
        for x, column in enumerate(rows):
            lx = max_x - min_x
            ly = max_y - min_y
            x_unit = lx / len(rows)
            y_unit = ly / len(coord)
            canvas.create_rectangle(x * x_unit, y * y_unit, (x * x_unit) + x_unit, ( y * y_unit) + y_unit, outline="#fb0", fill=colors[column])

    canvas.pack()
    root.mainloop()

def draw_graph(nodes: List[Node]):
    graph_edges = []
    graph_nodes = []
    edges_labels = {}
    labels = {}
    for index, node in enumerate(nodes):
        values = {0:"SAFE", 1:"UNSAFE", 2:"UNSURE", -1:"INCREASE RESOLUTION!"}
        graph_nodes.append(node.i)
        labels[index] = values[node.value]
        for n in node.nodes:
            graph_edges.append((node.i, n.i))
            edges_labels[(node.i, n.i)] = n.region_formula_text()

    G = nx.Graph()
    G.add_nodes_from(graph_nodes)
    G.add_edges_from(graph_edges)
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1, node_color='pink', alpha=0.9, labels=labels)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edges_labels, font_color='red')
    plt.axis('off')
    plt.show()

def draw_graph_basic(nodes: List[Node]):
    root = tkinter.Tk()
    min_x, min_y = 0, 0
    max_x, max_y = 500, 500
    node_size = 20
    canvas = tkinter.Canvas(root, bg="white", height=max_x, width=max_y)
    
    lx = max_x - min_x
    ly = max_y - min_y
    x = lx / 3
    y_unit = ly / (len(nodes) + 1)


    node_positions = {}
    for i, node in enumerate(nodes):
        node_positions[node] = (x, y_unit * (i+1))
        canvas.create_oval(x, y_unit * (i+1), x + node_size, y_unit * (i+1) + node_size, fill=colors[node.value])
        canvas.create_text(x * 2, y_unit * (i+1), fill=colors[node.value], font="Times 20 italic bold", text=node.region_formula_text())

    for i, node in enumerate(nodes):
        for n in node.nodes:
            coord1 = node_positions[node]
            coord2 = node_positions[n]
            canvas.create_line(coord1[0] - (5 + i*5), coord1[1], coord2[0] - (5 + i*5), coord2[1])


    canvas.pack()
    root.mainloop()
