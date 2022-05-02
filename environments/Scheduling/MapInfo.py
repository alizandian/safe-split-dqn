from enum import IntEnum

class color(IntEnum):
    R = 1
    G = 2
    B = 3
    Y = 4

class MapInfo(object):
    def __init__(self):
        # system map graph
        self.all_nodes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13] # the index == state name for convenience

        self.input_tasks = {
                2: [],
                5: [],
                8: [],
                11: [],
            }

        self.output_target = {
                0: color.R,
                1: color.G,
                12: color.B,
                13: color.Y,
            }

        self.edges = [ # index is automatically the ID of each edge (e.g., (2,3) has the ID of 0)
                # input paths
                (2,3), 
                (5,4),
                (8,9), 
                (11,10),
                # output paths
                (3,0),
                (4,1),
                (9,12),
                (10,13),
                # intermediate paths       
                (3,4), 
                (6,7),
                (9,10),
                (3,6),
                (4,7),
                (6,9), 
                (7,10), 
            ]

        self.connected_edges = {} 
        """ stores all the associated edges from a node as a dictionary of the form:
        {
            3 : 
            [ 
                [4, 8, 11],     # the index of edges that starts with 3 
                [0]             # the index of edges that finishes with 3 
            ],
        }
        """
        for n in self.all_nodes:
            forward = []
            reverse = []
            e_id = 0
            for (s,d) in self.edges:
                if s == n:
                    forward.append(e_id)
                elif d == n:
                    reverse.append(e_id)
                e_id += 1 # edge id is automatically the index of the edge array
            self.connected_edges[n] = [forward, reverse]
    
    def load_scenario1(self):
        self.input_tasks[2] = [color.R, color.B, color.B, color.Y, color.G]
        self.input_tasks[5] = [color.B, color.R, color.Y, color.R, color.G]
        self.input_tasks[8] = [color.Y, color.B, color.G, color.R, color.R]
        self.input_tasks[11] = [color.G, color.R, color.B, color.Y, color.B]

    def get_connected_edges(self, node):
        return self.connected_edges[node]

    def get_tasks_dict(self):
        return self.input_tasks

    def get_nodes(self):
        return self.all_nodes
    
    def get_edges(self):
        return self.edges
    
    def get_output_dict(self):
        return self.output_target

    def generate_tasks(self, seed):
        # random generation of tasks
        pass

    """ Return predefiend geometric position (x-y coordinates) of each node for drawing purpose """
    def get_geometric_info(self): 
        point_coord = {
            0 : (4,0),
            1 : (8,0),
            2 : (0,4),
            3 : (4,4),
            4 : (8,4),
            5 : (12,4),
            6 : (4,8),
            7 : (8,8),
            8 : (0,12),
            9 : (4,12),
            10 : (8,12),
            11 : (12,12),
            12 : (4,16),
            13 : (8,16),
        }
        i = 0
        point_pairs = {}
        for e in self.edges:
            point_pairs[i]=e
            i+=1

        return [point_coord, point_pairs]
