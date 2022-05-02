from __future__ import absolute_import
from __future__ import print_function

import os
import random
import sys
import optparse
import numpy as np
from gym import spaces, logger
import csv

try:
    #sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary 
    import traci
except ImportError:
    sys.exit("please declare environment variable 'SUMO_HOME'")


"""
    observations:                              min             max
        0       phase_memory                    0               3
        1       Lane 1 Queue                  -Int32          +Int32
        ...
        n       Lane n

    actions:
        0       GGrrrrGGrrrr 
        1       rrGrrrrrGrrr
        2       rrrGGrrrrGGr
        3       rrrrrGrrrrrG

    Reward: traffic throughput
"""

green_phase = {
    0: "GGrrrrGGrrrr",
    1: "rrGrrrrrGrrr",
    2: "rrrGGrrrrGGr",
    3: "rrrrrGrrrrrG",
}

yellow_phase = {
    0: "yyrrrryyrrrr",
    1: "rryrrrrryrrr",
    2: "rrryyrrrryyr",
    3: "rrrrryrrrrry",
}

class IntersectionEnv():
    def __init__(self, route_file=None, use_gui=False, random=True, vehicle_spawn_rate=0.5, max_vehicles=300):
        self._net = "net/single-intersection.sumocfg"
        self.tid = 't'
        self.random = random
        self.random_spawn_rate = vehicle_spawn_rate
        self.max_vehicles = max_vehicles

        self.yellow_time = 3 # seconds
        self.slot_length = 10 # seconds
        
        if use_gui:
            self.sumoBinary = checkBinary('sumo-gui')
        else:
            self.sumoBinary = checkBinary('sumo')

        self.simulation_is_on = False

        traci.start([checkBinary('sumo'), "-c", self._net, "-W", "True"])
        self.approaching_lanes = list(set(traci.trafficlight.getControlledLanes(self.tid)))
        self.lanes_length = {lane: traci.lane.getLength(lane) for lane in self.approaching_lanes}
        self.routes = traci.route.getIDList()
        traci.close()
        print("IntersectionEnv has successfully loaded the configuration of 'single-ingersection.sumocfg' file.")

        self.state = None

    def process_time(self, time):
        steps = 1 if time <= 0 else time # trigger at least once
        for i in range(0, steps):
            if self.vehicle_count < self.max_vehicles:
                self.spawn_vehicle()
            traci.simulationStep()

    def spawn_vehicle(self):
        if self.random:
            self.vehicle_spawn_by_random()
        else:
            # TODO: weight base spawn
            self.vehicle_spawn_by_weights()

    def get_vehicle_count(self):
        return self.vehicle_count

    def get_max_vehicles(self):
        return self.max_vehicles

    def vehicle_spawn_by_random(self):
        if self.random_spawn_rate < 1:
            spawn = 1 if random.random() <= self.random_spawn_rate else 0
        else:
            spawn = self.random_spawn_rate

        selected_routes = random.sample(self.routes, spawn)
        for l in selected_routes:
            vid = 'veh_' + str(l) + str(self.get_time())
            traci.vehicle.add(vid, l, departLane='best', departSpeed='max')
        self.vehicle_count += spawn

    def get_random_weights(self):
        # TODO: implement later
        # vehicle spawn rate order is clockwise, starting from North
        # I.e., [ne, ns, nw], [es, ew, en], [sw, sn, se], [wn, we, ws]
        self.vehicle_spawn_weights = self.get_random_weights()

        return [ [1,1,1], [1,1,1], [1,1,1], [1,1,1] ]

    def set_gui(self, enable=True):
        # user can change the gui setting 
        if enable:
            self.sumoBinary = checkBinary('sumo-gui')
        else:
            self.sumoBinary = checkBinary('sumo')

    # users must always call the reset method before execuring step()
    def reset(self, use_gui=False):
        if self.simulation_is_on:
            traci.close()

        traci.start([self.sumoBinary, "-c", self._net, "--step-length", "1",
                            "--no-step-log", "true", "--waiting-time-memory", "500", "-W", "True"])

        self.simulation_is_on = True

        # initialization
        self.set_phase(green_phase[0])
        self.phase_memory = 0

        self.vehicle_count = 0
        self.accumulative_outflow = 0
        
        self.process_time(1) # trigger a single step to spawn vehicles
        self.state = [0] + self.get_vehicles_per_lane() + self.get_queues()

        return np.array(self.state, dtype=np.int32) 
    
    def step(self, action):
        if not self.simulation_is_on:
            print("SUMO simulation is not initiated. Use env.reset() to start a new simulation")
            return np.array(self.state, dtype=np.int32), 0.0, True, {} 

        # for reward calculation later
        pre_vehicles = self.get_vehicles()
        pre_queue_sum = sum(self.get_queues())

        # execute the controller
        self.execute_trafficlight(action)

        # next state
        self.state = [action] + self.get_vehicles_per_lane() + self.get_queues()

        # Reward
        new_vehicles = self.get_vehicles()
        reward = len(pre_vehicles) - len(new_vehicles)
        
        #print(self.state)
        outflow = len(pre_vehicles) - len(new_vehicles.intersection(pre_vehicles))
        self.accumulative_outflow += outflow
        
        # done 
        done = bool(
            len(self.get_vehicles()) == 0 and self.vehicle_count == self.max_vehicles
            #or outflow == 0 and pre_queue_sum > 0
        )

        return np.array(self.state, dtype=np.int32), reward, done, {}

    def get_time(self):
        return traci.simulation.getTime()

    def get_outflow(self):
        return self.accumulative_outflow

    def set_phase(self, phase):
        traci.trafficlight.setRedYellowGreenState(self.tid, phase)

    def get_vehicles_per_lane(self):
        return [ traci.lane.getLastStepVehicleNumber(lane) for lane in self.approaching_lanes]

    def get_waiting_time(self):
        return sum(self.get_waiting_time_per_lane())

    def get_queues(self):
        return [traci.lane.getLastStepHaltingNumber(lane) for lane in self.approaching_lanes]

    def execute_trafficlight(self, action):
        if self.phase_memory != action:
            # yellow phase
            self.set_phase(yellow_phase[self.phase_memory])
            self.process_time(self.yellow_time)
            self.set_phase(green_phase[action])
            self.process_time(self.slot_length - self.yellow_time)
            self.phase_memory = action
        else:
            # continue the green light
            self.process_time(self.slot_length)

    def get_waiting_time_per_lane(self):
        wait_time_per_lane = []
        for lane in self.approaching_lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                wait_time += traci.vehicle.getAccumulatedWaitingTime(veh)
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_vehicles(self):
        vehicles = []
        for l in self.approaching_lanes:
            vehicles = vehicles + list(traci.lane.getLastStepVehicleIDs(l))
        return set(vehicles)

    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepVehicleNumber(lane) / (self.lanes_length[lane] / vehicle_size_min_gap)) for lane in self.approaching_lanes]

    def get_total_flow(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return sum([min(1, traci.lane.getLastStepVehicleNumber(lane) / (self.lanes_length[lane] / vehicle_size_min_gap)) * traci.lane.getLastStepMeanSpeed(lane)
                 for lane in self.approaching_lanes])

    def seed(self, seed=None):
        pass

    def render(self):
        pass