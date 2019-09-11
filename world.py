import json
import os.path as osp
import cityflow

import numpy as np
from math import atan2, pi
import sys

def _get_direction(road, out=True):
    if out:
        x = road["points"][1]["x"] - road["points"][0]["x"]
        y = road["points"][1]["y"] - road["points"][0]["y"]
    else:
        x = road["points"][-2]["x"] - road["points"][-1]["x"]
        y = road["points"][-2]["y"] - road["points"][-1]["y"]
    tmp = atan2(x, y)
    return tmp if tmp >= 0 else (tmp + 2*pi)

class Intersection(object):
    def __init__(self, intersection, world):
        self.id = intersection["id"]
        self.eng = world.eng
        
        # incoming and outgoing roads of each intersection, clock-wise order from North
        self.roads = []
        self.outs = []
        self.directions = []
        self.out_roads = None
        self.in_roads = None

        # links and phase information of each intersection
        self.roadlinks = []
        self.lanelinks_of_roadlink = []
        self.startlanes = []
        self.lanelinks = []
        self.phase_available_roadlinks = []
        self.phase_available_lanelinks = []
        self.phase_available_startlanes = []

        # define yellow phases, currently default to 0
        self.yellow_phase_id = [0]
        self.yellow_phase_time = 3

        # parsing links and phases
        for roadlink in intersection["roadLinks"]:
            self.roadlinks.append((roadlink["startRoad"], roadlink["endRoad"]))
            lanelinks = []
            for lanelink in roadlink["laneLinks"]:
                startlane = roadlink["startRoad"] + "_" + str(lanelink["startLaneIndex"])
                self.startlanes.append(startlane)
                endlane = roadlink["endRoad"] + "_" + str(lanelink["endLaneIndex"])
                lanelinks.append((startlane, endlane))
            self.lanelinks.extend(lanelinks)
            self.lanelinks_of_roadlink.append(lanelinks)

        self.startlanes = list(set(self.startlanes))

        phases = intersection["trafficLight"]["lightphases"]
        self.phases = [i for i in range(len(phases)) if not i in self.yellow_phase_id]
        for i in self.phases:
            phase = phases[i]
            self.phase_available_roadlinks.append(phase["availableRoadLinks"])
            phase_available_lanelinks = []
            phase_available_startlanes = []
            for roadlink_id in phase["availableRoadLinks"]:
                lanelinks_of_roadlink = self.lanelinks_of_roadlink[roadlink_id]
                phase_available_lanelinks.extend(lanelinks_of_roadlink)
                for lanelinks in lanelinks_of_roadlink:
                    phase_available_startlanes.append(lanelinks[0])
            self.phase_available_lanelinks.append(phase_available_lanelinks)
            phase_available_startlanes = list(set(phase_available_startlanes))
            self.phase_available_startlanes.append(phase_available_startlanes)

        # record phase info
        self.current_phase = 0 # phase id in self.phases (excluding yellow)
        self._current_phase = self.phases[0] # true phase id (including yellow)
        self.current_phase_time = 0
        self.action_before_yellow = None


    def insert_road(self, road, out):
        self.roads.append(road)
        self.outs.append(out)
        self.directions.append(_get_direction(road, out))

    def sort_roads(self, RIGHT):
        order = sorted(range(len(self.roads)), key=lambda i: (self.directions[i], self.outs[i] if RIGHT else not self.outs[i]))
        self.roads = [self.roads[i] for i in order]
        self.directions = [self.directions[i] for i in order]
        self.outs = [self.outs[i] for i in order]
        self.out_roads = [self.roads[i] for i, x in enumerate(self.outs) if x]
        self.in_roads = [self.roads[i] for i, x in enumerate(self.outs) if not x]

    def _change_phase(self, phase, interval):
        self.eng.set_tl_phase(self.id, phase)
        self._current_phase = phase
        self.current_phase_time = interval

    def step(self, action, interval):
        # if current phase is yellow, then continue to finish the yellow phase
        # recall self._current_phase means true phase id (including yellows)
        # self.current_phase means phase id in self.phases (excluding yellow)
        if self._current_phase in self.yellow_phase_id:
            if self.current_phase_time >= self.yellow_phase_time:
                self._change_phase(self.phases[self.action_before_yellow], interval)
                self.current_phase = self.action_before_yellow
            else:
                self.current_phase_time += interval
        else:
            if action == self.current_phase:
                self.current_phase_time += interval
            else:
                if self.yellow_phase_time > 0:
                    self._change_phase(self.yellow_phase_id[0], interval)
                    self.action_before_yellow = action
                else:
                    self._change_phase(action, interval)
                    self.current_phase = action


class World(object):
    """
    Create a CityFlow engine and maintain informations about CityFlow world
    """
    def __init__(self, cityflow_config, thread_num):
        print("building world...")
        self.eng = cityflow.Engine(cityflow_config, thread_num=thread_num)
        with open(cityflow_config) as f:
            cityflow_config = json.load(f)
        self.roadnet = self._get_roadnet(cityflow_config)
        self.RIGHT = True # vehicles moves on the right side, currently always set to true due to CityFlow's mechanism
        self.interval = cityflow_config["interval"]

        # get all non virtual intersections
        self.intersections = [i for i in self.roadnet["intersections"] if not i["virtual"]]
        self.intersection_ids = [i["id"] for i in self.intersections]

        # create non-virtual Intersections
        print("creating intersections...")
        non_virtual_intersections = [i for i in self.roadnet["intersections"] if not i["virtual"]]
        self.intersections = [Intersection(i, self) for i in non_virtual_intersections]
        self.intersection_ids = [i["id"] for i in non_virtual_intersections]
        self.id2intersection = {i.id: i for i in self.intersections}
        print("intersections created.")

        # id of all roads and lanes
        print("parsing roads...")
        self.all_roads = []
        self.all_lanes = []

        for road in self.roadnet["roads"]:
            self.all_roads.append(road["id"])
            i = 0
            for _ in road["lanes"]:
                self.all_lanes.append(road["id"] + "_" + str(i))
                i += 1

            iid = road["startIntersection"]
            if iid in self.intersection_ids:
                self.id2intersection[iid].insert_road(road, True)
            iid = road["endIntersection"]
            if iid in self.intersection_ids:
                self.id2intersection[iid].insert_road(road, False)

        for i in self.intersections:
            i.sort_roads(self.RIGHT)
        print("roads parsed.")

        # initializing info functions
        self.info_functions = {
            "vehicles": (lambda : self.eng.get_vehicles(include_waiting=True)),
            "lane_count": self.eng.get_lane_vehicle_count,
            "lane_waiting_count": self.eng.get_lane_waiting_vehicle_count,
            "lane_vehicles": self.eng.get_lane_vehicles,
            "time": self.eng.get_current_time
        }
        self.fns = []
        self.info = {}

        print("world built.")

    def _get_roadnet(self, cityflow_config):
        roadnet_file= osp.join(cityflow_config["dir"], cityflow_config["roadnetFile"])
        with open(roadnet_file) as f:
            roadnet = json.load(f)
        return roadnet

    def subscribe(self, fns):
        if isinstance(fns, str):
            fns = [fns]
        for fn in fns:
            if fn in self.info_functions:
                if not fn in self.fns:
                    self.fns.append(fn)
            else:
                raise Exception("info function %s not exists" % fn)

    def step(self, actions=None):
        if actions is not None:
            for i, action in enumerate(actions):
                self.intersections[i].step(action, self.interval)
        self.eng.next_step()
        self._update_infos()

    def reset(self):
        self.eng.reset()
        self._update_infos()

    def _update_infos(self):
        self.info = {}
        for fn in self.fns:
            self.info[fn] = self.info_functions[fn]()

    def get_info(self, info):
        return self.info[info]


if __name__ == "__main__":
    world = World("examples/config.json", thread_num=1)
    #print(len(world.intersections[0].startlanes))
    print(world.intersections[0].phase_available_startlanes)