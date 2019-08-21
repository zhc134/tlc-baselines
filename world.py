import json
import os.path as osp
import cityflow

import numpy as np
from math import atan2, pi
import sys

class World:
    def __init__(self, cityflow_config, thread_num):
        print("building world...")
        self.eng = cityflow.Engine(cityflow_config, thread_num=thread_num)
        self.roadnet = self._get_roadnet(cityflow_config)
        self.RIGHT = True # vehicles moves on the right side, currently always set to true due to CityFlow's mechanism

        # get all non virtual intersections
        self.intersections = []
        for intersection in self.roadnet["intersections"]:
            if not intersection["virtual"]:
                self.intersections.append(intersection)

        self.intersection_ids = [i["id"] for i in self.intersections]

        # get incoming and outgoing roads of each intersection, clock-wise order from North
        print("parsing road networks...")
        self.intersection_roads = {
            i_id: {
                "roads": [],
                "outs": [],
                "directions": []
            } for i_id in self.intersection_ids
        }

        def insert_road(iid, road, out):

            def get_direction(road, out=True):
                if out:
                    x = road["points"][1]["x"] - road["points"][0]["x"]
                    y = road["points"][1]["y"] - road["points"][0]["y"]
                else:
                    x = road["points"][-2]["x"] - road["points"][-1]["x"]
                    y = road["points"][-2]["y"] - road["points"][-1]["y"]
                tmp = atan2(x, y)
                return tmp if tmp >= 0 else (tmp + 2*pi)

            if iid in self.intersection_roads:
                roads = self.intersection_roads[iid]
                roads["roads"].append(road)
                roads["outs"].append(out)
                roads["directions"].append(get_direction(road, out))

        for road in self.roadnet["roads"]:
            insert_road(road["startIntersection"], road, True)
            insert_road(road["endIntersection"], road, False)

        for iid in self.intersection_ids:
            roads = self.intersection_roads[iid]
            directions = roads["directions"]
            outs = roads["outs"]
            order = sorted(range(len(roads["roads"])), key=lambda i: (directions[i], outs[i] if self.RIGHT else not outs[i]))
            roads["roads"] = [roads["roads"][i] for i in order]
            roads["directions"] = [directions[i] for i in order]
            roads["outs"] = [outs[i] for i in order]
            roads["out_roads"] = [roads["roads"][i] for i, x in enumerate(roads["outs"]) if x]
            roads["in_roads"] = [roads["roads"][i] for i, x in enumerate(roads["outs"]) if not x]
        print("road network parsed.")
        print("world built.")

    def _get_roadnet(self, cityflow_config):
        with open(cityflow_config) as f:
            cityflow_config = json.load(f)
        roadnet_file= osp.join(cityflow_config["dir"], cityflow_config["roadnetFile"])
        with open(roadnet_file) as f:
            roadnet = json.load(f)
        return roadnet

if __name__ == "__main__":
    world = World("examples/config.json", thread_num=1)