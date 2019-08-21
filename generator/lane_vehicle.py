import numpy as np
from . import BaseGenerator

class LaneVehicle(BaseGenerator):
    """
    Generate State or Reward based on statistics of lane vehicles.

    Parameters
    ----------
    world : World object
    fns : list of statistics to get, currently support "count" and "waiting_count"
    in_only : boolean, whether to compute incoming lanes only
    average : None or str
        None means no averaging
        "road" means take average of lanes on each road
        "all" means take average of all lanes
    negative : boolean, whether return negative values (mostly for Reward)
    """
    def __init__(self, world, fns, in_only=False, average=None, negative=False):
        self.world = world

        # get lanes of intersections
        self.intersection_lanes = {}
        iroads = self.world.intersection_roads
        for iid in self.world.intersection_ids:
            lane_ids = []
            roads = iroads[iid]
            if in_only:
                roads = roads["in_roads"]
            else:
                roads = roads["roads"]
            for road in roads:
                from_zero = (road["startIntersection"] == iid) if self.world.RIGHT else (road["endIntersection"] == iid)
                lane_ids.append([road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])
            self.intersection_lanes[iid] = lane_ids

        self.eng = self.world.eng

        # init functions
        self.fns = []
        for fn in fns:
            if fn == "count":
                self.fns.append(self.eng.get_lane_vehicle_count)
            elif fn == "waiting_count":
                self.fns.append(self.eng.get_lane_waiting_vehicle_count)
            else:
                raise Exception("statistic not exists")

        self.average = average
        self.negative = negative

    def generate(self):
        results = [fn() for fn in self.fns]

        rets = []
        for iid in self.world.intersection_ids:
            i_result = np.array([])
            for i in range(len(self.fns)):
                result = results[i]
                fn_result = np.array([])

                for road_lanes in self.intersection_lanes[iid]:
                    road_result = []
                    for lane_id in road_lanes:
                        road_result.append(result[lane_id])
                    if self.average == "road" or self.average == "all":
                        road_result = np.mean(road_result)
                    else:
                        road_result = np.array(road_result)
                    fn_result = np.append(fn_result, road_result)
                
                if self.average == "all":
                    fn_result = np.mean(fn_result)
                i_result = np.append(i_result, fn_result)
            if self.negative:
                i_result = i_result * (-1)
            rets.append(i_result)
        return rets

if __name__ == "__main__":
    from world import World
    world = World("examples/config.json", thread_num=1)
    laneVehicle = LaneVehicle(world, ["count"], False, "road")
    for _ in range(100):
        world.eng.next_step()
    print(laneVehicle.generate())