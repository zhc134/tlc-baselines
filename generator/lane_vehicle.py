import numpy as np
from . import BaseGenerator

class LaneVehicleGenerator(BaseGenerator):
    """
    Generate State or Reward based on statistics of lane vehicles.

    Parameters
    ----------
    world : World object
    iid : id of intersection
    fns : list of statistics to get, currently support "count" and "waiting_count"
    in_only : boolean, whether to compute incoming lanes only
    average : None or str
        None means no averaging
        "road" means take average of lanes on each road
        "all" means take average of all lanes
    negative : boolean, whether return negative values (mostly for Reward)
    """
    def __init__(self, world, iid, fns, in_only=False, average=None, negative=False):
        self.world = world

        # get lanes of intersections
        self.lanes = []
        roads = self.world.intersection_roads[iid]
        if in_only:
            roads = roads["in_roads"]
        else:
            roads = roads["roads"]
        for road in roads:
            from_zero = (road["startIntersection"] == iid) if self.world.RIGHT else (road["endIntersection"] == iid)
            self.lanes.append([road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])

        # subscribe functions
        self.world.subscribe(fns)
        self.fns = fns

        self.average = average
        self.negative = negative

    def generate(self):
        results = [self.world.get_info(fn) for fn in self.fns]

        ret = np.array([])
        for i in range(len(self.fns)):
            result = results[i]
            fn_result = np.array([])

            for road_lanes in self.lanes:
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
            ret = np.append(ret, fn_result)
        if self.negative:
            ret = ret * (-1)
        return ret

if __name__ == "__main__":
    from world import World
    world = World("examples/config.json", thread_num=1)
    laneVehicle = LaneVehicle(world, world.intersection_ids[0], ["count"], False, "road")
    for _ in range(100):
        world.step()
    print(laneVehicle.generate())