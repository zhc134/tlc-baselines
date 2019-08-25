from . import BaseMetric
import numpy as np

class TravelTimeMetric(BaseMetric):
    """
    Calculate average travel time of all vehicles.
    For each vehicle, travel time measures time between it entering and leaving the roadnet.
    """
    def __init__(self, world):
        self.world = world
        self.world.subscribe(["lane_vehicles", "time"])
        self.vehicle_enter_time = {}
        self.travel_times = []

    def update(self, done=False):
        lane_vehicles = self.world.get_info("lane_vehicles")
        current_time = self.world.get_info("time")
        vehicles = sum(lane_vehicles.values(), [])

        for vehicle in vehicles:
            if not vehicle in self.vehicle_enter_time:
                self.vehicle_enter_time[vehicle] = current_time

        for vehicle in list(self.vehicle_enter_time):
            if done or not vehicle in vehicles:
                self.travel_times.append(current_time - self.vehicle_enter_time[vehicle])
                del self.vehicle_enter_time[vehicle]
        
        return np.mean(self.travel_times) if len(self.travel_times) else 0
        
