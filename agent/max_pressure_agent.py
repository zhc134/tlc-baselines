from . import BaseAgent

class MaxPressureAgent(BaseAgent):
    """
    Agent using Max-Pressure method to control traffic light
    """
    def __init__(self, action_space, I, world):
        super().__init__(action_space)
        self.I = I
        self.world = world
        world.subscribe("lane_count")
        
        # the minimum duration of time of one phase
        self.t_min = 5

    def get_ob(self):
        return None

    def get_action(self, ob):
        # get lane pressure
        lvc = self.world.get_info("lane_count")

        if self.I.current_phase_time < self.t_min:
            return self.I.current_phase

        max_pressure = None
        action = -1
        for phase_id in range(len(self.I.phases)):
            pressure = sum([lvc[start] - lvc[end] for start, end in self.I.phase_available_lanelinks[phase_id]])
            if max_pressure is None or pressure > max_pressure:
                action = phase_id
                max_pressure = pressure

        return action

    def get_reward(self):
        return None