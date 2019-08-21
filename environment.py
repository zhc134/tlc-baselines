import gym
import numpy as np
import cityflow

class TSCEnv(gym.Env):
    def __init__(self, world, ob_generator, reward_generator):
        self.world = world
        self.ob_generator = ob_generator
        self.reward_generator = reward_generator
        
        self.eng = self.world.eng
        self.n_agents = len(self.world.intersection_ids)

        action_dims = []
        for intersection in self.world.intersections:
            action_dims.append(len(intersection["trafficLight"]["lightphases"]))
        self.action_space = gym.spaces.MultiDiscrete(action_dims)


    def step(self, actions):
        assert len(actions) == self.n_agents

        for i in range(self.n_agents):
            self.eng.set_tl_phase(self.world.intersection_ids[i], actions[i])
        
        self.eng.next_step()

        obs = self.ob_generator.generate()
        rewards = self.reward_generator.generate()
        dones = [False] * self.n_agents
        infos = {}

        return obs, rewards, dones, infos

    def reset(self):
        self.eng.reset()
        obs = self.ob_callback.get_obs()
        return obs


    