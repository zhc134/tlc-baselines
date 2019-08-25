import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import RLAgent
from metric import TravelTimeMetric

world = World("examples/config.json", thread_num=1)

agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i["trafficLight"]["lightphases"]))
    agents.append(RLAgent(
        action_space, 
        LaneVehicleGenerator(world, i["id"], ["lane_count"], in_only=True),
        LaneVehicleGenerator(world, i["id"], ["lane_waiting_count"], in_only=True, average="all", negative=True)
    ))

metric = TravelTimeMetric(world)
env = TSCEnv(world, agents, metric)

obs = env.reset()
for i in range(100):
    if i % 5 == 0:
        actions = env.action_space.sample()
    obs, rewards, dones, info = env.step(actions)
    #print(obs)
    print(rewards)
    print(info["metric"])