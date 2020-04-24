import json
import gym
from environment import TSCEnv
from world import World
from agent import Fixedtime_Agent
from metric import TravelTimeMetric
import argparse

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('signal_plan_address', type=str, help='path of signal plan folder')
parser.add_argument('signal_plan_prefix', type=str, help='prefix of signal plan file')
parser.add_argument('single_inter', type=str, help='if threre is only one intersection')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
args = parser.parse_args()

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(Fixedtime_Agent(action_space, args.signal_plan_address, args.signal_plan_prefix, i.id, int(args.single_inter)))

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)

# simulate
obs = env.reset()
for i in range(args.steps):
    actions = []
    try:
        for agent in agents:
            actions.append(agent.get_action(world))
        obs, rewards, dones, info = env.step(actions)
        print(i, actions)
    except:
        break

print(env.eng.get_average_travel_time())