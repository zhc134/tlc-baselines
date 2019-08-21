from environment import TSCEnv
from world import World
from generator import LaneVehicle

world = World("examples/config.json", thread_num=1)

env = TSCEnv(
    world,
    ob_generator=LaneVehicle(world, ["count"], in_only=True),
    reward_generator=LaneVehicle(world, ["waiting_count"], in_only=True, average="all", negative=True)
)

for _ in range(100):
    obs, rewards, dones, info = env.step(env.action_space.sample())
    print(obs)
    print(rewards)