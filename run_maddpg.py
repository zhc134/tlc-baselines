import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent.maddpg_agent import MADDPGAgent
from metric import TravelTimeMetric
import argparse
import tensorflow as tf
import os

# parse args
def parse_args():
    parser = argparse.ArgumentParser(description='Run Example')
    # Environment
    parser.add_argument('config_file', type=str, help='path of config file')
    parser.add_argument('--thread', type=int, default=1, help='number of threads')
    parser.add_argument('--steps', type=int, default=100, help='number of steps')
    parser.add_argument('--action_interval', type=int, default=1, help='how often agent make decisions')
    parser.add_argument('--episodes', type=int, default=500, help='training episodes')
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="model/maddpg", help="directory in which model should be saved")
    parser.add_argument("--save-rate", type=int, default=10, help="save model once every time this many episodes are completed")
    return parser.parse_args()
args = parse_args()

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(MADDPGAgent(
        action_space,
        LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
        LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
        args,
        i.id
    ))
ob_space_n = []
action_space_n = []
for agent in agents:
    ob_space_n.append(agent.ob_shape)
    action_space_n.append(agent.action_space)
print(ob_space_n)
print(action_space_n)
for i, agent in enumerate(agents):
    agent.build_model(ob_space_n, action_space_n, i)

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)

# train maddpg_agent
def train():
    config = tf.ConfigProto(
        intra_op_parallelism_threads=4,
        allow_soft_placement=True,
        log_device_placement=False
    )
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    with sess:
        # Initialize
        sess.run(tf.variables_initializer(tf.global_variables()))
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        train_step = 0

        print('Starting iterations...')
        for e in range(args.episodes):
            episode_rewards = [0.0]  # sum of rewards for all agents
            agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
            obs_n = env.reset()
            episode_step = 0
            step = 0
            while step < args.steps:
                if step % args.action_interval == 0:
                    # get action
                    action_n = [agent.get_action(obs) for agent, obs in zip(agents, obs_n)]
                    action_prob_n = [agent.get_action_prob(obs) for agent, obs in zip(agents, obs_n)]
                    # environment step
                    for _ in range(args.action_interval):
                        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                        step += 1

                    episode_step += 1
                    # collect experience
                    for i, agent in enumerate(agents):
                        agent.experience(obs_n[i], action_prob_n[i], rew_n[i], new_obs_n[i], done_n[i])
                    obs_n = new_obs_n

                    for i, rew in enumerate(rew_n):
                        episode_rewards[-1] += rew
                        agent_rewards[i][-1] += rew

                    # increment global step counter
                    train_step += 1

                    # update all trainers, if not in display or benchmark mode
                    loss = None
                    for agent in agents:
                        loss = agent.update(agents, train_step)
                        # print(loss)
                        # if loss is not None:
                        #     print(loss[0], loss[1])

            print("episode:{}/{}, total agent episode mean reward:{}".format(e, args.episodes, episode_rewards[0]/episode_step))
            for i in range(len(agents)):
                print("agent:{}, episode mean reward:{}".format(i, agent_rewards[i][-1]/episode_step))
            if e % args.save_rate == 0:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                saver.save(sess, os.path.join(args.save_dir, "maddpg_{}.ckpt".format(e)))

def test(model_id=None):
    sess = tf.Session()
    with sess:
        # Initialize
        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint(args.save_dir)
        if model_id is not None:
            saver.restore(sess, "model/maddpg/maddpg_{}.ckpt".format(model_id))
        else:
            saver.restore(sess, model_file)
        obs_n = env.reset()
        for i in range(args.steps):
            if i % args.delta_t == 0:
                # get action
                action_n = [agent.get_action(obs) for agent, obs in zip(agents, obs_n)]
                # environment step
                obs_n, rew_n, done_n, info_n = env.step(action_n)
                done = all(done_n)
                if done:
                    break
        print("Final Travel Time is %.4f" % env.metric.update(done=True))

# simulate
train()
#test()
