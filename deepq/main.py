from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.deepq.noise import *

import gym

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=0)
    #parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)
    env = bench.Monitor(env, logger.get_dir())
    v_func = deepq.models.mlp(
        hiddens=[200, 200],
    )
    l_func = deepq.models.mlp(
        hiddens=[200, 200],
    )
    mu_func = deepq.models.mlp(
        hiddens=[200, 200],
    )

    stddev = 0.3
    nb_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))

    act = deepq.learn(
        env,
        models=[mu_func, v_func, l_func],
        action_noise=action_noise,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized)
    )
    # act.save("pong_model.pkl") XXX
    env.close()


if __name__ == '__main__':
    main()
