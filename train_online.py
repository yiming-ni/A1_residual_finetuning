#! /usr/bin/env python
import os
import pickle
import shutil

import numpy as np
import tqdm
import time

import gym
import wandb
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from rl.agents import SACLearner
from rl.data import ReplayBuffer
from rl.evaluation import evaluate
from rl.wrappers import wrap_gym
from misc import RecordEpisodeStatisticsWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'A1Run-v0', 'Environment name.')
flags.DEFINE_string('exp_group', 'default', 'Experiment group name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_boolean('just_eval', False, 'Only evaluate a policy.')
flags.DEFINE_boolean('test_base', False, 'Only evaluate the base policy on the real robot.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_integer('ep_len', 1000, 'Episode Length.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('wandb', True, 'Log wandb.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_float('action_filter_high_cut', None, 'Action filter high cut.')
flags.DEFINE_integer('action_history', 1, 'Action history.')
flags.DEFINE_integer('control_frequency', 33, 'Control frequency.')
flags.DEFINE_integer('utd_ratio', 1, 'Update to data ratio.')
flags.DEFINE_boolean('real_robot', False, 'Use real robot.')
flags.DEFINE_boolean('just_render', False, 'Just render.')
flags.DEFINE_string('load_dir', '', 'Directory of model and buffer to load from.')
flags.DEFINE_float('residual_scale', 0.1, 'Defines the residual action range.')
flags.DEFINE_float('energy_weight', 0.0, 'Weightage for energy reward term.')
flags.DEFINE_string('object_type', 'sphere', 'Type of the object: sphere, box, cylinder.')
flags.DEFINE_list('object_size', ['0.097'], 'Size of the regular shaped object.')
flags.DEFINE_boolean('sparse_reward', False, 'Whether to use sparse distance reward.')
flags.DEFINE_boolean('randomize_object', False, 'Whether to randomize the object params in each episode.')
flags.DEFINE_boolean('separate_log', False, 'Whether to separately log rewards and energy penalty.')
flags.DEFINE_boolean('ep_update', False, 'Whether to use episode update for the agent (default is step udpate).')
flags.DEFINE_string('receive_ip', '127.0.0.1', 'Receive IP Address')
flags.DEFINE_string('send_ip', '127.0.0.1', 'Send IP Address')
flags.DEFINE_integer('receive_port', 8001, 'Receive port')
flags.DEFINE_integer('send_port', 8000, 'Send port')
flags.DEFINE_bool('real_ball', False, 'Whether to use a real ball through cameras.')
flags.DEFINE_bool('real_pos', False, 'Whether to use a real robot odometry.')
flags.DEFINE_bool('fov', False, 'Whether to limit camera field of view.')
config_flags.DEFINE_config_file(
    'config',
    'configs/sac_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def evaluate_with_video(agent, env: gym.Env, num_episodes: int, zero_acs: bool = False):
    videos = []

    for ep in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            if ep == 0:
                img = env.render(mode='rgb_array', width=128, height=128)
                videos.append(img)

            if zero_acs:
                action = np.zeros(12)
            else:
                action = agent.eval_actions(observation)

            observation, _, done, _ = env.step(action)

    eval_info = {
        'return': np.mean(env.return_queue),
        'length': np.mean(env.length_queue),
        'video': wandb.Video(np.stack(videos).transpose(0, 3, 1, 2), fps=33, format="gif")
    }
    return eval_info

def eval_only(agent, env, num_episodes, zero_acs=False):
    videos = []
    returns = []
    for ep in range(num_episodes):
        observation, done = env.reset(), False
        episodic_return = 0
        while not done:
            if ep == 0:
                img = env.render(mode='rgb_array', width=128, height=128)
                videos.append(img)

            if zero_acs:
                action = np.zeros(12)

            else: action = agent.eval_actions(observation)

            observation, reward, done, _ = env.step(action)
            episodic_return += reward
        returns.append(episodic_return)

    eval_info = {
        'return': np.mean(returns),
        'return_std': np.std(returns),
        'video': wandb.Video(np.stack(videos).transpose(0, 3, 1, 2), fps=33, format="gif")
    }
    return eval_info


def main(_):
    wandb.init(project='a1',
               group=FLAGS.exp_group,
               dir=os.getenv('WANDB_LOGDIR'))
    if FLAGS.ep_update:
        exp_name = FLAGS.exp_group + '_epu' + str(FLAGS.object_size[0]) + FLAGS.object_type
    else:
        exp_name = FLAGS.exp_group + '_stpu' + str(FLAGS.object_size[0]) + FLAGS.object_type
    wandb.run.name = exp_name
    wandb.run.save()
    wandb.config.update(FLAGS)

    object_params = {'object_type': FLAGS.object_type,
                     'object_size': FLAGS.object_size}

    if FLAGS.real_robot:
        from real.envs.locomotion_gym_env_finetune import LocomotionGymEnv
        from real.envs.env_wrappers.residual import ResidualWrapper
        env = LocomotionGymEnv(recv_IP=FLAGS.receive_ip, recv_port=FLAGS.receive_port, send_IP=FLAGS.send_ip, send_port=FLAGS.send_port, episode_length=FLAGS.ep_len, real_ball=FLAGS.real_ball, real_pos=FLAGS.real_pos)
        env = ResidualWrapper(env, real_robot=True, residual_scale=FLAGS.residual_scale, action_history=FLAGS.action_history)

        if FLAGS.test_base:
            for ep in range(10):
                env.reset()
                for timestep in range(1000):
                    env.step(np.zeros(12))
            print("finished running base policy. Exit.")
            return
    else:
        from env_utils import make_mujoco_env
        env = make_mujoco_env(
            FLAGS.env_name,
            fov=FLAGS.fov,
            separate_log=FLAGS.separate_log,
            sparse_reward=FLAGS.sparse_reward,
            ep_len=FLAGS.ep_len,
            object_params=object_params,
            randomize_object=FLAGS.randomize_object,
            residual_scale=FLAGS.residual_scale,
            energy_weight=FLAGS.energy_weight,
            control_frequency=FLAGS.control_frequency,
            action_filter_high_cut=FLAGS.action_filter_high_cut,
            action_history=FLAGS.action_history)

    env = wrap_gym(env, rescale_actions=False)
    if not FLAGS.separate_log:
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    else:
        env = RecordEpisodeStatisticsWrapper(env, deque_size=1)

    env.seed(FLAGS.seed)
    if FLAGS.just_render:
        import imageio
        video = []
        for ep in range(10):
            env.reset()
            # ball_pos = env.env.env.env.env.env.env._env.task._ball_frame.pos
            # ball_x, ball_y, ball_z = np.around(ball_pos[0], 2), np.around(ball_pos[1], 2), np.around(ball_pos[2], 2)
            # print(ball_pos[2])

            for i in range(20):
                video.append(env.render(mode='rgb_array'))
                # obs, reward, next_obs, done = env.step(gym.spaces.Box(-1., 1., shape=env.action_space.low.shape, dtype=np.float32).sample())
                obs, reward, next_obs, done = env.step(np.zeros(12))
                # env.render(mode='rgb_array')
                if done:
                    break
            # for _ in range(5):
            #     video.append(np.zeros_like(video[-1]))
                ball_pos = env.env.env.env.env.env.env._env.task._floor._ball.mjcf_model.find('body', 'ball').pos
                ball_x, ball_y, ball_z = np.around(ball_pos[0], 2), np.around(ball_pos[1], 2), np.around(ball_pos[2], 2)
                # print(ball_pos[2])
            imageio.mimsave('ballvisualization'+str(ball_x) + '-' + str(ball_y) + "-" + str(ball_z) +'.gif', video)
            # imageio.mimsave('ballvisualization' +'.gif', video)
            video=[]
        1/0

    if not FLAGS.real_robot:
        eval_env = make_mujoco_env(
            FLAGS.env_name,
            fov=FLAGS.fov,
            separate_log=FLAGS.separate_log,
            sparse_reward=FLAGS.sparse_reward,
            ep_len=FLAGS.ep_len,
            object_params=object_params,
            randomize_object=FLAGS.randomize_object,
            residual_scale=FLAGS.residual_scale,
            energy_weight=FLAGS.energy_weight,
            control_frequency=FLAGS.control_frequency,
            action_filter_high_cut=FLAGS.action_filter_high_cut,
            action_history=FLAGS.action_history)

        eval_env = wrap_gym(eval_env, rescale_actions=False)
        eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env, deque_size=FLAGS.eval_episodes)
        eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    agent = SACLearner.create(FLAGS.seed, env.observation_space,
                              env.action_space, **kwargs)

    chkpt_dir = os.path.join('runs', os.path.join(FLAGS.log_dir, os.path.join(exp_name, 'checkpoints')))
    os.makedirs(chkpt_dir, exist_ok=True)
    buffer_dir = os.path.join('runs', os.path.join(FLAGS.log_dir, os.path.join(exp_name, 'buffers')))

    if FLAGS.load_dir:
        load_chkpt_dir = os.path.join(FLAGS.load_dir, 'checkpoints')
        load_buffer_dir = os.path.join(FLAGS.load_dir, 'buffers')
        last_checkpoint = checkpoints.latest_checkpoint(load_chkpt_dir)
    else:
        last_checkpoint = None

    if last_checkpoint is None:
        start_i = 0
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                     FLAGS.max_steps)
        replay_buffer.seed(FLAGS.seed)
    else:
        start_i = int(last_checkpoint.split('_')[-1])

        agent = checkpoints.restore_checkpoint(last_checkpoint, agent)
        try:
            with open(os.path.join(load_buffer_dir, 'buffer'), 'rb') as f:
                replay_buffer = pickle.load(f)
        except:
            print("Failed to load buffer from", load_buffer_dir)

    if FLAGS.just_eval:
        base_eval_info = eval_only(agent, eval_env, num_episodes=FLAGS.eval_episodes, zero_acs=True)
        wandb.log({f'evaluation/{"base_video"}': base_eval_info['video']}, step=0)
        eval_info = eval_only(agent, eval_env, num_episodes=FLAGS.eval_episodes)
        wandb.log({f'evaluation/{"trained_video"}': eval_info['video']}, step=0)
        print('=====base policy evaluation=====')
        print('return mean: {} \nreturn std: {}'.format(base_eval_info['return'], base_eval_info['return_std']))
        print('=====trained policy evaluation=====')
        print('return mean: {} \nreturn std: {}'.format(eval_info['return'], eval_info['return_std']))
        return

    videos = []
    observation, done = env.reset(), False
    save_training_video = False
    if not FLAGS.real_robot:
        save_training_video = True
    for i in tqdm.tqdm(range(start_i, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        curr_start = 0
        if i == start_i == 0:
            if not FLAGS.real_robot:
                if FLAGS.save_video:
                    eval_info = evaluate_with_video(agent,
                                         eval_env,
                                         num_episodes=FLAGS.eval_episodes,
                                         zero_acs=True)
                for k, v in eval_info.items():
                    wandb.log({f'evaluation/{k}': v}, step=i)

        if FLAGS.ep_update:
            if i < FLAGS.start_training:
                action = gym.spaces.Box(-1., 1., shape=env.action_space.low.shape, dtype=np.float32).sample()
            else:
                action, agent = agent.sample_actions(observation)
        else:
            if i < FLAGS.start_training:
                # action = env.action_space.sample()
                action = gym.spaces.Box(-1., 1., shape=env.action_space.low.shape, dtype=np.float32).sample()
            else:
                action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if save_training_video:
            img = env.render(mode='rgb_array', width=128, height=128)
            videos.append(img)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(observations=observation,
                 actions=action,
                 rewards=reward,
                 masks=mask,
                 dones=done,
                 next_observations=next_observation))
        observation = next_observation

        if len(videos) > 30*20:
            train_videos = {
                'video': wandb.Video(np.stack(videos).transpose(0, 3, 1, 2), fps=33, format="gif")
            }
            wandb.log({f'training/{"video"}': train_videos['video']}, step=i)
            videos = []
            save_training_video = False

        if done:
            
            observation, done = env.reset(), False
            # start episode update
            if FLAGS.ep_update:
                print("Start updating...")
                
                if i >= FLAGS.start_training:
                    
                    for _ in range(i - curr_start): # actual length of episode
                        batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
                        agent, update_info = agent.update(batch, FLAGS.utd_ratio)
                    for k, v in update_info.items():
                            wandb.log({f'training/{k}': v}, step=i)
            # end episode update
            for k, v in info['episode'].items():
                decode = {'r': 'return', 'dr': 'distance_return', 'er': 'energy_return', 'l': 'length', 't': 'time'}
                wandb.log({f'training/{decode[k]}': v}, step=i)

            curr_start = i

        if not FLAGS.ep_update:
            # start step update
            if i >= FLAGS.start_training:
                batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
                agent, update_info = agent.update(batch, FLAGS.utd_ratio)

                if i % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        wandb.log({f'training/{k}': v}, step=i)
            # end step update

        if i % FLAGS.eval_interval == 1:
            if not FLAGS.real_robot:
                save_training_video = True
                if FLAGS.save_video:
                    eval_info = evaluate_with_video(agent,
                                         eval_env,
                                         num_episodes=FLAGS.eval_episodes)
                else:
                    eval_info = evaluate(agent,
                                     eval_env,
                                     num_episodes=FLAGS.eval_episodes)
                for k, v in eval_info.items():
                    wandb.log({f'evaluation/{k}': v}, step=i)

            checkpoints.save_checkpoint(chkpt_dir,
                                        agent,
                                        step=i + 1,
                                        keep=20,
                                        overwrite=True)

            try:
                shutil.rmtree(buffer_dir)
            except:
                pass

            os.makedirs(buffer_dir, exist_ok=True)
            with open(os.path.join(buffer_dir, 'buffer'), 'wb') as f:
                pickle.dump(replay_buffer, f)


if __name__ == '__main__':
    app.run(main)
