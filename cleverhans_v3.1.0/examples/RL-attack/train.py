import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import tempfile
import time
import json
import random

import rlattack.common.tf_util as U

from rlattack import logger
from rlattack import deepq
from rlattack.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from rlattack.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
    SimpleMonitor
)
from rlattack.common.schedules import LinearSchedule, PiecewiseSchedule
# when updating this to non-deprecated ones, it is important to
# copy over LazyFrames
from rlattack.common.atari_wrappers_deprecated import wrap_dqn
from rlattack.common.azure_utils import Container
from model import model, dueling_model
from statistics import statistics


def parse_args():
  parser = argparse.ArgumentParser("DQN experiments for Atari games")
  # Environment
  parser.add_argument("--env", type=str, default="Pong",
                      help="name of the game")
  parser.add_argument("--seed", type=int, default=42,
                      help="which seed to use")
  # Core DQN parameters
  parser.add_argument("--replay-buffer-size", type=int, default=int(1e6),
                      help="replay buffer size")
  parser.add_argument("--lr", type=float, default=1e-4,
                      help="learning rate for Adam optimizer")
  parser.add_argument("--num-steps", type=int, default=int(2e8),
                      help="total number of steps to \
                        run the environment for")
  parser.add_argument("--batch-size", type=int, default=32,
                      help="number of transitions to optimize \
                        at the same time")
  parser.add_argument("--learning-freq", type=int, default=4,
                      help="number of iterations between \
                        every optimization step")
  parser.add_argument("--target-update-freq", type=int, default=40000,
                      help="number of iterations between \
                        every target network update")
  # Bells and whistles
  boolean_flag(parser, "noisy", default=False,
               help="whether or not to NoisyNetwork")
  boolean_flag(parser, "double-q", default=True,
               help="whether or not to use double q learning")
  boolean_flag(parser, "dueling", default=False,
               help="whether or not to use dueling model")
  boolean_flag(parser, "prioritized", default=False,
               help="whether or not to use prioritized replay buffer")
  parser.add_argument("--prioritized-alpha", type=float, default=0.6,
                      help="alpha parameter for prioritized replay buffer")
  parser.add_argument("--prioritized-beta0", type=float, default=0.4,
                      help="initial value of beta \
                        parameters for prioritized replay")
  parser.add_argument("--prioritized-eps", type=float, default=1e-6,
                      help="eps parameter for prioritized replay buffer")
  # Checkpointing
  parser.add_argument("--save-dir", type=str, default=None, required=True,
                      help="directory in which \
                        training state and model should be saved.")
  parser.add_argument("--save-azure-container", type=str, default=None,
                      help="It present data will saved/loaded from Azure. \
                        Should be in format ACCOUNT_NAME:ACCOUNT_KEY:\
                        CONTAINER")
  parser.add_argument("--save-freq", type=int, default=1e6,
                      help="save model once every time this many \
                        iterations are completed")
  boolean_flag(parser, "load-on-start", default=True,
               help="if true and model was previously saved then training \
                 will be resumed")

  # V: Attack Arguments #
  parser.add_argument("--attack", type=str, default=None,
                      help="Method to attack the model.")
  parser.add_argument("--attack-init", type=int, default=0,
                      help="Iteration no. to begin attacks")
  parser.add_argument("--attack-prob", type=float, default=0.0,
                      help="Probability of attack at each step, \
                        float in range 0 - 1.0")
  return parser.parse_args()


def make_env(game_name):
  env = gym.make(game_name + "NoFrameskip-v4")
  monitored_env = SimpleMonitor(env)
  env = wrap_dqn(monitored_env)
  return env, monitored_env


def maybe_save_model(savedir, container, state):
  if savedir is None:
    return
  start_time = time.time()
  model_dir = "model-{}".format(state["num_iters"])
  U.save_state(os.path.join(savedir, model_dir, "saved"))
  if container is not None:
    container.put(os.path.join(savedir, model_dir), model_dir)
  relatively_safe_pickle_dump(state,
                              os.path.join(savedir,
                                           'training_state.pkl.zip'),
                              compression=True)
  if container is not None:
    container.put(os.path.join(savedir, 'training_state.pkl.zip'),
                  'training_state.pkl.zip')
  relatively_safe_pickle_dump(state["monitor_state"],
                              os.path.join(savedir, 'monitor_state.pkl'))
  if container is not None:
    container.put(os.path.join(savedir, 'monitor_state.pkl'),
                  'monitor_state.pkl')
  logger.log("Saved model in {} seconds\n".format(time.time() - start_time))


def maybe_load_model(savedir, container):
  """Load model if present at the specified path."""
  if savedir is None:
    return

  state_path = os.path.join(os.path.join(savedir, 'training_state.pkl.zip'))
  if container is not None:
    logger.log("Attempting to download model from Azure")
    found_model = container.get(savedir, 'training_state.pkl.zip')
  else:
    found_model = os.path.exists(state_path)
  if found_model:
    state = pickle_load(state_path, compression=True)
    model_dir = "model-{}".format(state["num_iters"])
    if container is not None:
      container.get(savedir, model_dir)
    U.load_state(os.path.join(savedir, model_dir, "saved"))
    logger.log("Loaded models checkpoint at {} iterations".format(
        state["num_iters"]))
    return state


if __name__ == '__main__':
  args = parse_args()
  # Parse savedir and azure container.
  savedir = args.save_dir
  if args.save_azure_container is not None:
    account_name, account_key, container_name = \
        args.save_azure_container.split(":")
    container = Container(
        account_name=account_name,
        account_key=account_key,
        container_name=container_name,
        maybe_create=True
    )
    if savedir is None:
      # Careful! This will not get cleaned up.
      savedir = tempfile.TemporaryDirectory().name
  else:
    container = None
  # Create and seed the env.
  env, monitored_env = make_env(args.env)
  if args.seed > 0:
    set_global_seeds(args.seed)
    env.unwrapped.seed(args.seed)

  # V: Save arguments, configure log dump path to savedir #
  if savedir:
    with open(os.path.join(savedir, 'args.json'), 'w') as f:
      json.dump(vars(args), f)
    logger.configure(dir=savedir)  # log to savedir

  with U.make_session(4) as sess:
    # Create training graph and replay buffer
    act, train, update_target, debug, craft_adv = deepq.build_train(
        make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape,
                                              name=name),
        q_func=dueling_model if args.dueling else model,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=args.lr,
                                         epsilon=1e-4),
        gamma=0.99,
        grad_norm_clipping=10,
        double_q=args.double_q,
        noisy=args.noisy,
        attack=args.attack
    )
    approximate_num_iters = args.num_steps / 4
    exploration = PiecewiseSchedule([
        (0, 1.0),
        (approximate_num_iters / 50, 0.1),
        (approximate_num_iters / 5, 0.01)
    ], outside_value=0.01)

    if args.prioritized:
      replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size,
                                              args.prioritized_alpha)
      beta_schedule = LinearSchedule(approximate_num_iters,
                                     initial_p=args.prioritized_beta0,
                                     final_p=1.0)
    else:
      replay_buffer = ReplayBuffer(args.replay_buffer_size)

    U.initialize()
    update_target()
    num_iters = 0

    # Load the model
    state = maybe_load_model(savedir, container)
    if state is not None:
      num_iters, replay_buffer = state["num_iters"], state[
          "replay_buffer"],
      monitored_env.set_state(state["monitor_state"])

    start_time, start_steps = None, None
    steps_per_iter = RunningAvg(0.999)
    iteration_time_est = RunningAvg(0.999)
    obs = env.reset()
    # Record the mean of the \sigma
    sigma_name_list = []
    sigma_list = []
    for param in tf.trainable_variables():
      # only record the \sigma in the action network
      if 'sigma' in param.name \
              and 'deepq/q_func/action_value' in param.name:
        summary_name = \
            param.name.replace(
                'deepq/q_func/action_value/', '').replace(
                    '/', '.').split(':')[0]
        sigma_name_list.append(summary_name)
        sigma_list.append(tf.reduce_mean(tf.abs(param)))
    f_mean_sigma = U.function(inputs=[], outputs=sigma_list)
    # Statistics
    writer = tf.summary.FileWriter(savedir, sess.graph)
    im_stats = statistics(scalar_keys=['action', 'im_reward', 'td_errors',
                                       'huber_loss'] + sigma_name_list)
    ep_stats = statistics(scalar_keys=['ep_reward', 'ep_length'])
    # Main trianing loop
    ep_length = 0
    while True:
      num_iters += 1
      ep_length += 1

      # V: Perturb observation if we are past the init stage
      # and at a designated attack step
      # if craft_adv != None and (num_iters >= args.attack_init)
      # and ((num_iters - args.attack_init) % args.attack_freq == 0) :
      if craft_adv is not None and (num_iters >= args.attack_init) and (
              random.random() <= args.attack_prob):
        obs = craft_adv(np.array(obs)[None])[0]

      # Take action and store transition in the replay buffer.
      if args.noisy:
        # greedily choose
        action = act(np.array(obs)[None], stochastic=False)[0]
      else:
        # epsilon greedy
        action = act(np.array(obs)[None],
                     update_eps=exploration.value(num_iters))[0]
      new_obs, rew, done, info = env.step(action)
      replay_buffer.add(obs, action, rew, new_obs, float(done))
      obs = new_obs
      if done:
        obs = env.reset()

      if (num_iters > max(5 * args.batch_size,
                          args.replay_buffer_size // 20) and
              num_iters % args.learning_freq == 0):
        # Sample a bunch of transitions from replay buffer
        if args.prioritized:
          experience = replay_buffer.sample(args.batch_size,
                                            beta=beta_schedule.value(
                                                num_iters))
          (obses_t, actions, rewards, obses_tp1, dones, weights,
           batch_idxes) = experience
        else:
          obses_t, actions, rewards, obses_tp1, dones = \
              replay_buffer.sample(args.batch_size)
          weights = np.ones_like(rewards)
        # Minimize the error in Bellman's and compute TD-error
        td_errors, huber_loss = train(obses_t, actions, rewards,
                                      obses_tp1, dones, weights)
        # Update the priorities in the replay buffer
        if args.prioritized:
          new_priorities = np.abs(td_errors) + args.prioritized_eps
          replay_buffer.update_priorities(
              batch_idxes, new_priorities
          )
        # Write summary
        mean_sigma = f_mean_sigma()
        im_stats.add_all_summary(writer,
                                 [action, rew, np.mean(td_errors),
                                  np.mean(huber_loss)] + mean_sigma,
                                 num_iters)

      # Update target network.
      if num_iters % args.target_update_freq == 0:
        update_target()

      if start_time is not None:
        steps_per_iter.update(info['steps'] - start_steps)
        iteration_time_est.update(time.time() - start_time)
      start_time, start_steps = time.time(), info["steps"]

      # Save the model and training state.
      if num_iters > 0 and (num_iters % args.save_freq == 0 or info[
              "steps"] > args.num_steps):
        maybe_save_model(savedir, container, {
            'replay_buffer': replay_buffer,
            'num_iters': num_iters,
            'monitor_state': monitored_env.get_state()
        })

      if info["steps"] > args.num_steps:
        break

      if done:
        steps_left = args.num_steps - info["steps"]
        completion = np.round(info["steps"] / args.num_steps, 1)
        mean_ep_reward = np.mean(info["rewards"][-100:])
        logger.record_tabular("% completion", completion)
        logger.record_tabular("steps", info["steps"])
        logger.record_tabular("iters", num_iters)
        logger.record_tabular("episodes", len(info["rewards"]))
        logger.record_tabular("reward (100 epi mean)",
                              np.mean(info["rewards"][-100:]))
        if not args.noisy:
          logger.record_tabular("exploration",
                                exploration.value(num_iters))
        if args.prioritized:
          logger.record_tabular("max priority",
                                replay_buffer._max_priority)
        fps_estimate = (
            float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
            if steps_per_iter._value is not None else "calculating:")
        logger.dump_tabular()
        logger.log()
        logger.log("ETA: " +
                   pretty_eta(int(steps_left / fps_estimate)))
        logger.log()
        # add summary for one episode
        ep_stats.add_all_summary(writer, [mean_ep_reward, ep_length],
                                 num_iters)
        ep_length = 0
