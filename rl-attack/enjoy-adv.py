""" DQN - Test-time attacks

============ Sample usage ============ 

No attack, testing a DQN model of Breakout trained without parameter noise:
$> python3 enjoy-adv.py --env Breakout --model-dir ./data/Breakout/model-173000 --video ./Breakout.mp4

No attack, testing a DQN model of Breakout trained with parameter noise (NoisyNet implementation):
$> python3 enjoy-adv.py --env Breakout --noisy --model-dir ./data/Breakout/model-173000 --video ./Breakout.mp4

Whitebox FGSM attack, testing a DQN model of Breakout trained without parameter noise:
$> python3 enjoy-adv.py --env Breakout --model-dir ./data/Breakout/model-173000 --attack fgsm --video ./Breakout.mp4

Whitebox FGSM attack, testing a DQN model of Breakout trained with parameter noise (NoisyNet implementation):
$> python3 enjoy-adv.py --env Breakout --noisy --model-dir ./data/Breakout/model-173000 --attack fgsm --video ./Breakout.mp4

Blackbox FGSM attack, testing a DQN model of Breakout trained without parameter noise:
$> python3 enjoy-adv.py --env Breakout --model-dir ./data/Breakout/model-173000 --attack fgsm --blackbox --model-dir2 ./data/Breakout/model-173000-2 --video ./Breakout.mp4

Blackbox FGSM attack, testing a DQN model of Breakout trained with parameter noise (NoisyNet implementation), replica model trained without parameter noise:
$> python3 enjoy-adv.py --env Breakout --noisy --model-dir ./data/Breakout/model-173000 --attack fgsm --blackbox --model-dir2 ./data/Breakout/model-173000-2 --video ./Breakout.mp4

Blackbox FGSM attack, testing a DQN model of Breakout trained with parameter noise (NoisyNet implementation), replica model trained with parameter noise:
$> python3 enjoy-adv.py --env Breakout --noisy --model-dir ./data/Breakout/model-173000 --attack fgsm --blackbox --model-dir2 ./data/Breakout/model-173000-2 --noisy2 --video ./Breakout.mp4

"""

import argparse
import gym
import os
import numpy as np
import random

#from gym.monitoring import VideoRecorder
from gym import wrappers
from time import time
import rlattack.common.tf_util as U

from rlattack import logger
from rlattack import deepq
from rlattack.common.misc_util import (
	boolean_flag,
	SimpleMonitor,
)
from rlattack.common.atari_wrappers_deprecated import wrap_dqn
#from rlattack.deepq.experiments.atari.model import model, dueling_model

#V: imports#
import tensorflow as tf
import cv2
from collections import deque
from model import model, dueling_model
from statistics import statistics

from cleverhans.plot import pyplot_image, image


import matplotlib.pyplot as plt
from PIL import Image



class DQNModel:
	"""
	Creating Q-graph, FGSM graph
	Supports loading multiple graphs - needed for blackbox attacks
	"""

	def __init__(self, env, dueling, noisy, fname):
		self.g = tf.Graph()
		self.noisy = noisy
		self.dueling = dueling 
		self.env = env
		with self.g.as_default():
			self.act = deepq.build_act_enjoy(
				make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
				q_func=dueling_model if dueling else model,
				num_actions=env.action_space.n,
				noisy=noisy
				)
			self.saver = tf.train.Saver()
		self.sess = tf.Session(graph=self.g)	
		
		if fname is not None:
			print ('Loading Model...')
			self.saver.restore(self.sess, fname)
	
	def get_act(self):
		return self.act
		
	def get_session(self):
		return self.sess 
	
	def craft_adv(self):
		with self.sess.as_default():
			with self.g.as_default():
				craft_adv_obs = deepq.build_adv(
								make_obs_tf=lambda name: U.Uint8Input(self.env.observation_space.shape, name=name),
								q_func=dueling_model if self.dueling else model,
								num_actions=self.env.action_space.n,
								epsilon = 1.0/255.0,
								noisy=self.noisy, attack = args.attack
								)
		return craft_adv_obs
		
			
	
def parse_args():
	parser = argparse.ArgumentParser("Run an already learned DQN model.")
	# Environment
	parser.add_argument("--env", type=str, required=True, help="name of the game")
	parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
	parser.add_argument("--video", type=str, default=None, help="Path to mp4 file where the video of first episode will be recorded.")
	boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
	boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
	#V: Attack Arguments#
	parser.add_argument("--model-dir2", type=str, default=None, help="load adversarial model from this directory (blackbox attacks). ")
	parser.add_argument("--attack", type=str, default=None, help="Method to attack the model.")
	parser.add_argument("--attack-prob", type=float, default=0.0, help="attack probability")
	boolean_flag(parser, "noisy", default=False, help="whether or not to NoisyNetwork")
	boolean_flag(parser, "noisy2", default=False, help="whether or not to NoisyNetwork")
	boolean_flag(parser, "blackbox", default=False, help="whether or not to NoisyNetwork")
	return parser.parse_args()


def make_env(game_name):
	env = gym.make(game_name + "NoFrameskip-v4")
	monitored_env = SimpleMonitor(env)
	env = wrappers.Monitor(env, './videos/' + str(time()) + '/')
	env = wrap_dqn(monitored_env)
	return env,monitored_env


def play(env, act, craft_adv_obs, craft_adv_obs2, stochastic, video_path, attack, m_target, m_adv):
    num_episodes = 0
    num_moves = 0
    num_transfer = 0
    obs = env.reset()
    
   # only in case for Pong! If -env Seaquest, then comment the lines with #Seaquest
	actiondict = {0:"NOOP", 1:"NOOP", 2:"UP", 3:"DOWN", 4:"UP", 5:"DOWN"} #Seaquest
	titles = ["original", "perturbation", "adversarial"] #Seaquest


    while num_episodes<100:

	#V: Attack #
        if attack != None and (random.random() <= args.attack_prob):
			# Craft adv. examples
            with m_adv.get_session().as_default():
                adv_obs = craft_adv_obs(np.array(obs)[None], stochastic_adv=stochastic)[0]


            with m_target.get_session().as_default():
                action = act(np.array(adv_obs)[None], stochastic=stochastic)[0]
                action2 = act(np.array(obs)[None], stochastic=stochastic)[0]
                num_moves += 1
                if (action != action2):
                    num_transfer += 1 
                    
                    
            org_str = "action: " + actiondict[action]#Seaquest
            adv_str = "action: " + actiondict[action2]#Seaquest
            
            diff = np.array(adv_obs)[None][0][:, :, 0] - np.array(obs)[None][0][:, :, 0]#Seaquest
            diff = np.abs(diff)#Seaquest
            print("maximum value in perturbation: ", np.max(diff))#Seaquest


			#print(env.unwrapped.get_action_meanings())
            if actiondict[action] != actiondict[action2]:#Seaquest
                
                fig = pyplot_image.pair_visual(np.array(obs)[None][0][:, :, 0], np.array(adv_obs)[None][0][:, :, 0])#Seaquest
                #Seaquest
                for index, ax in enumerate(fig.axes):
                    if index == 0:
                        ax.text(10, 100, org_str)
                        ax.set_title("original")
                    elif index == 1:
                        ax.set_title("perturbation")
                    elif index == 2:
                        ax.set_title("perturbed image")
                        ax.text(10, 100, adv_str)
                        
                fig.savefig('pics/cwl2-eps1-' + str(num_moves) +'.png')

        else:
			# Normal
            action = act(np.array(obs)[None], stochastic=stochastic)[0]
        
        obs, rew, done, info = env.step(action)
        
        if done:
            obs = env.reset()
            
        if done:
            if len(info["rewards"]) > num_episodes:
                #print((info["rewards"]))
                episode_rewards = info["rewards"]
                #print("rew: "+str(rew))
                #print("--")            
                #print("list info[rewards]:" +str(episode_rewards))
                num_episodes = len(episode_rewards)
                #print(num_episodes)
                success = float((num_transfer)/num_moves) * 100.0
                logger.log("Percentage of successful attacks: "+str(success))
                
            if num_episodes == 100:                
                episode_rewards = np.mean(info["rewards"][-100:])
                logger.log('Reward: ' + str(episode_rewards))
            #num_episodes = len(episode_rewards)
            
            
            num_moves = 0
            num_transfer = 0


if __name__ == '__main__':
	args = parse_args()
	env, monitored_env = make_env(args.env)
	g1 = tf.Graph()
	g2 = tf.Graph()
	logger.configure(dir='.')
	with g1.as_default():
		m1 = DQNModel(env, args.dueling, args.noisy, os.path.join(args.model_dir, "saved"))
	if args.blackbox == True:
		with g2.as_default():
			m2 = DQNModel(env, args.dueling, args.noisy2, os.path.join(args.model_dir2, "saved"))
			with m2.get_session().as_default():
				craft_adv_obs = m2.craft_adv()
			with m1.get_session().as_default():
				craft_adv_obs2 = m1.craft_adv()
				play(env, m1.get_act(), craft_adv_obs, craft_adv_obs2, args.stochastic, args.video, args.attack, m1, m2)
	else:
		with m1.get_session().as_default():
			craft_adv_obs = m1.craft_adv()
			play(env, m1.get_act(), craft_adv_obs, None, args.stochastic, args.video, args.attack, m1, m1)
