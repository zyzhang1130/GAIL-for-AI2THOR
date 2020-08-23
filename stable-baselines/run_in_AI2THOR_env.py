#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:15:15 2020

@author: user
"""


import gym

from stable_baselines import GAIL, SAC, PPO1
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from stable_baselines.common.cmd_util import make_vec_env


from ai2thor.controller import Controller
import gym
from gym.spaces import Discrete, Box
from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np
import random

from stable_baselines.common.env_checker import check_env

from ai2thor.controller import Controller
import gym
from gym.spaces import Discrete, Box
from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np
import random

from stable_baselines.common.env_checker import check_env

import re
import numpy as np

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import gutenberg
from multiprocessing import Pool
from scipy import spatial

import pickle
import os
import time

class GymController(gym.Env):
    """An object navigation RoboTHOR Gym Environment.

    Observations are RGB images in the form of Box(height, width, 3)
        with values from [0:1]. 
    
    Actions are Discrete(4); that is, (0, 1, 2, 3). Here,
        0 = 'RotateLeft'
        1 = 'MoveAhead'
        2 = 'RotateRight'
        3 = 'Done' (see https://arxiv.org/pdf/1807.06757.pdf)
    """

    valid_target_objects = {
        'AlarmClock',
        'Apple',
        'BaseballBat',
        'BasketBall',
        'Bowl',
        'GarbageCan',
        'HousePlant',
        'Laptop',
        'Mug',
        'RemoteControl',
        'SprayBottle',
        'Television',
        'Vase',
        'CounterTop',
        'Fridge',
        'Egg',
        "CounterTop",
        "TableTop",
        "Sink",
        "Bowl",
        "Fridge",
        "Drawer",
        "Cup",
        "GarbageCan",
        "Cabinet",
        "Pot",
        "Microwave",
        "Mug"
    }

    def __init__(self,
            target_object: str='Egg',
            controller_properties: dict=dict(),
            horizon: int=200,
            seed: int=42):
        """Initializes a new AI2-THOR Controller and gym environment.

        Args:
            'target_object' (str='Television'): The name of the target object. See
                https://ai2thor.allenai.org/robothor/documentation/#object-types-
                for more options.

            'conroller_properties' (dict=None): The properties used to initialize
                an AI2-THOR Controller object. For more information, see:
                https://ai2thor.allenai.org/robothor/documentation/#initialization.

            'horizon' (int=200): Maximum number of time steps before failure.

            'seed' (int=42): The random seed used for reproducing results.
        """
        assert target_object in self.valid_target_objects, \
            'Invalid target object, see https://ai2thor.allenai.org/robothor/documentation/#object-types-.'
        self.controller = Controller(**controller_properties)
        
        self.objects = {'pickupables': ["Egg"],
                            'receptacles': [
        "CounterTop",
        "TableTop",
        "Sink",
        "Bowl",
        "Fridge",
        "Drawer",
        "Cup",
        "GarbageCan",
        "Cabinet",
        "Pot",
        "Microwave",
        "Mug"
    ],
                            'openables':   ["Fridge","Drawer","Cabinet"]}
        # helper fields
        self.target_object = target_object
        self.current_time_step = 0
        self.episode_already_done = False
        self.horizon = horizon
        self.rotation_amount = 30.0            
        self.anglehandX = 0.0
        self.anglehandY = 0.0
        self.anglehandZ = 0.0
        
        # set scenes and get reachable/target positions for each scene
        self._scene_names = self.scene_names()
        self.reachable_positions = self._get_all_reachable_positions()

        # for reset rotations
        self.rotateStepDegrees = 90 if 'rotateStepDegrees' not in \
            controller_properties else controller_properties['rotateStepDegrees']

        # set gym gym action space and observation space
        self.width = 300 if 'width' not in controller_properties \
            else controller_properties['width']
        self.height = 300 if 'height' not in controller_properties \
            else controller_properties['height']


        # set random seeds
        self.seed(seed)
    
    @property
    def observation_space(self) -> gym.spaces:
        """Returns the gym.spaces observation space."""
        # return Box(low=0, high=255, shape=(self.height*self.width*3,), dtype=np.float64)
        return Box(low=-10, high=10, shape=(200,), dtype=np.float32)

    @property
    def action_space(self) -> gym.spaces:
        """Returns the gym.spaces action space."""
        return Discrete(21)
    

    
    def reward_range(self) -> gym.spaces:
        """Returns the gym.spaces action space."""
        return (float('-inf'), float('inf'))
    
    def metadata(self) -> gym.spaces:
        """Returns the gym.spaces action space."""
        return self.controller.event.metadata['reachablePositions']

    def scene_names(self) -> List[str]:
        """Returns a list of the RoboTHOR training scene names.
        
        For more information on RoboTHOR scenes, see:
        https://ai2thor.allenai.org/robothor/documentation/#training-scenes
        """
        scenes = []
        # for wall_config in range(1, 13):
        #     for object_config in range(1, 6):
        #         scenes.append(f'FloorPlan_Train{wall_config}_{object_config}')
        scenes.append('FloorPlan12')
        return scenes

    def step(self, action: int) -> Tuple[np.array, float, bool, dict]:
        """Takes a step in the AI2-THOR environment.

        If the episode is already over the action will not be called
            and nothing will return.

        Actions are Discrete(4); that is, (0, 1, 2, 3). Here,
            0 = 'RotateLeft'
            1 = 'MoveAhead'
            2 = 'RotateRight'
            3 = 'Done' (see https://arxiv.org/pdf/1807.06757.pdf)
        
        Returns a tuple of (
            observation (np.array): The agent's observation after taking
                a step. This is updated in get_observation(),

            reward (float): The reward from the environment after
                taking a step. This is updated in reward_function().

            done (bool): True if the episode ends after the action,

            metadata (dict): The metadata after taking the action. See
                https://ai2thor.allenai.org/robothor/documentation/#metadata
                for more information.
        )
        """
        assert action in self.action_space, 'Invalid action'
        if self.episode_already_done:
            return None, None, True, None
        
        visible_objects = [obj for obj in self.controller.last_event.metadata['objects'] if obj['visible']]
        interaction_obj, distance = None, float('inf')
        inventory_before = self.controller.last_event.metadata['inventoryObjects'][0]['objectType'] \
                if self.controller.last_event.metadata['inventoryObjects'] else []
        
        
        
        if action == 0:
            self.controller.step(dict(action='MoveAhead',moveMagnitude=0.25))
        elif action == 1:
            self.controller.step(dict(action='MoveBack',moveMagnitude=0.25))
        elif action == 2:
            self.controller.step(dict(action='MoveRight',moveMagnitude=0.25))
        elif action == 3:
            self.controller.step(dict(action='MoveLeft',moveMagnitude=0.25))
        elif action == 4:
            self.controller.step(dict(action='LookUp', moveMagnitude=0.1))
        elif action == 5:
            self.controller.step(dict(action='LookDown',moveMagnitude=0.1))
        elif action == 6:
            self.controller.step(dict(action='RotateRight'))
        elif action == 7:
            self.controller.step(dict(action='RotateLeft'))
            
        elif action == 8:
            closest_openable = None
            for obj in visible_objects:
                # look for closest closed receptacle to open it
                is_closest_closed_receptacle = obj['openable'] and \
                        obj['distance'] < distance and not obj['isOpen'] and \
                        obj['objectType'] in self.objects['openables']
                if is_closest_closed_receptacle:
                    closest_openable = obj
                    distance = closest_openable['distance']
            if closest_openable:
                interaction_obj = closest_openable
                self.controller.step(
                    dict(action='OpenObject', objectId=interaction_obj['objectId']))
                self.controller.last_event.metadata['lastObjectOpened'] = interaction_obj
                
                
        elif action == 9:
            closest_openable = None
            for obj in visible_objects:
                # look for closest opened receptacle to close it
                is_closest_open_receptacle = obj['openable'] and obj['distance'] < distance \
                                              and obj['isOpen'] and \
                                              obj['objectType'] in self.objects['openables']
                if is_closest_open_receptacle:
                    closest_openable = obj
                    distance = closest_openable['distance']
            if closest_openable:
                interaction_obj = closest_openable
                self.controller.step(
                    dict(action='CloseObject', objectId=interaction_obj['objectId']))
                self.controller.last_event.metadata['lastObjectClosed'] = interaction_obj
                    
        elif action == 10:
            closest_pickupable = None
            for obj in visible_objects:
                # look for closest object to pick up
                closest_object_to_pick_up = obj['pickupable'] and \
                                            obj['distance'] < distance and \
                            obj['objectType'] in self.objects['pickupables']
                if closest_object_to_pick_up:
                    closest_pickupable = obj
            if closest_pickupable and not self.controller.last_event.metadata['inventoryObjects']:
                interaction_obj = closest_pickupable
                self.controller.step(
                    dict(action='PickupObject', objectId=interaction_obj['objectId']))
                self.controller.last_event.metadata['lastObjectPickedUp'] = interaction_obj
                    
                    
        elif action == 11:
            closest_receptacle = None
            if self.controller.last_event.metadata['inventoryObjects']:
                for obj in visible_objects:
                    # look for closest receptacle to put object from inventory
                    closest_receptacle_to_put_object_in = obj['receptacle'] and \
                                                          obj['distance'] < distance \
                                    and obj['objectType'] in self.objects['receptacles']
                    if closest_receptacle_to_put_object_in:
                        closest_receptacle = obj
                        distance = closest_receptacle['distance']
                if closest_receptacle:
                    interaction_obj = closest_receptacle
                    object_to_put = self.controller.last_event.metadata['inventoryObjects'][0]
                    self.controller.step(
                            dict(action='PutObject',
                                  objectId=object_to_put['objectId'],
                                  receptacleObjectId=interaction_obj['objectId']))
                    self.controller.last_event.metadata['lastObjectPut'] = object_to_put
                    self.controller.last_event.metadata['lastObjectPutReceptacle'] = interaction_obj
                        
        elif action == 12:
            self.controller.step(dict(action='MoveHandAhead', moveMagnitude=0.1))
        elif action == 13:
            self.controller.step(dict(action='MoveHandLeft', moveMagnitude=0.1))
        elif action == 14:
            self.controller.step(dict(action='MoveHandRight', moveMagnitude=0.1))
        elif action == 15:
            self.controller.step(dict(action='MoveHandBack', moveMagnitude=0.1))
        elif action == 16:
            self.controller.step(dict(action='MoveHandUp', moveMagnitude=0.1))
        elif action == 17:
            self.controller.step(dict(action='MoveHandDown', moveMagnitude=0.1))
            
        elif action == 18:
            if self.controller.last_event.metadata['inventoryObjects']:
                self.controller.last_event.metadata['lastObjectDropped'] =self.controller.last_event.metadata['inventoryObjects'][0]
                self.controller.step(dict(action='DropHandObject'))
                
        elif action == 19:        
            self.controller.step(
                                dict(action='Crouch'))
            
        elif action == 20:
            self.controller.step(
                                dict(action='Stand'))
            

            
        # elif action == 21:
        #     self.anglehandX+=self.rotation_amount
        #     self.controller.step(dict(action='RotateHand', x=self.anglehandX))
        # elif action == 22:
        #     anglehandY=anglehandY+30.0
        #     self.controller.step(dict(action='RotateHand', y=anglehandY))
        # elif action == 23:
        #     anglehandZ=anglehandZ+30.0
        #     self.controller.step(dict(action='RotateHand', z=anglehandZ))

        self.current_time_step += 1
        done = self.episode_done(done_action_called=False)

        return (
            self.get_observation(),
            self.reward_function(),
            done,
            self.controller.last_event.metadata
        )

    def reward_function(self) -> float:
        """Returns 1 if the episode is a success and done, otherwise -1."""
        return 10. if self.episode_success() else 0.

    def episode_success(self) -> bool:
        """Returns True if the episode is done and a target object is visible."""
        # if self.episode_already_done:
        for i in self.controller.last_event.metadata['objects']:
            if i['objectType']=='Egg':
                if i['parentReceptacles']!=None and 'Fridge' in i['parentReceptacles'][0]:
                    for j in self.controller.last_event.metadata['objects']:
                        if j['objectType']=='Fridge':
                            if j['isOpen']==False:
                                return True
                elif i['parentReceptacles']!=None and 'Pot' in i['parentReceptacles'][0]:
                    for j in self.controller.last_event.metadata['objects']:
                        if j['objectType']=='Pot':
                            if j['isOpen']==False:
                                return True
            # objects = self.controller.last_event.metadata['objects']
            # for obj in objects:
            #     if obj['objectType'] == self.target_object and obj['visible']:
            #             return True
        return False

    def episode_done(self, done_action_called: bool=False) -> None:
        """Returns True if the episode is done.

        Args:
            'done_action_called' (bool=False): Did the agent call the Done
                action? For embodied navigation, it is recommended that the
                agent calls a 'Done' action when it believes it has
                finished its task. For more information, see
                https://arxiv.org/pdf/1807.06757.pdf.
        """
        self.episode_already_done = self.episode_already_done or \
            done_action_called or self.current_time_step > self.horizon
        return self.episode_already_done

    def get_observation(self) -> np.array:
        """Returns the normalized RGB image frame from THOR."""
        # rgb_image = self.controller.last_event.frame
        # # return rgb_image / 255
        # meta=self.controller.last_event.metadata
        # return (rgb_image/255*255).flatten()
    
        embedding = self.controller.last_event.metadata['objects']
        model = Doc2Vec.load('/home/user/Documents/Zeyu/Embedding/Word-embedding-with-Python-master/doc2vec/source code/doc2vec_model_FloorPlan12')
        met=[]
        for j in embedding:
            temp=list(j.values())
            for k in range(len(temp)):
                temp[k]=str(temp[k])
            #print(temp)
            met=met+temp
        sentence_embeddings = model.infer_vector(met)
        # sentence_embeddings = np.float64(sentence_embeddings)
        return sentence_embeddings

    def reset(self) -> np.array:
        """Resets the agent to a random position/rotation in a random scene
           and returns an initial observation."""
        self.episode_already_done = False

        # choose a random scene
        # scene = random.choice(self._scene_names)
        scene = 'FloorPlan12'
        self.controller.reset(scene)

        # set a random initial position
        rand_xyz_pos = random.choice(self.reachable_positions[scene])

        # note that np.arange works with decimals, while range doesn't
        rand_yaw = random.choice(np.arange(0, 360, self.rotateStepDegrees))

        self.controller.step(action='TeleportFull',
            rotation=dict(x=0.0, y=rand_yaw, z=0.0),
            **rand_xyz_pos
        )

        return self.get_observation()

    def close(self):
        """Ends the controllers session."""
        self.controller.stop()

    def _get_all_reachable_positions(self) -> Dict[str, Dict[str, float]]:
        """Sets the reachable positions for each scene in 'scene_names()'."""
        reachable_positions = dict()
        for scene in self._scene_names:
            self.controller.reset(scene)
            event = self.controller.step(action='GetReachablePositions')
            reachable_positions[scene] = event.metadata['reachablePositions']
        return reachable_positions

    def seed(self, seed_num: int=42):
        """Sets the random seed for reproducibility."""
        random.seed(seed_num)

    def render(self, mode=None) -> None:
        """Provides a warning that render doesn't need to be called for AI2-THOR.

        We have provided it in case somebody copies and pastes code over
        from OpenAI Gym."""
        import warnings
        warnings.warn('The render function call is unnecessary for AI2-THOR.')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.controller.stop()
        

        
        
env = GymController()
env = make_vec_env(lambda: env, n_envs=1)
        
# Generate expert trajectories (train expert)
# model = PPO1('MlpPolicy', env, verbose=1)
# generate_expert_traj(model, '/home/user/Documents/Zeyu/stable-baselines/stable_baselines/gail/dataset/ai2thor', n_timesteps=4000, n_episodes=50)

# Load the expert dataset
dataset = ExpertDataset(expert_path='/home/user/Documents/Zeyu/cups-rl2_metadata_collection (manual)/data/floorplan12.npz', traj_limitation=10, verbose=1)

model = GAIL('MlpPolicy', env, dataset, verbose=1)
# # Note: in practice, you need to train for 1M steps to have a working policy
model.learn(total_timesteps=10000)
model.save("floorplan12_10000(2)")

del model # remove to demonstrate saving and loading

model = GAIL.load("/home/user/Documents/Zeyu/stable-baselines/floorplan12_10000(2)")

obs = env.reset()
while True:
  time.sleep(0.3)
  action, _states = model.predict(obs)
  obs, rewards, dones, info = env.step(action)
  # env.render()
