# GAIL-for-AI2THOR
  This repository contains the source code of my project on implementing GAIL for AI2THOR environment.<br/><br />
cups_rl_data_collection(manual) is modified from cups-rl (https://github.com/TheMTank/cups-rl) to collect input to GAIL (metadata from AI2THOR). To use it: 
1. follow https://github.com/TheMTank/cups-rl for installaition 
2. and add GAIL-for-AI2THOR/cups_rl_data_collection(manual)/main.py to root directory
3. configure GAIL-for-AI2THOR/cups_rl_data_collection(manual)/gym_ai2thor/config_files/rainbow_example.json according to the respective specification of the cusomized envirnment 21 actions supported at the moment.
4. replace cups-rl/gym_ai2thor/envs/ai2thor_env.py with GAIL-for-AI2THOR/cups_rl_data_collection(manual)/gym_ai2thor/envs/ai2thor_env.py. Modify this script to add more actions. There are 21 actions supported at the moment.
5. Refer to https://github.com/SamsonYuBaiJian/actionet to generate expert trajectories. Use GAIL-for-AI2THOR/cups_rl_data_collection(manual)/metadatacollection.py to convert it to the format readable by the script.
6. modify line 147 to 152 of GAIL-for-AI2THOR/cups_rl_data_collection(manual)/main.py to the data of interest generated by ActioNet. There are certain compatibility issue spotted so it is advised to run the data colletion script frame by frame in order to monitor any inconsistency. Stop the script immediately after the line being executed for the last action. Use the rederred graphics for guidance.

stable-baselines is modified from Stable Baselines (https://stable-baselines.readthedocs.io/en/master/) with newly defined env for AI2THOR in run_in_AI2THOR_env.py. Run run_in_AI2THOR_env.py to start training. Currently this env supports 21 actions for the agent in AI2THOR envirnment. Feel free to add more if needed. Please refer to https://stable-baselines.readthedocs.io/en/master/modules/gail.html for more details on how to run and refine tune GAIL and https://stable-baselines.readthedocs.io/en/master/index.html for installation of stable-baselines.
 
