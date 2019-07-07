# Info-Sec-Hammerer-and-Brunori
Adversarial Attacks on Reinfrocement Learning

by Martin Hammerer and Elisa Brunori


Adversarial Attacks on Reinforcement Learning Agents


We adapted the framework from  "Whatever Does Not Kill Deep Reinforcement Learning, Makes It Stronger" (Behzadan & Munir, 2017 - https://arxiv.org/abs/1712.09344 ) to perform adversarial attacks in training and test time on the Atari games Pong and Seaquest.

In the train.py script for training-time attacks we disabled saving the training progress because our RAM was not powerful enough (even on the Colab GPU with 12 GB RAM). 
This means that our version cannot resume training for an already pre-trained agent, but one has to start training models from the beginning.

In 'deep/build_graph.py' we changed some function calls when using CW-L2 and bim attacks in the function "build_train", as the original ones did not work for us. 

For the test time attacks we adapted the provided enjoy-adv.py to limit the number of computed episodes to 100 and added the possibility to specify the attack probability. For printing the statistics we added the a SimpleMonitor Wrapper to the environment (similar to the train.py). Furthermore we are now able to print the original and the perturbed image side by side.


You can find our original source code in the folder "image-scripts", where the scripts are used to extract the results from the log file created by enjoy-env.py and train.py

PS: Unfortunately we were not able to upload the models as they are too large.
