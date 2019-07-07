# Info-Sec-Hammerer-and-Brunori
Adversarial Attacks on Reinfrocement Learning

by Martin Hammerer and Elisa Brunori


Adversarial Attacks on Reinforcement Learning Agents


We adapted the framework from  "Whatever Does Not Kill Deep Reinforcement Learning, Makes It Stronger" (Behzadan & Munir, 2017 - https://arxiv.org/abs/1712.09344 ) to perform adversarial attacks in training and test time on the Atari games Pong and Seaquest.

In the train.py script for training time we disabled saving the training progress because it blew up our RAM (also on the Colab GPU with 12 GB RAM). 

This means that our version cannot resume training for an already pre-trained agent, but one has to start training models from the beginning.

In 'deep/build_graph.py' we changed some function calls when using cwl2 and bim attacks in the function ", as the original ones did not work for us.

For the test time attacks adapted the provided enjoy-adv.py to limit the number of computed episodes to 100 and added the possibility to specify the attack probability. For printing the statistics we added the a SimpleMonitor Wrapper to the environment (similar to the train.py)
