# Udacity Reinforcement Learning Nano Degree: Continuous Control Project

In this project, we will implement a DDPG (Deep Deterministic Policy Gradient) agent to solve Unity's Reacher environment.

## 1. Environment Details: 
In this project, we have to train an agent to move a robotic arm to reach a moving target. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/134915969-2c5bf625-8ea1-4a7f-b37a-3f54e039d99b.png" alt="drawing" width="400"/>
</p>

NOTE:

1. This project was completed in the Udacity Workspace, but the project can also be completed on a local Machine. Instructions on how to download and setup Unity ML environments can be found in [Unity ML-Agents Github repo](https://github.com/Unity-Technologies/ml-agents).
1. The environment provided by Udacity is similar to, but not identical to the Reacher environment on the [Unity ML-Agents Github page](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md).

In this environment, a double-jointed arm can move to target locations. A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of `33` variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between `-1` and `1`.

There are 2 versions of this environment. Both are solved:

### Version 1: One Agent:

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/134917884-d467102a-31c6-4cc2-a7a6-a677bb024bf2.png" alt="drawing" width="400"/>
</p>

The task is episodic, and in order to solve the environment, your agent must get an average score of `+30` over `100` consecutive episodes.

### Version 2: Twenty Agents:

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/134917645-53c47841-572c-4b92-b447-75967c6d9406.png" alt="drawing" width="400"/>
</p>

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of `+30` (over `100` consecutive episodes, and over all agents). Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
- This yields an average score for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over `100` episodes) of those average scores is at least `+30`.

## 2. Requirements for running the code:

#### Step 1: Install Numpy
Follow the instructions [here](https://numpy.org/install/) to install the latest version of Numpy.

#### Step 2: Install Pytorch and ML-Agents
If you haven't already, please follow the instructions in the [DRLND (Deep Reinforcement Learning Nano Degree) GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in `README.md` at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

_(For Windows users)_ The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

#### Step 3: Download the Reacher Environment
For this project, you will __not__ need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

##### Version 1: One Agent:
- Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

##### Version 2: Twenty Agents:
- Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Then, place the file in the `p2_continuous-control/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.

_(For Windows users)_ Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

_(For AWS)_ If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

## 3. Explore the Environment
After you have followed the instructions above, open `Continuous_Control.ipynb` located in the `project_continuous_control/` folder and follow the instructions to learn how to use the Python API to control the agent. The saved weights file is named `checkpoint.pth` located in the `project_navigation/` folder. There is another saved weights file named `BestCheckpoint.pth` that is the result of a 2000 episodes run with an average score of 15.82.

## 4. Implementation Details
To solve this environment, we used Reinforcement Learning algorithms to train an agent to collect the yellow bananas not the blue ones. In particulat, we used the DQN (Deep Q-Network) algorithm. DQN uses machine learning algorithms, such as neural networks, as a function approximator to the Q-table. We briefly explain these concepts in this section.

### Introduction
Based on MDP (Markov Decision Process), the relationship between the agent (that we want to train) and the environment (that we want the agent to interact with) consists of state, action and reward. The agent begins in a specific state, takes a specific action, and the environment responds with a new state for the agent and a reward based on this transition between the previous and current states. The goal of the agent is to maximize this reward over time. That is where RL (Reinforcement Learning) algorithms come to work.


<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/133955246-8859bfa6-cb2a-4fba-a1fd-d9d23b4d49aa.png" alt="drawing" width="500"/>
</p>
<p align="center">
  <em>Fig.1: The agent–environment interaction in a Markov decision process</em>
</p>

There are many Reinforcement Learning algorithms that was invented over the years such as Monte Carlo methods and Temporal Difference. They use a Q-table (Action-Value Function) to correlate states, actions and cumulative rewards. For this table to exist, the state space and action space have to be discrete. But most of the applications in real life don't have finite number of states and actions. They have continuous state and action spaces (for example: the torque of a motor is not a discrete value but a continuous one). In this case, the Q-table will have infinite number of cells and this is, of course, not logical!

To read more about these topics, refer to the textbook [An Introduction to Reinforcement Learning by Richard S. Sutton and Andrew G. Barto](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf).

<p align="center">
  <img src="https://video.udacity-data.com/topher/2018/May/5aecac13_latex-image-1-copy/latex-image-1-copy.png" alt="drawing" width="400"/>
</p>
<p align="center">
  <em>Fig.2: Q-table example</em>
</p>

To overcome the problem of infinite state and action spaces, methods such as discretization, tile coding and function approximation were introduced. Deep RL is a function approximation method that uses machine learning to approximate the Q-table. This is the method used in this project.

### Deep Q-Networks
The agent's Q-table is replaced by a neural network. Its input is the state of the environment. Its output is the action value. If the state is an image (or a batch of images), we use convolutional layers.

### Neural Network Architecture
In our environment's case, there are `37 states` and `4 actions`. The input layer has `37` neurons. The output layer has `4` neurons. Between them we used 2 hidden layers: the first one has `256` neurons and the other one has `64` neurons. The arcitecture is shown in _fig.3_.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/134067521-1e16370b-85af-4144-a9b4-e06fa08cc454.png" alt="drawing" width="400"/>
</p>
<p align="center">
  <em>Fig.3: Neural Network Architecture Used in Project</em>
</p>

We use 2 neural networks with the same architecture to be able to have 2 sets of weights: __*w*__ and __*w<sup>-</sup>*__. This will be illustrated in the _Fixed Q-Targets_ section.

The hyperparameters are:
* Activation function: `relu`.
* Batch size: `64`.
* Learning rate &alpha; : `0.0005`.

### RL Algorithm
#### Experience Replay
When the agent interacts with the environment, the sequence of experience tuples can be highly correlated. The naive Q-learning algorithm that learns from each of these experience tuples in sequential order runs the risk of getting swayed by the effects of this correlation. By instead keeping track of a __replay buffer__ and using __experience replay__ to sample from the buffer at random, we can prevent action values from oscillating or diverging catastrophically.

The __replay buffer__ contains a collection of experience tuples __*(S, A, R, S')*__. The tuples are gradually added to the buffer as we are interacting with the environment.

The act of sampling a small batch of tuples from the replay buffer in order to learn is known as __experience replay__. In addition to breaking harmful correlations, experience replay allows us to learn more from individual tuples multiple times, recall rare occurrences, and in general make better use of our experience.

The hyperparameters are:
* Buffer size: `100,000`.

#### Fixed Q-Targets
In Q-Learning, we update a guess with a guess, and this can potentially lead to harmful correlations. To avoid this, we can update the parameters __*w*__ in the network $Q_hat$ to better approximate the action value corresponding to state __*S*__ and action __*A*__ with the following update rule:

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/134077362-35bad85b-cb3e-41fc-9d35-35f2a66201ca.png" alt="drawing" width="600"/>
</p>

where __*w<sup>-</sup>*__ are the weights of a separate target network that are not changed during the learning step, and __*(S, A, R, S')*__ is an experience tuple.

The hyperparameters are:
* Discount factor &gamma;: `0.99`.

#### Double Q-Learning
The popular Q-learning algorithm is known to overestimate action values under certain conditions. It was not previously known whether, in practice, such overestimations are common, whether they harm performance, and whether they can generally be prevented. Double Q-learning algorithm show that the resulting algorithm not only reduces the observed overestimations, as hypothesized, but that this also leads to much better performance.

The usual update rule was:
<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/134080314-b8750711-476f-4a17-aaba-831b17dc04e1.png" alt="drawing" width="500"/>
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/134079693-9111d9ef-b321-4f0b-b691-97f71cff23b5.png" alt="drawing" width="300"/>
</p>

We replace __*Y<sub>t</sub><sup>Q</sup>*__ with this equation:
<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/134080060-51b11024-9e39-41d9-a8de-2dba79bf825c.png" alt="drawing" width="400"/>
</p>

Notice that the selection of the action, in the argmax, is still due to the online weights __*w<sub>t</sub>*__. This means that, as in Q-learning, we are still estimating the value of the greedy policy according to the current values, as defined by __*w<sub>t</sub>*__. However, we use the second set of weights __*w<sub>t</sub><sup>-</sup>*__ to fairly evaluate the value of this policy. This second set of weights can be updated symmetrically by switching the roles of __*w<sub>t</sub>*__ and __*w<sub>t</sub><sup>-</sup>*__.

For more information about Double Q-Learning, read the paper [here](https://arxiv.org/abs/1509.06461).

#### Soft Update of Target Network
Instead of updating the target network parameters every number of steps. The target network parameters are updated at every step decayed by a parameter &tau;:
<p align="center">
  w<sub>t</sub><sup>-</sup> = &tau; w<sub>t</sub> + (1 - &tau;) w<sub>t</sub><sup>-</sup>
</p>

The hyperparameters are:
* Soft update parameter &tau;: `0.001`.

### Plot of Rewards
For the environment to be solved, the average reward over 100 episodes must reach at least 13. The implementation provided here needed just `501 episodes` to be completed! The average score reached `13.04`. The plot of rewards is shown in _fig.4_.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/134100064-bfaaeedc-b14b-4a82-a426-55129f7b798e.png" alt="drawing" width="400"/>
</p>
<p align="center">
  <em>Fig.4: Rewards Plot in 501 episodes</em>
</p>

Another run of `2000 episodes` gave an average score of `15.82`. The plot of rewards of this long run is shown in _fig.5_.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/134100605-7a0badc9-dac3-4e1e-baf6-f79e91e73d3d.png" alt="drawing" width="400"/>
</p>
<p align="center">
  <em>Fig.5: Rewards Plot in 2000 episodes</em>
</p>

### Ideas for Future Work
Some additional features could be added to provide better performance:
* __Prioritized Experience Replay:__ Instead of randomly choosing the expriences, we choose them based on how much they affect our learning process. For more information read this paper [here](https://arxiv.org/abs/1511.05952).
* __Dueling DQN:__ Instead of having one output layer representing the action values, we have 2 output layers representing state values and advantage values. Combining both results the action values. For more information read this paper [here](https://arxiv.org/abs/1511.06581).
* __Using images:__ Instead of states, we use the image of the game itself as an input to the neural network. We then have to introduce some changes to the architecture of the netwok. We could use convolutional layers. This will be a more challenging problem.
