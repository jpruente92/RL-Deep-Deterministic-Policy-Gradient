# RL_class_project_2

This project is an implementation of a the Deterministic Deep Policy Gradient algorithm for solving a unity environment.

### Required packages:
- numpy
- python (version 3.6)
- pytorch
- unityagents 

### Dependencies

The best way to run the code in this repository is to create a  conda environment by following the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone this repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/jpruente92/RL_class_project_1
cd RL_class_project_1/python
pip install .
```
4. Use the dlrnd environment for starting the program.

### Required files:
The unity exe file has to be inside this folder; here is an environment for Windows (64-bit) called "Reacher.exe" included. If you do not have Windows (64-bit), you can download the environment with one of the following links:
- Linux (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit) (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)

### Reacher environment:
In the Reacher environment, 20 double-jointed arms can move to target locations indicated by bubbles. 
A reward of +0.1 is provided for each step that an agent's hand is in the goal location.
The observation space consists of 33 variables corresponding to position, rotation, velocity, 
and angular velocities of the arms. Each action is a vector with four numbers, corresponding 
to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Starting the program:
- hyperparameters and settings for the algorithm can be changed in the file "hyperparameters.py".
    -> for viewing a trained agent set "LOAD" to True,"FILENAME_FOR_LOADING" to the name of the files of the model
    weights(without "reacher_20") and "ENV_TRAIN" to False.
    -> for training a new agent set "LOAD" to False and "ENV_TRAIN" to True and if you want to
    save the model weights, set "Save" to True and "FILENAME_FOR_SAVING" to the name for the weight files
- after the hyperparameters are set, the file "main.py" has to be started
- changes in all other files are not recommended.


