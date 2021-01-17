LOAD = True            # loading neural networks from file
FILENAME_FOR_LOADING="reacher_20"
SAVE = False            # saving neural networks to file
FILENAME_FOR_SAVING="reacher_20_new"
PLOT=False
PLOTNAME="reacher_20_9.png"
TRAINMODE = False
VAL_ENV_SOLVED = 30

MAX_NR_EPISODES= 2000000
UPDATE_EVERY = 20
NR_UPDATES=200
BUFFER_SIZE = 100000  # replay buffer size
BATCH_SIZE = 128      # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.001              # for soft update of target parameters
LR_ACTOR = 0.00005        # learning rate of the actor
LR_CRITIC = 0.0001       # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

