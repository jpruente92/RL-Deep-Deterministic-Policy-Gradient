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

class Hyperparameters():
    def __init__(self):
        self.LOAD=LOAD
        self.FILENAME_FOR_LOADING=FILENAME_FOR_LOADING
        self.SAVE=SAVE
        self.FILENAME_FOR_SAVING=FILENAME_FOR_SAVING
        self.PLOT=PLOT
        self.PLOTNAME=PLOTNAME
        self.TRAINMODE=TRAINMODE
        self.VAL_ENV_SOLVED=VAL_ENV_SOLVED


        self.MAX_NR_EPISODES=MAX_NR_EPISODES
        self.UPDATE_EVERY=UPDATE_EVERY
        self.NR_UPDATES=NR_UPDATES
        self.BUFFER_SIZE=BUFFER_SIZE
        self.BATCH_SIZE=BATCH_SIZE
        self.GAMMA=GAMMA
        self.TAU=TAU
        self.LR_ACTOR=LR_ACTOR
        self.LR_CRITIC=LR_CRITIC
        self.WEIGHT_DECAY=WEIGHT_DECAY
