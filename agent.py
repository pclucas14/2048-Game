from board import * 
import numpy as np
import os 
os.environ['THEANO_FLAGS'] = 'floatX=float32,nvcc.fastmath=True'
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as  ll
import pdb

# hyperparameters
initial_eta = 1e-2
gamma = 0.2
invalid_move_penalty = -5000
batch_size = 50
num_games = 5000000
gui = False
downcast = True

'''
class responsible for modeling P(action | state).
Policy Gradient is used. 
'''
class Agent():
    def __init__(self):
        self.default = 2
        self.build_model_and_functions()
        self.pick_random = 0.25
        self.is_last_move_random = False

    def choose_move(self, board, choose_random):
        # board is "tile matrix"
        board_rshp = np.array(board).reshape(-1, 4, 4)
        #pdb.set_trace()
        if choose_random or np.random.randn() < self.pick_random:
            self.is_last_move_random = True
            return randint(0,3)
        else : 
            self.is_last_move_random = False
            return np.argmax(self.policy_test_fn([board_rshp])[0])

    def train(self, boards, actions, rewards):
        boards = boards.reshape((-1, 1, 4, 4))
        loss = self.policy_train_fn(boards, actions, rewards)
        return loss

    def build_model_and_functions(self):
        # step 1 : build model, serving as policy
        model = []
        self.input_var = T.tensor4('inputs')
        self.target_var = T.ivector('targets')
        self.advantage = T.vector('advantage')

        model.append(ll.InputLayer(shape=(None, 1, 4, 4), input_var=self.input_var))
        model.append(ll.DenseLayer(model[-1], num_units=800))
        model.append(ll.DropoutLayer(model[-1], p=0.5))
        model.append(ll.DenseLayer(model[-1], num_units=800))
        model.append(ll.DenseLayer(model[-1], num_units=4, 
                     nonlinearity=lasagne.nonlinearities.softmax))
        self.model = model
        self.params = ll.get_all_params(model[-1], trainable=True)

        # step 2 : build functions
        #   a) eval move / test fn
        pred = ll.get_output(model[-1])
        self.policy_test_fn = theano.function([self.input_var], pred,
                                               allow_input_downcast=downcast)

        #   b) train functions
        ce_loss = lasagne.objectives.categorical_crossentropy(pred, self.target_var) #.mean()

        # scale loss by advantage --> Policy Gradient magic happens right here.
        # we normalize the advantage 
        self.advantage -= T.mean(self.advantage)
        self.advantage /=  T.std(self.advantage)

        pg_loss = (ce_loss  * self.advantage).mean()
        updates = lasagne.updates.rmsprop(pg_loss, self.params, learning_rate=initial_eta)

        self.policy_train_fn = theano.function([self.input_var, self.target_var, self.advantage],
                                               pg_loss, 
                                               updates=updates,
                                               allow_input_downcast=downcast)

'''
helper methods 
'''
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r




agent = Agent()
board = Board()

boards = None
actions = np.array([])
rewards = np.array([])
choose_random = False


# storing state/action pairs that are invalid
invalid_boards = None
invalid_actions = None

# tracking progress
game_scores = []
invalid_moves = [0,0]


# play a game 
for i in range(num_games):

    board.init_game()
    rewards_per_game = np.array([])

    if boards is None : 
        boards = np.array(board.tileMatrix).reshape(1,4,4)
    else : 
        boards = np.concatenate((boards, np.array(board.tileMatrix).reshape(1,4,4)), axis=0)

    while True:
        if not board.game_over():
            move = agent.choose_move(board.tileMatrix, choose_random=choose_random)
            reward = board.play_move(move)

            if reward != -1 : # valid move played
                choose_random = False 
                # saving move, action, reward
                boards = np.concatenate((boards, np.array(board.tileMatrix).reshape(1,4,4)), axis=0)
                actions = np.append(actions, move)
                rewards_per_game = np.append(rewards_per_game, reward)
                if not agent.is_last_move_random : invalid_moves[0] += 1
            else : 
                choose_random = True
                if invalid_boards is None : 
                    invalid_boards = np.array(board.tileMatrix).reshape(1,4,4)
                    invalid_actions = np.array([move])
                else : 
                    invalid_boards = np.concatenate((invalid_boards, np.array(board.tileMatrix).reshape(1,4,4)), axis=0)
                    invalid_actions = np.append(invalid_actions, move)
                if not agent.is_last_move_random : invalid_moves[1] += 1


        else:
            if gui : board.printGameOver()
            # drop last -useless- board
            boards = boards[:-1]
            rewards_per_game = discount_rewards(rewards_per_game)
            rewards = np.append(rewards, rewards_per_game)
            game_scores.append(np.max([boards[-1]]))
            break
    
    if i % batch_size == batch_size -1 : 
        # add invalid moves 
        boards = np.concatenate((boards, invalid_boards), axis=0)
        rewards = np.append(rewards, invalid_move_penalty * np.ones((invalid_boards.shape[0])))
        actions = np.append(actions, invalid_actions)

        agent.train(boards, actions, rewards)
        print 'average score : ', np.mean(game_scores)
        print 'invalid moves : ', invalid_moves
        # reset data structures 
        boards = None; invalid_boards = None; invalid_actions = None; 
        actions = np.array([]); rewards = np.array([]); game_scores = []; invalid_moves = [0,0]
        agent.pick_random *= 0.999


print 'game ended'
pdb.set_trace()