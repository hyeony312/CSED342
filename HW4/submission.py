import util, math, random
from collections import defaultdict
from util import ValueIteration
import numpy as np


############################################################
# Problem 1a: Volcano Crossing


class VolcanoCrossing():
    """
    grid_world: a 2D numpy array where 0 is explorable, negative integer is a volcano, and positive integer is the goal.
    discount: discount factor
    moveReward: reward of moving from one cell to another
    value_table: a 2D numpy array where each cell represents the value of the cell
    actions: a list of possible actions
    """
    def __init__(self, grid_world, discount=1, moveReward=-1):
        self.grid_world = grid_world
        self.discount = discount
        self.moveReward = moveReward
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Return the value table after running |numIters| of value iteration.
    # You do not need to modify this function.
    def value_iteration(self, numIters=1):
        self.value_table = np.zeros(self.grid_world.shape) # Initialize value table

        for _ in range(numIters):
            self.value_table = self.value_update(self.value_table)
        return self.value_table

    # Return the state is an end point or not.
    # You do not need to modify this function.
    # If the state is an end point(Volcano or Island), return True.
    # Otherwise(self.grid_world[state] == 0), return False.
    # This function below has been implemented for your convenience, but it is not necessarily required to be used.
    def check_end(self, state):
        return self.grid_world[state] != 0

    # Checks if the agent can move to the next state.
    # This function below has been implemented for your convenience, but it is not necessarily required to be used.
    def movable(self, state, action):
        x, y = state
        i, j = action
        return 0 <= x + i < self.grid_world.shape[0] and 0 <= y + j < self.grid_world.shape[1]


    # Return the value table after updating the value of each grid cell.
    def value_update(self, value_table):
        # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
        next_valueTable = np.zeros(self.grid_world.shape) # Initialize value table

        for i in range(self.grid_world.shape[0]):
            for j in range(self.grid_world.shape[1]):
                state = (i,j)
                # check valid start state
                if (self.check_end(state)):
                    next_valueTable[state] = self.grid_world[state]                                 
                    continue

                Q_list = {}
                for action in self.actions:
                    next_state = (state[0]+action[0],state[1]+action[1])
                    # check valid move
                    if (self.movable(state,action)!=True):
                        continue
                    # set reward
                    if (self.check_end(next_state)):
                        reward = self.grid_world[next_state] + self.moveReward
                        vopt = 0
                    else:
                        reward = self.moveReward
                        vopt = value_table[next_state]
                    # calculate Q value
                    Q_list[action] = reward + self.discount*vopt

                next_valueTable[state] = max(Q_list.values())

        return next_valueTable
        raise NotImplementedError  # remove this line before writing code
        # END_YOUR_ANSWER

############################################################
# Problem 2a: BlackjackMDP


class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        super().__init__()

        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_ANSWER (our solution is 44 lines of code, but don't worry if you deviate from this)
        def cal_Count(deck): # 덱의 카드 수를 반환
            return sum(count for count in deck)
        
        def cal_nextDeck(target, deck): # 덱 상태 업데이트
            nextDeck = []
            for card in range(len(deck)):
                if card == target:
                    nextDeck.append(deck[card]-1)
                else:
                    nextDeck.append(deck[card])
            return tuple(nextDeck)
        
        hand, peek, deck = state
        nextState = []

        if deck == None: # check bust
            return nextState

        if action == 'Take':
            if peek != None: # 이전에 peek을 했을 경우
                nextHand = hand+self.cardValues[peek]
                nextPeek = None
                nextDeck = cal_nextDeck(peek, deck)

                prob = 1

                if (nextHand > self.threshold): # Bust
                        nextPeek = None
                        nextDeck = None
                        reward = 0
                elif (nextHand == self.threshold) or (sum(nextDeck)==0): # game end
                        nextPeek = None
                        nextDeck = None
                        reward = nextHand
                else: # common game
                        reward = 0

                nextState.append(((nextHand, nextPeek, nextDeck),prob,reward))

            else:
                for card in range(len(deck)):
                    if deck[card] == 0: # 카드가 덱에 없는 경우
                        continue

                    nextHand = hand+self.cardValues[card]
                    nextPeek = None
                    nextDeck = cal_nextDeck(card, deck)

                    prob = deck[card]/cal_Count(deck) # calculate prob

                    if (nextHand > self.threshold): # Bust
                        nextPeek = None
                        nextDeck = None
                        reward = 0
                    elif (nextHand == self.threshold) or (sum(nextDeck)==0): # game end
                        nextPeek = None
                        nextDeck = None
                        reward = nextHand
                    else: # common game
                        reward = 0

                    nextState.append(((nextHand, nextPeek, nextDeck),prob,reward))

        elif action == 'Peek':
            if peek == None:
                for card in range(len(deck)):
                    if deck[card] == 0: # 카드가 덱에 없는 경우
                            continue
                    
                    nextHand = hand
                    nextPeek = card
                    nextDeck = deck

                    prob = deck[card]/cal_Count(deck) # calculate prob

                    reward = -1

                    nextState.append(((nextHand, nextPeek, nextDeck),prob,reward))

        elif action =='Quit' or cal_Count(deck)==0:
            nextHand = hand
            nextPeek = None
            nextDeck = None

            prob = 1

            reward = hand

            nextState.append(((nextHand, nextPeek, nextDeck),prob,reward))

        else:
            print("Invalid action input")

        return nextState
        raise NotImplementedError  # remove this line before writing code
        # END_YOUR_ANSWER

    def discount(self):
        return 1


############################################################
# Problem 3a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class Qlearning(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with episode=[..., state, action,
    # reward, newState], which you should use to update
    # |self.weights|. You should update |self.weights| using
    # self.getStepSize(); use self.getQ() to compute the current
    # estimate of the parameters. Also, you should assume that
    # V_opt(newState)=0 when isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        state, action, reward, newState = episode[-4:]

        if isLast(state):
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        prediction = self.getQ(state, action)

        Vopt = max(self.getQ(newState, newAction) for newAction in self.actions(state))
        target = reward + self.discount*Vopt

        Eta = self.getStepSize()

        for feature, value in self.featureExtractor(state,action):
            self.weights[feature] -= Eta*(prediction - target)*value

        return
        raise NotImplementedError  # remove this line before writing code
        # END_YOUR_ANSWER


############################################################
# Problem 3b: Q SARSA

class SARSA(Qlearning):
    # We will call this function with episode=[..., state, action,
    # reward, newState, newAction, newReward, newNewState], which you
    # should use to update |self.weights|. You should
    # update |self.weights| using self.getStepSize(); use self.getQ()
    # to compute the current estimate of the parameters. Also, you
    # should assume that Q_pi(newState, newAction)=0 when when
    # isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        assert (len(episode) - 1) % 3 == 0
        if len(episode) >= 7:
            state, action, reward, newState, newAction = episode[-7: -2]
        else:
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        prediction = self.getQ(state, action)
        estimate = self.getQ(newState, newAction)
        
        Eta = self.getStepSize()

        for feature, value in self.featureExtractor(state,action):
            self.weights[feature] = (1-Eta)*prediction + Eta*(reward + self.discount*estimate)

        return
        raise NotImplementedError  # remove this line before writing code
        # END_YOUR_ANSWER

# Return a singleton list containing indicator feature (if exist featurevalue = 1)
# for the (state, action) pair.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 3c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs
# (see identityFeatureExtractor() above for an example).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card type and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card type is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).
#       Example: if the deck is (3, 4, 0, 2), you should have four features (one for each card type).
#       And the first feature key will be (0, 3, action)
#       Only add these features if the deck != None

def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
    if counts == None:
        counts = []
    checkDeck = []

    feature1 = ((total,action), 1)
    
    for card in range(len(counts)):
        if counts[card] == 0:
            checkDeck.append(0)
        else:
            checkDeck.append(1)
    feature2 = ((tuple(checkDeck), action), 1)

    feature3 = [] 
    for card in range(len(counts)):
        feat = ((card, counts[card], action), 1)
        feature3.append(feat)

    f_extracted = [feature1, feature2]
    f_extracted += feature3

    # print(f_extracted)
    return f_extracted
    raise NotImplementedError  # remove this line before writing code
    # END_YOUR_ANSWER
