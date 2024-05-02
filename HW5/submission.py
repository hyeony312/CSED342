from util import manhattanDistance
from game import Directions
import random, util

from game import Agent



class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """
  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER
    Table = {}
    for action in gameState.getLegalActions(0):
      Table[action]=self.getQ(gameState,action)
    for k, v in Table.items():
      if v == max(Table.values()):
        return k
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    def minimax_search(agent, gameState, depth):
      valueList = []
      next_agent = agent+1
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)
      
      if agent == 0: #pacman
        for action in gameState.getLegalActions(agent):
          valueList.append(minimax_search(next_agent,gameState.generateSuccessor(agent, action),depth))
        return max(valueList)
      else: #ghost
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth = depth - 1

        for action in gameState.getLegalActions(agent):
          valueList.append(minimax_search(next_agent,gameState.generateSuccessor(agent, action),depth))
        return min(valueList)
      
    return minimax_search(1,gameState.generateSuccessor(0, action),self.depth)
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    Table = {}
    for action in gameState.getLegalActions(0):
      Table[action]=self.getQ(gameState,action)
    for k, v in Table.items():
      if v == max(Table.values()):
        return k
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def Expectimax_search(agent, gameState, depth):
      valueList = []
      next_agent = agent+1
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)
      
      if agent == 0: #pacman
        for action in gameState.getLegalActions(agent):
          valueList.append(Expectimax_search(next_agent,gameState.generateSuccessor(agent, action),depth))
        return max(valueList)
      else: #ghost
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth = depth - 1
        
        actionNum = len(gameState.getLegalActions(agent))
        phi = 1/actionNum
        value = 0

        for action in gameState.getLegalActions(agent):
          value += phi*Expectimax_search(next_agent,gameState.generateSuccessor(agent, action),depth)
        return value
      
    return Expectimax_search(1,gameState.generateSuccessor(0, action),self.depth)
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    Table = {}
    for action in gameState.getLegalActions(0):
      Table[action]=self.getQ(gameState,action)
    for k, v in Table.items():
      if v == max(Table.values()):
        return k
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def BiasedExp_search(agent, gameState, depth):
      valueList = []
      next_agent = agent+1
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)
      
      if agent == 0: #pacman
        for action in gameState.getLegalActions(agent):
          valueList.append(BiasedExp_search(next_agent,gameState.generateSuccessor(agent, action),depth))
        return max(valueList)
      else: #ghost
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth = depth - 1
        
        actionNum = len(gameState.getLegalActions(agent))
        prob = 1/actionNum
        value = 0

        for action in gameState.getLegalActions(agent):
          if action == Directions.STOP:
            phi = 0.5+0.5*prob
          else:
            phi = 0.5*prob

          value += phi*BiasedExp_search(next_agent,gameState.generateSuccessor(agent, action),depth)
        return value
      
    return BiasedExp_search(1,gameState.generateSuccessor(0, action),self.depth)
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    Table = {}
    for action in gameState.getLegalActions(0):
      Table[action]=self.getQ(gameState,action)
    for k, v in Table.items():
      if v == max(Table.values()):
        return k
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def ExpMinimax_search(agent, gameState, depth):
      valueList = []
      next_agent = agent+1
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)
      
      if agent == 0: #pacman
        for action in gameState.getLegalActions(agent):
          valueList.append(ExpMinimax_search(next_agent,gameState.generateSuccessor(agent, action),depth))
        return max(valueList)
      elif agent%2 == 1: #oddNum ghost
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth = depth - 1
          
        for action in gameState.getLegalActions(agent):
          valueList.append(ExpMinimax_search(next_agent,gameState.generateSuccessor(agent, action),depth))
        return min(valueList)
      else: #evenNum ghost
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth = depth - 1
        
        actionNum = len(gameState.getLegalActions(agent))
        phi = 1/actionNum
        value = 0

        for action in gameState.getLegalActions(agent):
          value += phi*ExpMinimax_search(next_agent,gameState.generateSuccessor(agent, action),depth)
        return value
      
    return ExpMinimax_search(1,gameState.generateSuccessor(0, action),self.depth)
    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER
    Table = {}
    for action in gameState.getLegalActions(0):
      Table[action]=self.getQ(gameState,action)
    for k, v in Table.items():
      if v == max(Table.values()):
        return k
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def AlphaBeta_search(agent, gameState, depth, alpha, beta):
      valueList = []
      next_agent = agent+1
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)
      
      if agent == 0: #pacman
        for action in gameState.getLegalActions(agent):
          NewValue = AlphaBeta_search(next_agent,gameState.generateSuccessor(agent, action),depth,alpha,beta)
          valueList.append(NewValue)
          if NewValue > alpha: 
            alpha = NewValue #expected pacman choice
          if alpha >= beta: #expected pre agent choice
            break
        return max(valueList)
      
      elif agent%2 == 1: #oddNum ghost
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth = depth - 1
          
        for action in gameState.getLegalActions(agent):
          NewValue = AlphaBeta_search(next_agent,gameState.generateSuccessor(agent, action),depth,alpha,beta)
          valueList.append(NewValue)
          if NewValue < beta: 
            beta = NewValue #expected ghost choice
          if alpha >= beta: #expected pre agent choice
            break
        return min(valueList)
      else: #evenNum ghost
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth = depth - 1
        
        actionNum = len(gameState.getLegalActions(agent))
        phi = 1/actionNum
        value = 0

        for action in gameState.getLegalActions(agent):
          value += phi*AlphaBeta_search(next_agent,gameState.generateSuccessor(agent, action),depth,alpha,beta)
        return value
      
    return AlphaBeta_search(1,gameState.generateSuccessor(0, action),self.depth,-float('inf'), float('inf'))
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """

  # BEGIN_YOUR_ANSWER
  score = currentGameState.getScore()
  PacmanPosition = currentGameState.getPacmanPosition()
  GhostState = currentGameState.getGhostStates()
  food = currentGameState.getFood()
  capsules = currentGameState.getCapsules()

  d_ghost = []
  for ghost in GhostState:
    distance = manhattanDistance(PacmanPosition, ghost.getPosition())
    d_ghost.append(distance)
  
  d_food = []
  for target in food.asList():
    distance = manhattanDistance(PacmanPosition, target)
    d_food.append(distance)

  d_capsule = []
  for target in capsules:
    distance = manhattanDistance(PacmanPosition, target)
    d_capsule.append(distance)

  score -= 5*len(d_food) # Remaning food
  if d_food:
    score += 10/(min(d_food)+2)

  score -= 15*len(d_capsule) # Remaning capsule

  i=0
  for ghost in GhostState:
    if ghost.scaredTimer > 0:
      if d_ghost[i]<=1:
        score += 200
      else:
        score -= d_ghost[i]
        score += ghost.scaredTimer
    else:
      if d_ghost[i]<=1:
        score -= 350
    i += 1

  return score
  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER
  return 'AlphaBetaAgent'
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction
