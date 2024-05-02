from typing import Callable, List, Set

import shell
import util
import wordsegUtil


############################################################
# Problem 1a: Solve the segmentation problem under a unigram model

class WordSegmentationProblem(util.SearchProblem):
    def __init__(self, query: str, unigramCost: Callable[[str], float]):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0
        raise NotImplementedError
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == len(self.query)
        raise NotImplementedError
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        result=[]
        for i in range (state+1, len(self.query)+1):
            action = self.query[state:i]
            new_state = i
            cost = self.unigramCost(action)
            result. append((action,new_state,cost))
        return result
        raise NotImplementedError
        # END_YOUR_CODE


def segmentWords(query: str, unigramCost: Callable[[str], float]) -> str:
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(WordSegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
    return ' '.join(ucs.actions)
    raise NotImplementedError
    # END_YOUR_CODE


############################################################
# Problem 1b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords: List[str], bigramCost: Callable[[str, str], float],
            possibleFills: Callable[[str], Set[str]]):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (0,wordsegUtil.SENTENCE_BEGIN) #(state, pre_action)
        raise NotImplementedError
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[0] == len(self.queryWords)
        raise NotImplementedError
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        result=[]
        order = state[0]
        # print(state,self.queryWords[order])
        possibleAct = self.possibleFills(self.queryWords[order])
        # print(possibleAct) 
        
        if len(possibleAct) == 0:
            possibleAct = set()
            possibleAct.add(self.queryWords[order])

        for action in possibleAct:
            # print(order,state[1],action)
            # print((action,order+1,self.bigramCost(state[1],action)))
            result.append((action,(order+1,action),self.bigramCost(state[1],action)))

        return result
        raise NotImplementedError
        # END_YOUR_CODE


def insertVowels(queryWords: List[str], bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]]) -> str:
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    # print(queryWords)

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    # print(queryWords, ", check: 2")

    return ' '.join(ucs.actions)
    raise NotImplementedError
    # END_YOUR_CODE


############################################################
# Problem 1c: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query: str, bigramCost: Callable[[str, str], float],
            possibleFills: Callable[[str], Set[str]]):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (0,wordsegUtil.SENTENCE_BEGIN)
        raise NotImplementedError
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[0] == (len(self.query))
        raise NotImplementedError
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        result = []
        order = state[0]
        for i in range(1, len(self.query)+1):
            newState = order + i
            _action = self.query[order:newState]
            possinleAct = self.possibleFills(_action)
            for action in possinleAct:
                result.append((action,(order+i,action),self.bigramCost(state[1],action)))

        return result
        raise NotImplementedError
        # END_YOUR_CODE


def segmentAndInsert(query: str, bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]]) -> str:
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))

    return ' '.join(ucs.actions)
    raise NotImplementedError
    # END_YOUR_CODE


############################################################
# Problem 2a: Solve the maze search problem with uniform cost search

class MazeProblem(util.SearchProblem):
    def __init__(self, start: tuple, goal: tuple, moveCost: Callable[[tuple, tuple], float],
            possibleMoves: Callable[[tuple], Set[tuple]]) -> float:
        self.start = start
        self.goal = goal
        self.moveCost = moveCost
        self.possibleMoves = possibleMoves

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.start
        raise NotImplementedError
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == self.goal
        raise NotImplementedError
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        result = []
        # print("state is", state)
        for move in self.possibleMoves(state):
            result.append((move[0], move[1], self.moveCost(state,move[0])))

        return result
        raise NotImplementedError
        # END_YOUR_CODE
            

def UCSMazeSearch(start: tuple, goal: tuple, moveCost: Callable[[tuple, tuple], float],
            possibleMoves: Callable[[tuple], Set[tuple]]) -> float:
    ucs = util.UniformCostSearch()
    ucs.solve(MazeProblem(start, goal, moveCost, possibleMoves))
    
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    i=0
    for step in ucs.actions:
        i+=1
    return i
    raise NotImplementedError
    # END_YOUR_CODE


############################################################
# Problem 2b: Solve the maze search problem with A* search

def consistentHeuristic(goal: tuple):
    def _consistentHeuristic(state: tuple) -> float:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        raise NotImplementedError
        # END_YOUR_CODE
    return _consistentHeuristic

def AStarMazeSearch(start: tuple, goal: tuple, moveCost: Callable[[tuple, tuple], float],
            possibleMoves: Callable[[tuple], Set[tuple]]) -> float:
    ucs = util.UniformCostSearch()
    ucs.solve(MazeProblem(start, goal, moveCost, possibleMoves), heuristic=consistentHeuristic(goal))
    
    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    raise NotImplementedError
    # END_YOUR_CODE

############################################################


if __name__ == '__main__':
    shell.main()
