# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from itertools import chain

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    initial_state = problem.getStartState()
    frontier = util.Stack()
    frontier.push((initial_state, [])) # save the state and the direction
    visited = set()

    while not frontier.isEmpty():

        actual_state, path = frontier.pop()
        if problem.isGoalState(actual_state):
            return path

        visited.add(actual_state)
        for state, direction, cost in problem.getSuccessors(actual_state):
            if state not in visited:
                frontier.push((state, path + [direction]))

    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first.

    frontier = util.Queue()
    visited = dict()

    state = problem.getStartState()
    node = {}
    node["parent"] = None
    node["action"] = None
    node["state"] = state
    frontier.push(node)

    while not frontier.isEmpty():
        node = frontier.pop()
        state = node["state"]
        if state in visited.keys():
            continue

        visited[state] = True
        if problem.isGoalState(state) == True:
            break

        for child in problem.getSuccessors(state):
            if child[0] not in visited:
                sub_node = {}
                sub_node["parent"] = node
                sub_node["state"] = child[0]
                sub_node["action"] = child[1]
                frontier.push(sub_node)

    actions = []
    while node["action"] != None:
        actions.insert(0, node["action"])
        node = node["parent"]

    return actions

    """
    "*** YOUR CODE HERE ***"

    # 269
    initial_state = problem.getStartState()
    frontier = util.Queue()
    frontier.push((initial_state, []))

    frontier_nodes = {}
    frontier_nodes[initial_state] = True

    visited = set()

    while not frontier.isEmpty():

        actual_state, path = frontier.pop()
        del frontier_nodes[actual_state]

        if problem.isGoalState(actual_state):
            return path

        visited.add(actual_state)
        for state, direction, cost in problem.getSuccessors(actual_state):
            if state not in visited:
                if state not in frontier_nodes.keys():
                    frontier.push((state, path + [direction]))
                    frontier_nodes[state] = True
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first.
    initial_state = problem.getStartState()
    frontier = util.PriorityQueue()
    frontier.push((initial_state, [], 0), 0)  # save the state and the direction

    frontier_nodes = {}
    frontier_nodes[initial_state] = True

    visited = set()

    while not frontier.isEmpty():

        actual_state, path, cost = frontier.pop()
        del frontier_nodes[actual_state]

        if problem.isGoalState(actual_state):
            return path

        visited.add(actual_state)
        for state, direction, cost_child in problem.getSuccessors(actual_state):
            if state not in visited:
                if state not in frontier_nodes.keys():
                    frontier.push((state, path + [direction], cost + cost_child), cost + cost_child)
                    frontier_nodes[state] = True

    return []

    initial_state = problem.getStartState()
    frontier = util.PriorityQueue()
    frontier.push(initial_state, 0)  # save the state and the direction

    frontier_nodes = {}
    frontier_nodes[initial_state] = {}
    frontier_nodes[initial_state]['cost'] = 0
    frontier_nodes[initial_state]['path'] = []

    visited = set()

    best_cost = float('inf')
    best_path = None

    while not frontier.isEmpty():



        actual_state = frontier.pop()
        actual_cost = frontier_nodes[actual_state]['cost']
        actual_path = frontier_nodes[actual_state]['path']

        del frontier_nodes[actual_state]

        if (actual_cost < best_cost) and (problem.isGoalState(actual_state)):
            best_cost = actual_cost
            best_path = actual_path

        visited.add(actual_state)
        for state, direction, cost_child in problem.getSuccessors(actual_state):
            if state not in visited:
                if state not in frontier_nodes.keys():
                    frontier.push(state, actual_cost + cost_child)
                    frontier_nodes[state] = {}
                    frontier_nodes[state]['cost'] = actual_cost + cost_child
                    frontier_nodes[state]['path'] = actual_path + [direction]
                else:
                    frontier.update(state, actual_cost + cost_child)
                    frontier_nodes[state] = {}
                    frontier_nodes[state]['cost'] = actual_cost + cost_child
                    frontier_nodes[state]['path'] = actual_path + [direction]

    return best_path

    """
    "*** YOUR CODE HERE ***"
    initial_state = problem.getStartState()
    frontier = util.PriorityQueue()
    frontier.push(initial_state, 0)  # save the state and the direction

    frontier_nodes = {}
    frontier_nodes[initial_state] = {}
    frontier_nodes[initial_state]['cost'] = 0
    frontier_nodes[initial_state]['path'] = []

    visited = set()

    while not frontier.isEmpty():

        actual_state = frontier.pop()
        actual_cost = frontier_nodes[actual_state]['cost']
        actual_path = frontier_nodes[actual_state]['path']

        del frontier_nodes[actual_state]

        if problem.isGoalState(actual_state):
            return actual_path

        visited.add(actual_state)
        for state, direction, cost_child in problem.getSuccessors(actual_state):
            if state not in visited:
                if state not in frontier_nodes.keys():
                    frontier.update(state, actual_cost + cost_child)
                    frontier_nodes[state] = {}
                    frontier_nodes[state]['cost'] = actual_cost + cost_child
                    frontier_nodes[state]['path'] = actual_path + [direction]

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    initial_state = problem.getStartState()

    frontier = util.PriorityQueue()
    frontier.push((initial_state, [], heuristic(initial_state, problem)), heuristic(initial_state, problem))  # save the state and the direction

    frontier_nodes = {}
    frontier_nodes[initial_state] = True

    visited = set()

    while not frontier.isEmpty():

        actual_state, path, cost = frontier.pop()
        del frontier_nodes[actual_state]

        if problem.isGoalState(actual_state):
            return path
        visited.add(actual_state)
        for state, direction, cost_child in problem.getSuccessors(actual_state):
            if state not in visited:
                if state not in frontier_nodes.keys():
                    frontier.update((state, path + [direction], cost + heuristic(state, problem)),
                                    cost + heuristic(state, problem))
                    frontier_nodes[state] = True

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
