# mlLearningAgents.py

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class for extracting relevant features from the game state.
    Used to represent the state in a simplified way for Q-learning.
    """
    def __init__(self, state: GameState):
        """
        Initializes GameStateFeatures with relevant key features from the given game state.
        
        Args:
            state (GameState): The current game state
        """
        self.pacman_position = state.getPacmanPosition() # Pacman’s coordinates
        self.ghost_positions = state.getGhostPositions() # List of ghost coordinates
        self.food = state.getFood() # Boolean grid representing food locations
        self.score = state.getScore() # Current game score 

    def __eq__(self, other):
        """Checks if two GameStateFeatures instances are equal (needed for dictionary lookups).
        Args:
            other (GameStateFeatures): Another instance to compare with
        Returns:
            bool: True if both instances represent the same game state, False otherwise
        """
        return (self.pacman_position == other.pacman_position and
                self.ghost_positions == other.ghost_positions and
                self.food == other.food)

    def __hash__(self):
        """Generates a unique hash value for a given state. 
        This allows GameStateFeatures to be used as dictionary keys.
        Returns:
            int: A unique hash value for the state
        """
        return hash((self.pacman_position, tuple(self.ghost_positions), self.food))


class QLearnAgent(Agent):
    """
    A Q-learning agent that learns to play Pacman by updating Q-values based on rewards.
    """
    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 2000):
        """
        Initializes the Q-learning agent with hyperparameters.
        
        Args:
            alpha (float): Learning rate
            epsilon (float): Exploration rate
            gamma (float): Discount factor
            maxAttempts (int): How many times to try each action in each state
            numTraining (int): Number of training episodes
        """
        super().__init__()

        # Q-learning hyperparameters 
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)

        # Training settings 
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0 # Tracks the number of episodes played

        # Q-learning storage 
        self.q_values = {}  # Stores Q-values for (state, action) pairs
        self.counts = {}  # Tracks visit counts for (state, action) pairs

        # Track the last state-action pair
        self.last_state = None
        self.last_state_features = None
        self.last_action = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Computes the reward for transitioning from startState to endState.

        Rewards and penalties:
        - Default movement: -1 
        - Winning the game: +5000 
        - Losing the game: -5000 
        - Eating a food pellet: +50 
        - Moving closer to food: +10 
        - Moving closer to a ghost: -200 

        Args:
            startState (GameState): The initial state before the action
            endState (GameState): The resulting state after the action

        Returns:
            float: The computed reward for the state transition
        """
        # Handle terminal state
        if endState.isWin():
            return 5000  
        if endState.isLose():
            return -5000  

        # Default movement penalty for unnecessary wandering
        reward = -1
        
        # Reward for eating food
        food_before = startState.getNumFood()
        food_after = endState.getNumFood()
        if food_after < food_before:
            reward += 50

        # Get Pacman positions 
        pacman_start = startState.getPacmanPosition()
        pacman_end = endState.getPacmanPosition()

        # Reward for moving closer to food 
        walls = endState.getWalls()
        food_positions = [(x, y) for x in range(walls.width) for y in range(walls.height) if endState.hasFood(x, y)]
        if food_positions:
            start_food_dist = min(util.manhattanDistance(pacman_start, food) for food in food_positions)
            end_food_dist = min(util.manhattanDistance(pacman_end, food) for food in food_positions)
            if end_food_dist < start_food_dist:  # Pacman moved closer to food
                reward += 10  

        # Penalize Pacman for getting closer to ghosts
        for ghost in endState.getGhostPositions():
            start_ghost_dist = util.manhattanDistance(pacman_start, ghost)
            end_ghost_dist = util.manhattanDistance(pacman_end, ghost)

            if end_ghost_dist < start_ghost_dist:  # Pacman moved toward a ghost
                reward -= 200  
        return reward

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Retrieves the Q-value for a given state-action pair.
        
        Args:
            state (GameStateFeatures): The current state
            action (Directions): The action taken from this state
        
        Returns:
            float: The stored Q-value for (state, action), or 0.0 if not found
        """
        return self.q_values.get((state, action), 0.0)


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Returns the highest Q-value among all possible actions in the given state.

        Args:
            state (GameStateFeatures): The current state representation

        Returns:
            float: The maximum Q-value among all possible actions, or 0.0 if no values are stored
        """
        possible_actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        return max([self.getQValue(state, action) for action in possible_actions], default=0.0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Updates the Q-value for a given (state, action) pair using the Q-learning formula.

        Q-learning update rule:
            Q(s, a) ← Q(s, a) + α * (reward + γ * maxQValue(nextState) - Q(s, a))
        
        Args:
            state (GameStateFeatures): The initial state before the action
            action (Directions): The action taken
            reward (float): The reward received after taking the action
            nextState (GameStateFeatures): The resulting state after the action
        """
        current_q = self.getQValue(state, action) # Retrieve the current Q-value for (state, action)
        max_future_q = self.maxQValue(nextState)  # Get the maximum Q-value of the next state (best possible future reward)
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q) # Apply the Q-learning update formula
        self.q_values[(state, action)] = new_q # Store the updated Q-value

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Increments the visitation count for a given (state, action) pair.

        Args:
            state (GameStateFeatures): The state before taking the action
            action (Directions): The action taken in the given state
        """
        self.counts[(state, action)] = self.getCount(state, action) + 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Retrieves the visitation count for a given (state, action) pair.

        Args:
            state (GameStateFeatures): The state before taking the action
            action (Directions): The action taken in the given state

        Returns:
            int: The number of times the action has been taken in the given state
        """
        return self.counts.get((state, action), 0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes the exploration function to balance exploration and exploitation.

        Args:
            utility (float): The expected utility (Q-value) of taking an action in a given state
            counts (int): The number of times the (state, action) pair has been visited

        Returns:
            float: A modified value that encourages exploration for less-visited actions
        """
        if counts == 0:
            return 10  # Encourage exploring unvisited actions
        return utility + (1.0 / (counts + 1))  # Slightly favor less-visited actions)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Chooses an action that balances exploration and exploitation.
        The agent follows an epsilon-greedy strategy with count-based exploration:
        - With probability epsilon, it selects a random action (exploration)
        - Otherwise, it selects the action with the highest Q-value, adjusted with an exploration bonus (exploitation)
        - The exploration bonus is higher for actions that have been taken fewer times in a given state.

        Args:
            state (GameState): The current game state

        Returns:
            Directions: The chosen action
        """
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        state_features = GameStateFeatures(state)

        # Update Q-values based on the last action taken 
        if self.last_state is not None and self.last_action is not None:
            reward = self.computeReward(self.last_state, state) # Compute reward for last action
            self.learn(self.last_state_features, self.last_action, reward, state_features) # Update Q-table

        # Choose action based on epsilon-greedy exploration 
        if util.flipCoin(self.epsilon):
            action = random.choice(legal)  # Explore: choose a random action
        else:
            # Exploit: choose the best action based on Q-values adjusted by count-based exploration
            action = max(legal, key=lambda a: self.explorationFn(
                self.getQValue(state_features, a), self.getCount(state_features, a)
            ))

        self.updateCount(state_features, action) # Update visit counts for the selected action
        
        # Store the last action and state for the next learning update
        self.last_state = state
        self.last_state_features = state_features
        self.last_action = action
        return action
        

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state (GameState): The final state at the end of the episode
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Learn from the final state
        if self.last_state is not None and self.last_action is not None:
            reward = self.computeReward(self.last_state, state) # Compute the final reward
            state_features = GameStateFeatures(state)
            self.learn(self.last_state_features, self.last_action, reward, state_features)  # Final Q-learning update

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()

        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)

        # Reset tracking variables for the next episode
        self.last_state = None
        self.last_state_features = None
        self.last_action = None