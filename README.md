# Pacman Q-Learning Agent

A reinforcement learning agent that learns to play Pacman through experience.
The agent improves over thousands of training episodes by updating Q-values for state-action pairs and then executes the best-known policies during gameplay.


## Overview
This project explores autonomous decision-making using model-free reinforcement learning.
A Q-Learning algorithm is implemented to estimate the long-term value of actions in a dynamic grid-world environment. Over training, the agent learns effective behavior for collecting food, avoiding ghosts, and winning games consistently. 

Key characteristics of the approach:
- Online Q-Learning with per step updates
- Count-based exploration combined with epsilon-greedy strategy
- Reward shaping to encourage survival, food collection, and ghost avoidance
- Automatic switch to pure exploitation after training
- Compatible with smallGrid and other Pacman layouts 


## Tech Stack

- Python 3
- Standard utility modules (no external ML library required)


## How to Run

1. Clone the repository:
```bash
git clone https://github.com/lynyi07/pacman-qlearning-agent.git
cd pacman-qlearning-agent
```
2. Train the agent and execute test episodes:
```bash
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid


### Explanation:
- First 2000 episodes: training (no GUI)
- Final 10 episodes: evaluation with GUI and printed results gameplay

### Optional hyperparameter overrides:
```bash
python3 pacman.py -p QLearnAgent -l smallGrid -a alpha=0.3 -a epsilon=0.1 -a gamma=0.9
``` 

## Visual Output (Example on Terminal)
1. Train the learner for 2000 episodes
```
Game 1992 just ended!
Game 1993 just ended!
Game 1994 just ended!
Game 1995 just ended!
Game 1996 just ended!
Game 1997 just ended!
Game 1998 just ended!
Game 1999 just ended!
Training Done (turning off epsilon and alpha)
```
2. Run 10 non-training episodes
```
Pacman emerges victorious! Score: 489
Game 2000 just ended!
Pacman emerges victorious! Score: 477
Game 2001 just ended!
Pacman emerges victorious! Score: 499
Game 2002 just ended!
Pacman emerges victorious! Score: 503
Game 2003 just ended!
```

The agent runs inside the Pacman game display:
<p align="center">
  <img src="assets\pacman-smallgrid-demo.gif" alt="Pacman Gameplay Demo" width="350"><br>
  <em>Q-Learning Agent Gameplay</em>
</p>
![Pacman SmallGrid Game Demo](assets\pacman-smallgrid-demo.gif)

3. Summary
```
Average Score: 493.0
Scores:        489.0, 477.0, 499.0, 503.0, 489.0, 493.0, 493.0, 503.0, 503.0, 481.0
Win Rate:      10/10 (1.00)
Record:        Win, Win, Win, Win, Win, Win, Win, Win, Win, Win.....
```


# License

This project includes modifications and extensions built on top of the open-source Pacman AI framework originally developed at the University of California, Berkeley for their Artificial Intelligence course.
The original Pacman materials are available at:
http://ai.berkeley.edu/


