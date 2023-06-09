# Pacman-AI-Project
> This repository contains our assignment for implementing an evaluation function and other search algorithms in Pacman. The goal is to develop an AI agent that can navigate through a maze, collect food pellets, avoid ghosts, and maximize its score.

## Students
- Samer haj
- Mohamad Sayed Ahmad
- Majd Assad
## How to Play

 - To start a game, type at the command line:
  ~~~~
   py pacman.py
  ~~~~

<p align="center">
<img src="https://github.com/Samerhajj/Pacman-AI-Project/blob/main/gifs/interactive.gif" width="540" />
</p>
 - Use the following keyboard controls to move Pacman:

   - W: Move Up
   - S: Move Down
   - A: Move Left
   - D: Move Right

---
# Reflex Agent
- Implemented BetterEvaluationFunction that considers both food locations and ghost locations

to try out the reflexAgent , use this
```bash
py pacman.py -p ReflexAgent
```
this will launch pacman using reflex agent. u can also modify it with some flags
```bash
-k [number] ==> number of ghosts. max is 4
-q ==> play game without graphics
-n [number] ==> play number of games in a row
-l [layout name] ==> play on a different layout.
```
Example of running in default medium layout with 1 ghost

![Reflex](images/reflexagent.PNG)

# MinMaxAgent
implementation of minmax agent in pacman, better winrate, but sometimes would lose.

to run the minmax agent:
```bash
py .\pacman.py -p MinimaxAgent -q -n 10 -k 3
```
- -q -n -k are some modifcations we can use. as mentioned above.
- this will run  minmax agent. 10 games in a row with 3 ghosts.

![Alt text](images/minimax.PNG)

# AlphaBetaAgent
implementation of alphabeta agent,
to run the alphabeta agent:
```bash
py .\pacman.py -p AlphaBetaAgent -q -n 10 -k 3
```
- -q -n -k are some modifcations we can use. as mentioned above.
- this will run  AlphaBeta agent. 10 games in a row with 3 ghosts.
![Alt text](images/AlphaBetaAgent.PNG)


---
# RandomExpectimaxAgent
- Implementing expectimax Agent using random, 
- all ghosts should be modeled as choosing uniformly at random from their legal moves.
- How to run
```bash
py .\pacman.py -p RandomExpectimaxAgent -q -n 10 -k 3
```
![Alt text](images/RandomAgent.PNG)

# DiretionalExpectimaxAgent
- Implementing expectimax Agent using Directional,
- all ghosts should be modeled as choosing uniformly at random from their legal moves.
- How to run
```bash
py .\pacman.py -p DirectionalExpectimaxAgent -q -n 10 -k 3
```
![Alt text](images/DirectionalExpectimax.PNG)
