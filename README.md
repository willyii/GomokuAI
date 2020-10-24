# GomokuAI
This repository is tring to apply the principle of Alpha Go Zero to the Gomoku. Main idea is to combine the Monte Carlo Tree Search and CNN. 

The main idea can reference to [original paper](https://www.nature.com/articles/nature24270).


![Game](/imgs/Screen%20Shot%202020-08-27%20at%2014.40.44.png)

## Requirements

In order to run this project, you only need: 
- Pthon >= 3.5
- Tensorflow >= 1.11

## Getting start

Pre-trained model already saved in this repo, named **best_policy.hdf5**. You can using following command to run this project

```bash
git clone https://github.com/willyii/GomokuAI
cd GomokuAI
python human_play.py
```

Due to the hardware limitation. I only trained 8*8 chessboard, which is large 
enough for me to check if this idea work or not.

If you want to train your own model. You can change the board size in **train.py** and using following command to start training process:

```bash
python train.poy
```

The best model will be saved as **best_policy.hdf5** on root dictionary.


## Note 
1. It's good to start this project with small board size. It can help you to check if your idea works or not.
2. If you do not have GPU in your machine, it might take up to 30 second to take one action.
