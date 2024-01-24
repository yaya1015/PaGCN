# PaGCNs
This is a PyTorch implementation of "PaGCN: Incomplete Graph Learning via Partial Graph Convolutional Network".

## Requirements
- pytorch=1.4.0+cu100
- scikit-learn=0.22.1
- networkx=2.4
- scipy=1.4.1
- tqdm=4.42.1

## Usage
Run PaGCN model using:

```python main.py --dataset amaphoto --type Type1 --rate 0.5```



## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Taguchi & Hibiki, Graph Convolutional Networks for Graphs Containing Missing Features, 2021](https://arxiv.org/abs/2007.04583)
