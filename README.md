# ProxFly (UC Berkeley HiPeR Lab)
Hi there! This is the official codebase of our ICRA 2025 submission

[**ProxFly: Robust Control for Close Proximity Quadcopter Flight via Residual Reinforcement Learning**](https://arxiv.org/abs/2409.13193).

The multimedia is also available at Youtube.
[![Cover](figures/cover.png)](https://www.youtube.com/watch?v=NhPKgzd3l6w)

## Environment Setup
```
cd ProxFly
bash setup.bash
conda activate proxfly
python3 train.py
```

## Notation
The simulator used in this repo is a simplified version of HiPeR close-source C++ lab code. Hence it's common to get different conclusion from our paper, especially the running time.
For this, we also open the policy network we used for the large quadcopter in the paper at ``./data/model/``

## Citation
If you use this code in an academic context, please cite the following publication:

```
@misc{zhang2024proxfly,
      title={ProxFly: Robust Control for Close Proximity Quadcopter Flight via Residual Reinforcement Learning}, 
      author={Ruiqi Zhang and Dingqi Zhang and Mark W. Mueller},
      year={2024},
      eprint={2409.13193},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.13193}, 
}
```
