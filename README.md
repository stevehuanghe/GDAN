# GDAN

Code for the paper  ["Generative Dual Adversarial Network for Generalized Zero-shot Learning"](https://arxiv.org/abs/1811.04857)

Dataset is available [here](<https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/>), please download and extract it under ./data and make sure the "data_root" variable in config yaml file is correct.

### Dependencies

- Python 3.6, PyTroch 0.4
- sklearn, scipy, numpy, tqdm

### Steps to run the program

1. Modify config files as needed.

2. Pretrain CVAE, and checkpoints are stored in the directory specified by  "vae_dir" in config file: 

   ```bash
   python pretrain_gdan.py --config configs/cub.yml
   ```

3. Choose which CVAE checkpoint you want to use to initialize the GDAN model and modify the "vae_ckpt" variable in the yaml file. Then we can train GDAN by running:

   ```bash
   python train_gdan.py --config configs/cub.yml
   ```

4. Use validation data to  decide which saved checkpoint of GDAN (directory specified by "ckpt_dir" in config yaml file) to be used for testing and run evaluation:

   ```bash
   python valtest_gdan.py --config configs/cub.yml
   ```

Bibtex:

```
@InProceedings{Huang_2019_CVPR,
author = {Huang, He and Wang, Changhu and Yu, Philip S. and Wang, Chang-Dong},
title = {Generative Dual Adversarial Network for Generalized Zero-Shot Learning},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
