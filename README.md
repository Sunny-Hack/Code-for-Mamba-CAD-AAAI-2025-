# Code-for-Mamba-CAD-(AAAI-2025)
Code for Mamba-CAD: State Space Model for 3D Computer-Aided Design Generative Modeling
## Environment

We recommend creating a new Conda environment to ensure the project runs correctly. 

### Create and Activate Conda Environment

Run the following commands to create a new environment:

```bash
conda create -n mcad python=3.9.1
conda activate mcad
pip install -r environment.txt
```
## Dataset

Please download the pre-processed dataset from the following link:
- (https://drive.google.com/file/d/1q7CXAyPOYBK94zitCpJg06YYkDZQpYh7/view?usp=drive_link)

## Train & Test

To train and evaluate the Mamba-CAD model, we provide a unified shell script that handles the entire pipeline.

Simply run the following command:

```bash
bash start.sh
```
## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{li2025mamba,
  title={Mamba-cad: State space model for 3d computer-aided design generative modeling},
  author={Li, Xueyang and Lou, Yunzhong and Song, Yu and Zhou, Xiangdong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={5},
  pages={5013--5021},
  year={2025}
}
