# Knowledge Transfer for Acoustic Scene Classification

## Introduction 
This repo includes codes for (1) Our proposed Variational Bayesian Knowledge transfer (VBKT) algorithm, and (2) The implemetation of 13 recent cut-edging knowledge transfer (knowledge distillation / teacher-student learning) methods, including TSL, NLE, Fitnets, AT, AB, VID, FSP, COFD, SP, CCKD, PKT, NST, and RKD. More details can be referred to as in our paper [arxiv](https://arxiv.org/abs/2110.08598).

<img src="https://github.com/MihawkHu/ASC_Knowledge_Transfer/tree/main/fig/VBKT.png" width="500">  


## How to use   

### Environment Setup   
Tensorflow 1.14 and Keras 2.1. (via `pip install` or `conda install`)  

Noted - We use linux-ppc64le but it should be fine on other platforms follow the suggested version.    
> \$ conda env create -f environment.yml  

### Dataset   
We use DCASE 2020 Task 1a ASC data: [TAU Urban Acoustic Scenes 2020 Mobile, Development dataset](https://zenodo.org/record/3819968#.Yax3rfHP0-Q). Audio clips are grouped based on their recording devices. This repo focuses on the device adaptation problem, to transfer knowledge from the source domain (device A) to the target domain (device b, c, s1-s6).   

### Feature Extraction  
The acoustic features are extracted and dumped to local disk. Run the command below to extract log-mel filter bank (LMFB) features. Please specify the audio path.   
> \$ python tools/extr_feat_logmel.py   


### Model Training   
Two ASC models are covered in this repo: resnet and fcnn, based on [DCASE2020_task1](https://github.com/MihawkHu/DCASE2020_task1).  
- Train source/teacher model: `train_source.py`. Refer to the recipe `./scripts/run_source.sh` for parameter settings.  
- Train target/student model with knowledge transfer algorighms: `train_target.py`. Refer to the recipe `./scripts/run_target.sh` for detail parameter settings for each methods.  


### Pretrained Models     
We provide some pretrained models in `./pretrained_models/` as example. We have two pretrained resnet models with VBKT method on target device b and c.    

### Evaluation  
Use `./tools/eval.py` to evaluate a well-trained model on a target device. Example usages on pretrained models are shown below, should get 0.7212 and 0.7545, respectively.   
> \$ python tools/eval.py --model_path pretrained_models/model_resnet_vbkt_device-b.hdf5 --device b   
> \$ python tools/eval.py --model_path pretrained_models/model_resnet_vbkt_device-c.hdf5 --device c   


## Reference  
If you find this work useful, please consider to cite our paper. Thank you! Feel free to contact us for any quesitons or collaborations.  

```bib  
@article{hu2021variational,  
    title={A Variational Bayesian Approach to Learning Latent Variables for Acoustic Knowledge Transfer},   
    author={Hu, Hu and Siniscalchi, Sabato Marco and Yang, Chao-Han Huck and Lee, Chin-Hui},   
    journal={arXiv preprint arXiv:2110.08598},   
    year={2021},  
}
```


