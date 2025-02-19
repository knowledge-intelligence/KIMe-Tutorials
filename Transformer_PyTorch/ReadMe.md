# Transformer (PyTorch from Scratch) Tutorials

### Create Conda
```shell
conda create -n tformer python=3.11
```

## Pytorch (Locally)
### CUDA 11.8
```shell
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```
### CUDA 12.1 (*)
```shell
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
### CPU Only
```shell
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 cpuonly -c pytorch
```

## Additional (*)
```shell
conda install pandas numpy matplotlib tensorboard
conda install -c pytorch torchdata
conda install -c conda-forge torchmetrics spacy tqdm altair
conda install -c huggingface -c conda-forge datasets tokenizers
```


## For Jupyter Notebook
```shell
conda install -c conda-forge ipywidgets
conda install -c anaconda ipython
```

### Reference
1. [github.com/ES7/Transformer-from-Scratch](https://github.com/ES7/Transformer-from-Scratch)
2. [requirements.txt - conda, pip](https://maxo.tistory.com/115)