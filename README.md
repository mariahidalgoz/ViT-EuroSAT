# ViT

## Activate or set-up environment

Set-up the environment if it is not already configured.

1. Install *[pyenv](https://github.com/pyenv/pyenv)*
   and *[pipenv](https://github.com/pypa/pipenv)*.
   ```shell
   brew install pyenv
   brew install pipenv
   ```

2. Install python 3.10.9.
   ```shell
   pyenv install 3.10.9
   ```

3. Create virtual environment.
   ```shell
   pipenv --python $(pyenv root)/versions/3.10.9/bin/python
   ```

4. Install required packages from Pipfile.
   ```shell
   pipenv install
   ```

Or activate it if it is already configured.

```shell
pipenv shell
```

## Train and predict

### In your local machine

#### Train the ViT model

NOTE: This overwrites the weights and the execution may take a long time.

```shell
python architectures/ViT/train.py
```

#### Train the ViT model with different image and patch sizes

```shell
python architectures/ViT/train.py --image-size=32 --patch-size=4
```

#### Predict in a random set of EuroSAT image

```shell
python architectures/ViT/predict.py
```

#### Predict in a specific image

```shell
python architectures/ViT/predict.py one_image.png and_a_folder/*.jpg
```

```shell
python architectures/ViT/predict.py ./data/2750/Forest/Forest_1.jpg
```

#### Predict using a model for a specific image and patch size

```shell
python architectures/ViT/predict.py ./data/2750/Forest/Forest_1.jpg --image-size=32 --patch-size=4
```

Prediction list

| Land cover            | Id  |
|-----------------------|-----|
| Annual Crop           | 0   |
| Forest                | 1   |
| Herbaceous Vegetation | 2   |
| Highway               | 3   |
| Industrial            | 4   |
| Pasture               | 5   |
| Permanent Crop        | 6   |
| Residential           | 7   |
| River                 | 8   |
| Sea Lake              | 9   |

### In PALMA II

#### Login

```shell
ssh -i ~/.ssh/id_rsa_palma <username>@palma.uni-muenster.de
```

```shell
ssh -i ~/.ssh/id_rsa_palma mhidalgo@palma.uni-muenster.de
```

#### Send code to PALMA from local machine

```shell
rsync -avP -e "ssh -i ~/.ssh/id_rsa_palma" /path/to/local/folder <username>@palma.uni-muenster.de: /path/to/folder/on/palma
```

```shell
rsync -avP -e "ssh -i ~/.ssh/id_rsa_palma" /Users/m.hidalgo/vit/architectures mhidalgo@palma.uni-muenster.de: /home/m/mhidalgo/vit
rsync -avP -e "ssh -i ~/.ssh/id_rsa_palma" /Users/m.hidalgo/vit/submitViT.sh mhidalgo@palma.uni-muenster.de: /home/m/mhidalgo/vit
rsync -avP -e "ssh -i ~/.ssh/id_rsa_palma" /Users/m.hidalgo/vit/submitResNet50.sh mhidalgo@palma.uni-muenster.de: /home/m/mhidalgo/vit
```

#### Get results from PALMA to local machine

```shell
rsync -avP -e "ssh -i ~/.ssh/id_rsa_palma" <username>@palma.uni-muenster.de:/path/to/folder/on/palma /path/to/local/folder
```

```shell
rsync -avP -e "ssh -i ~/.ssh/id_rsa_palma" mhidalgo@palma.uni-muenster.de:/home/m/mhidalgo/vit/runs /Users/m.hidalgo/vit
rsync -avP -e "ssh -i ~/.ssh/id_rsa_palma" mhidalgo@palma.uni-muenster.de:/home/m/mhidalgo/vit/weights /Users/m.hidalgo/vit
```

#### Change config variables

Change config variables to your paths: PALMA_DIR

#### Train and predict

```shell
cd vit/
sbatch submitViT.sh --batch-size 256
sbatch submitViT.sh --batch-size 256
```

squeue -u <username>
squeue -u mhidalgo
scancel __job_id__

## Tensorboard

```shell
tensorboard --logdir runs
```

```shell
tensorboard --logdir runs_final
```

## Jupyter Notebooks

### Go to notebooks root

```shell
cd notebooks/
```

### Run Jupyter

```shell
jupyter lab
```

# Resources

## ResNet50 with EuroSAT

https://github.com/artemisart/EuroSAT-image-classification

## ViT with CiFAR10

https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html

## EuroSAT dataset

https://github.com/phelber/EuroSAT

## ViT

https://colab.research.google.com/github/hirotomusiker/schwert_colab_data_storage/blob/master/notebook/Vision_Transformer_Tutorial.ipynb#scrollTo=wSX8S6FLoEuA

## Detail explanation ViT

https://data-science-blog.com/blog/2021/04/07/multi-head-attention-mechanism/

## PyTorch ViT paper replicating

https://www.learnpytorch.io/08_pytorch_paper_replicating/#4-equation-1-split-data-into-patches-and-creating-the-class-position-and-patch-embedding

## Vision Transformer Tutorial Colab

https://colab.research.google.com/github/hirotomusiker/schwert_colab_data_storage/blob/master/notebook/Vision_Transformer_Tutorial.ipynb#scrollTo=wSX8S6FLoEuA

## Visualize attention map Torch

https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
https://epfml.github.io/attention-cnn/

## Visualize attention map tf

https://keras.io/examples/vision/probing_vits/

## Computer Vision

https://keras.io/examples/vision/

## Others ViT Sources

http://nlp.seas.harvard.edu/2018/04/03/attention.html
https://www.youtube.com/watch?v=TrdevFK_am4
https://github.com/BobMcDear/PyTorch-Vision-Transformer
https://theaisummer.com/self-attention/
https://theaisummer.com/vision-transformer/#:~:text=Attention%20distance%20was%20computed%20as,0.5%20the%20distance%20is%2010.
https://sh-tsang.medium.com/review-vision-transformer-vit-406568603de0
https://analyticsindiamag.com/hands-on-guide-to-using-vision-transformer-for-image-classification/
https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
https://medium.datadriveninvestor.com/coding-the-vision-transformer-in-pytorch-part-1-birds-eye-view-1c0a79d8732e
https://github.com/lajanugen/zeshel
https://twitter.com/hardmaru/status/1359323333720875008?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1359323333720875008%7Ctwgr%5E%7Ctwcon%5Es1_&ref_url=https%3A%2F%2Ftheaisummer.com%2Fself-attention%2F
http://peterbloem.nl/blog/transformers
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py

## Dropout

https://www.youtube.com/watch?v=ARq74QuavAo

## Transformer models

https://www.youtube.com/watch?v=iFhYwEi03Ew
https://huggingface.co/docs/transformers/model_doc/vit

## ViT Arquitectures

https://theaisummer.com/transformers-computer-vision/

## MLPClassifier

https://analyticsindiamag.com/a-beginners-guide-to-scikit-learns-mlpclassifier/
https://scikit-learn.org/stable/modules/neural_networks_supervised.html

## Others

http://neuralnetworksanddeeplearning.com/chap2.html

## Intuitive Explanation of Skip Connections in Deep Learning

https://theaisummer.com/skip-connections/

## Vision Transformer and its Applications

## ViT like U-Net for segmentation

https://www.youtube.com/watch?v=hPb6A92LROc
