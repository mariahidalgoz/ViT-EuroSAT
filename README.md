# ViT

## Train and predict in local
### Train the ViT model (This overwriten the wights)
python ViT/train.py
### Predict in a random set of EuroSAT image
python ViT/predict.py
### Predict in a specific image
python ViT/predict.py ./data/2750/Forest/Forest_1.jpg
./predict.py one_image.png and_a_folder/*.jpg

## Login
ssh -i ~/.ssh/id_rsa_palma <username>@palma.uni-muenster.de
ssh -i ~/.ssh/id_rsa_palma mhidalgo@palma.uni-muenster.de

squeue -u <username>
squeue -u mhidalgo
scancel __job_id__

## Get data
rsync -avP -e "ssh -i ~/.ssh/id_rsa_palma" <username>@palma.uni-muenster.de:/path/to/folder/on/palma /path/to/local/folder
rsync -avP -e "ssh -i ~/.ssh/id_rsa_palma" mhidalgo@palma.uni-muenster.de:/home/m/mhidalgo/vit/runs /Users/TemporaryAdmin/Documents/MLSI/ViT/vit
rsync -avP -e "ssh -i ~/.ssh/id_rsa_palma" mhidalgo@palma.uni-muenster.de:/home/m/mhidalgo/vit/weights /Users/TemporaryAdmin/Documents/MLSI/ViT/vit

## Send data
rsync -avP -e "ssh -i ~/.ssh/id_rsa_palma" /path/to/local/folder <username>@palma.uni-muenster.de:/path/to/folder/on/palma
rsync -avP -e "ssh -i ~/.ssh/id_rsa_palma" /Users/TemporaryAdmin/Documents/MLSI/ViT/vit mhidalgo@palma.uni-muenster.de:/home/m/mhidalgo 
rsync -avP -e "ssh -i ~/.ssh/id_rsa_palma" /Users/TemporaryAdmin/Documents/MLSI/ViT/vit/ViT/train.py mhidalgo@palma.uni-muenster.de:/home/m/mhidalgo/vit/ViT 

## Train and predict in PalmaII
cd vit/ 
sbatch submitViT.sh --batch-size 256
sbatch -u mhidalgo

## Tensorboard
tensorboard --logdir runs

# Resources

## ResNet50 with EuroSAT
https://github.com/artemisart/EuroSAT-image-classification

## ViT with CiFAR10
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html

## EuroSAT dataset
https://github.com/phelber/EuroSAT

## ViT
https://colab.research.google.com/github/hirotomusiker/schwert_colab_data_storage/blob/master/notebook/Vision_Transformer_Tutorial.ipynb#scrollTo=wSX8S6FLoEuA