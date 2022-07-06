#!/bin/bash

# reference: https://github.com/mathcom/constrained-graph-variational-autoencoder/blob/master/data/get_zinc.py"
FILE_PATH='250k_rndm_zinc_drugs_clean_3.csv'
SOURCE_URL='https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv'

echo "Downloading data to $FILE_PATH ..."
wget -O $FILE_PATH $SOURCE_URL
echo "Finished downloading"
