#!/bin/bash
config_file="config/PCFNet_configure.yaml"
echo "Running the pipeline with the config file: $config_file"
echo "#########Preprocessing the data##########"
python src/preprocess_obs.py --config $config_file
python src/preprocess.py --config $config_file
echo "#########Training the model##########"
python src/train.py --config $config_file
python src/predict_obs.py --config $config_file