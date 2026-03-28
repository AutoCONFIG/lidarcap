#!/bin/bash

# Dataset preprocessing script for LidarCap
# This script converts raw data to HDF5 format for training

# Configuration
RAW_DATA_PATH="/media/yun/41306b47-5fbd-4e11-a4f8-13c59e123adf1/lidarhuman26M"
OUTPUT_PATH="/media/yun/41306b47-5fbd-4e11-a4f8-13c59e123adf1/lidarhuman26M"
SEQLEN=1
NPOINTS=512
TRAIN_IDS="5,6,8,25,26,27,28,30,31,32,33,34,35,36,37,38,39,40,42"
TEST_IDS="7,24,29,41"

# Modify the preprocessing script with correct paths
sed -i "s|ROOT_PATH = 'your_raw_data_path'|ROOT_PATH = '${RAW_DATA_PATH}'|g" datasets/preprocess/lidarcap.py
sed -i "s|extras_path = 'your_save_path'|extras_path = '${OUTPUT_PATH}'|g" datasets/preprocess/lidarcap.py

# Generate training set
echo "Generating training set..."
python datasets/preprocess/lidarcap.py dump \
    --seqlen ${SEQLEN} \
    --npoints ${NPOINTS} \
    --ids "${TRAIN_IDS}" \
    --name lidarcap_train

echo "Training HDF5 file generated at: ${OUTPUT_PATH}/lidarcap_train.hdf5"

# Generate test set
echo "Generating test set..."
python datasets/preprocess/lidarcap.py dump \
    --seqlen ${SEQLEN} \
    --npoints ${NPOINTS} \
    --ids "${TEST_IDS}" \
    --name lidarcap_test

echo "Test HDF5 file generated at: ${OUTPUT_PATH}/lidarcap_test.hdf5"
