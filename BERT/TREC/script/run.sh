#!/bin/bash

python -u inference.py --context_path ../dataset/model/context.json \
        --ckpt_DR_dir ../ckpt/ --data_path ../dataset/model/test.json \
        --ranked_list ../dataset/model/prediction.csv | tee ./log/"${1}" 