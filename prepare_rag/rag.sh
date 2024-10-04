#!/bin/bash

TOP_K=20
ANNOTATION_FILE=/data2/dxw/CTRG-ChestZ-npz/annotation.json
OUTPUT_FILE=/data2/dxw/CTRG-ChestZ-npz/annotation_top${TOP_K}.json
CUDA_VISIBLE_DEVICES=0,nohup python3 rag.py \
--topk $TOP_K \
--annotation_file $ANNOTATION_FILE \
--output_file $OUTPUT_FILE > rag_topk_$TOP_K.txt 2>&1 &