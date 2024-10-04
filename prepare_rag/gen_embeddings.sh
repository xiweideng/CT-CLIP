#!/bin/bash

# 获取输入报告的csv文件路径
DATA_PATH="/data1/dxw/CT-RATE/dataset/radiology_text_reports/radiology_reports.csv"

# 获取csv文件所在的目录
DIR_PATH=$(dirname "$DATA_PATH")

# 设置log文件的路径
LOG_PATH="$DIR_PATH/gen_embeddings.log"

# 运行Python脚本并将输出和错误重定向到log文件
python3 gen_report_embedding_multiprocess.py > "$LOG_PATH" 2>&1 &