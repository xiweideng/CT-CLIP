import os
import json
import csv

# 目标目录路径
target_dir = "/home/dxw/Desktop/common_datasets/CTRG-Chest-548K-3D-npz/"

# 读取annotation.json文件
with open(os.path.join(target_dir, "annotation_cxr_c14.json"), "r") as f:
    data = json.load(f)

# 处理train和val数据
for split in ["train", "val"]:
    # 创建对应的csv文件
    csv_file = os.path.join(target_dir, f"{split}.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        # 写入表头
        writer.writerow(["AccessionNo", "report"])

        # 写入数据
        for item in data[split]:
            writer.writerow([item["id"], '"' + item["impressions_en"] + '"'])

print("CSV files have been created successfully.")