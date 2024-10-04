import pandas as pd
import os


def read_csv_file(file_path):
    """读取CSV文件并返回DataFrame"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file or directory: '{file_path}'")
    return pd.read_csv(file_path)


def merge_findings_and_impressions(df):
    """拼接Findings_EN和Impressions_EN为Report_EN，并删除原始字段"""
    df['Report_EN'] = df['Findings_EN'] + df['Impressions_EN']
    df.drop(columns=['Findings_EN', 'Impressions_EN'], inplace=True)
    return df


def combine_datasets(train_df, valid_df):
    """合并训练集和验证集"""
    return pd.concat([train_df, valid_df], ignore_index=True)


def save_csv_file(df, file_path):
    """保存DataFrame到CSV文件"""
    df.to_csv(file_path, index=False)
    print(f"Combined dataset saved to {file_path}")


def main():
    # 文件路径
    base_dir = '/data1/dxw/CT-RATE/dataset/radiology_text_reports/'
    train_file = os.path.join(base_dir, 'train_reports.csv')
    valid_file = os.path.join(base_dir, 'dataset_radiology_text_reports_validation_reports.csv')
    combined_file = os.path.join(base_dir, 'radiology_reports.csv')

    try:
        # 读取CSV文件
        train_df = read_csv_file(train_file)
        valid_df = read_csv_file(valid_file)
    except FileNotFoundError as e:
        print(e)
        return

    # 处理DataFrame
    train_df = merge_findings_and_impressions(train_df)
    valid_df = merge_findings_and_impressions(valid_df)

    # 合并DataFrame
    combined_df = combine_datasets(train_df, valid_df)

    # 保存合并后的数据集
    save_csv_file(combined_df, combined_file)


# 运行主函数
if __name__ == "__main__":
    main()
