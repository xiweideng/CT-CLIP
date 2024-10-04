import json
import csv


def load_json_file(filepath):
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
    return data


def load_csv_file(filepath):
    with open(filepath, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # 跳过表头行
        return list(csv_reader)


def add_labels_to_data(data, csv_data, key):
    for row, obj in zip(csv_data, data[key]):
        labels = [int(x) for x in row[1:]]
        obj['labels'] = labels
    return data


def write_json_file(filepath, data):
    with open(filepath, 'w') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


def main():
    json_filepath = '/data2/dxw/CTRG-ChestZ-npz/annotation_top20.json'
    # train_csv_filepath = '/home/dxw/PycharmProjects/CT-CLIP/text_classifier/CTRG-Chest-3D/train.csv'
    val_csv_filepath = '/data2/dxw/CTRG-ChestZ-npz/val_label.csv'

    data = load_json_file(json_filepath)
    # train_csv_data = load_csv_file(train_csv_filepath)
    val_csv_data = load_csv_file(val_csv_filepath)

    # data = add_labels_to_data(data, train_csv_data, 'train')
    data = add_labels_to_data(data, val_csv_data, 'val')

    write_json_file(json_filepath, data)


if __name__ == "__main__":
    main()
