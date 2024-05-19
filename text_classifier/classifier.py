import os

# set the environment variable for the HF mirror and the cache directory
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/data4/dxw/huggingface_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel


class RadBertClassifier(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()

        self.config = AutoConfig.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')
        self.model = AutoModel.from_pretrained('zzxslp/RadBERT-RoBERTa-4m', config=self.config)

        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attn_mask):
        output = self.model(input_ids=input_ids, attention_mask=attn_mask)
        output = self.classifier(output.pooler_output)

        return output


if __name__ == '__main__':
    tokenizer=AutoTokenizer.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')
    model_path = "/home/dxw/Desktop/common_datasets/checkpoints/ZUH/RadBertClassifier.pth"
    label_cols = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion',
                  'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy', 'Emphysema',
                  'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela',
                  'Pleural effusion', 'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation',
                  'Bronchiectasis', 'Interlobular septal thickening']
    num_labels = len(label_cols)
    print('Label columns: ', label_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    if device == 'cuda':
        n_gpu = torch.cuda.device_count()
        print("Number of GPU available:{} --> {} \n".format(n_gpu, torch.cuda.get_device_name()))
    model = RadBertClassifier(n_classes=num_labels)
    model.load_state_dict(torch.load(model_path))
    model = model.eval().to(device)
    text = ["There is a pleural effusion in the chest", "There is a cardiomegaly in the chest"]
    encoded_input = tokenizer(text, return_tensors='pt', max_length=512, padding='max_length').to(device)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']
    out = model(input_ids, attention_mask)
    print(out)