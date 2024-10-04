import argparse
import json
import numpy as np
import torch
from tqdm import tqdm

from ct_clip import CTCLIP
from ct_dataset import create_dataloader
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel


def init_model(model_path, device):
    model_id = "microsoft/BiomedVLP-CXR-BERT-specialized"

    tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=True)
    text_encoder = BertModel.from_pretrained(model_id)
    text_encoder.resize_token_embeddings(len(tokenizer))
    # 加载预训练的图像编码器
    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=30,
        temporal_patch_size=15,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8
    )

    model = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=2097152,
        dim_text=768,
        dim_latent=512,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False
    )

    model.load_state_dict(torch.load(model_path, map_location=device))

    # 删除文本相关分支
    del model.to_text_latent
    del model.to_text_latent_extra
    del model.text_transformer

    model = model.to(device)
    return model


def predict(loader, text_embeddings, model, device, topk=1):
    predicted_corpus_indices = torch.zeros([len(loader.dataset), topk]).to(device)
    batch_index = 0
    # 打印一下数据集总大小以确认
    print(f"Total dataset size for prediction: {len(loader.dataset)}")

    # 验证 batch 数目以确保索引对应
    print(f"Total number of batches: {len(loader)}")
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            images = data['image'].to(device)

            # predict
            enc_image = model.visual_transformer(images, return_encoded_tokens=True)
            enc_image = enc_image.view(enc_image.shape[0], -1)
            image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image
            image_latents = model.to_visual_latent(image_embeds)
            image_latents /= image_latents.norm(dim=-1, keepdim=True)

            logits = image_latents @ text_embeddings.T
            """
            image_latents: [batch_size, 512], text_embeddings.T: [512, 50158]
            image_latents @ text_embeddings.T: [batch_size, 50158] 
            which means the similarity between image_latents and text_embeddings
            """
            preds = torch.argsort(logits, dim=-1, descending=True)[:, :topk]  # get topk reports

            predicted_corpus_indices[batch_index:batch_index + preds.size(0), :] = preds  # save batch to predictions

            batch_index += preds.size(0)  # batch size
    return predicted_corpus_indices.to('cpu').numpy()


def add_predictions_to_annotations(annotation_file, predictions, out_file):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # add predictions to annotations
    data = []
    for key in annotations.keys():
        data += annotations[key]

    for i, prediction in enumerate(predictions):
        data[i]['clip_indices'] = prediction.astype(int).tolist()

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)
    # 打印验证已插入数据数量
    print(f"Total entries with added predictions: {len(predictions)}")


# 主函数调用
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and add clip indices to annotations.')
    parser.add_argument('--base_dir', type=str, default='/data4/dxw/CTRG-Chest-548K-3D-npz/',
                        help='the parent directory of the checkpoint directory')
    parser.add_argument('--clip_model_name', type=str,
                        default='/data4/dxw/checkpoints/ZUH/CT_CLIP_zeroshot.pt',
                        help='name of clip model state dictionary for generating embeddings')
    parser.add_argument('--annotation_file', type=str,
                        default="/data1/dxw/CTRG-Chest-548K-3D-npz/annotation_promptmrg_cxr_c14.json",
                        help='path to the annotation file')
    parser.add_argument('--output_file', type=str,
                        default="/data4/dxw/CTRG-Chest-548K-3D-npz/annotation_cct_c18_clip_indices.json",
                        help='path to save the annotations with predictions')
    parser.add_argument('--text_embeddings_file', type=str,
                        default="/data4/dxw/checkpoints/ZUH/clip_report_embeddings.npz",
                        help='file containing the text embeddings')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for generating clip embeddings')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--topk', type=int, default=20, help='Top-k predictions to add to annotations')
    args = parser.parse_args()
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载text_embeddings
    text_embeddings = np.load(args.text_embeddings_file)['data']
    text_embeddings = torch.tensor(text_embeddings).to(device)

    # 初始化数据加载器
    dataloader = create_dataloader(args.annotation_file, args.base_dir, args.num_workers, args.batch_size)

    # 初始化模型
    model = init_model(args.clip_model_name, device)

    # 预测
    predictions = predict(dataloader, text_embeddings, model, device, topk=args.topk)

    # 保存带有预测结果的annotations
    add_predictions_to_annotations(args.annotation_file, predictions, args.output_file)
