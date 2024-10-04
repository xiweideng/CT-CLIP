import numpy as np
import os
import pandas as pd
import argparse
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from transformer_maskgit import CTViT


def encode_texts_nan(reports, model, tokenizer, device):
    with torch.no_grad():
        text_latents_list = []
        for report in reports:
            try:
                report_tokens = tokenizer([report], return_tensors="pt",
                                          padding="max_length", truncation=True,
                                          max_length=512).to(device)
                text_embeddings = model.text_transformer(report_tokens['input_ids'],
                                                         attention_mask=report_tokens['attention_mask'])
                enc_text = text_embeddings[0]
                text_embeds = enc_text[:, :] if enc_text.ndim == 3 else enc_text
                text_embeds = text_embeds[:, 0, :]
                text_latents = model.to_text_latent(text_embeds)
                text_latents /= text_latents.norm(dim=-1, keepdim=True)
                text_latents_list.append(text_latents.cpu().numpy())
            except Exception as e:
                print(f"Error occurred with report: {report}")
                print(f"Error message: {str(e)}")
        return np.concatenate(text_latents_list, axis=0)


def encode_texts(reports, model, tokenizer, device):
    with torch.no_grad():
        report_tokens = tokenizer(reports, return_tensors="pt",
                                  padding="max_length", truncation=True,
                                  max_length=512).to(device)
        text_embeddings = model.text_transformer(report_tokens['input_ids'],
                                                 attention_mask=report_tokens['attention_mask'])
        enc_text = text_embeddings[0]
        text_embeds = enc_text[:, :] if enc_text.ndim == 3 else enc_text
        text_embeds = text_embeds[:, 0, :]
        text_latents = model.to_text_latent(text_embeds)
        text_latents /= text_latents.norm(dim=-1, keepdim=True)
    return text_latents.cpu().numpy()


def main(args):
    # set up device
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    # set model id
    model_id = "microsoft/BiomedVLP-CXR-BERT-specialized"
    tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=True)
    text_encoder = BertModel.from_pretrained(model_id)
    text_encoder.resize_token_embeddings(len(tokenizer))
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
        # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False

    )
    model_path = os.path.join(args.base_dir, args.clip_model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
    del model.to_visual_latent
    del model.to_visual_latent_extra
    del model.visual_transformer
    model = model.to(device)

    df = pd.read_csv(args.data_path)
    df = df.dropna(subset=["Report_EN"])  # Drop rows with nan in "Report_EN" column
    findings_impressions = df["Report_EN"].tolist()
    print(f"Number of reports: {len(findings_impressions)}")
    findings_impressions_size = len(findings_impressions)

    bs = args.batch_size
    num_batches = findings_impressions_size // bs
    tensors = []
    for i in tqdm(range(num_batches)):
        batch = findings_impressions[bs * i:bs * i + bs]
        _weights = encode_texts(batch, model, tokenizer, device)
        tensors.append(_weights)
    _weights = encode_texts(findings_impressions[bs * num_batches:], model, tokenizer, device)
    tensors.append(_weights)

    clip_embeddings = np.concatenate(tensors, axis=0)
    if not os.path.exists(args.base_dir):
        os.makedirs(args.base_dir)
    out_path = os.path.join(args.base_dir, args.out)
    np.savez(out_path, data=clip_embeddings)
    print(f"Embeddings have been saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate clip embeddings for a CT-RATE corpora (either sentence level or report level')
    parser.add_argument('--base_dir', type=str, default='/data4/dxw/checkpoints/ZUH/',
                        help='the parent directory of the checkpoint directory')
    parser.add_argument('--clip_model_name', type=str,
                        default='CT_CLIP_zeroshot.pt',
                        help='name of clip model state dictionary for generating embeddings')
    parser.add_argument('--data_path', type=str,
                        default="/data1/dxw/CT-RATE/dataset/radiology_text_reports/radiology_reports.csv",
                        help='path of csv file containing CT-RATE corpora (either sentence level or report level)')
    parser.add_argument('--out', type=str, default="clip_report_embeddings.npz",
                        help='name of output npz file containing clip embeddings')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for generating clip embeddings')
    args = parser.parse_args()

    main(args)
