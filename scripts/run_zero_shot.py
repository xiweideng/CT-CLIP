import os

# set the environment variable for the HF mirror and the cache directory
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/data4/dxw/huggingface_cache/"
import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP


# device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',
                                          do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

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

clip = CTCLIP(
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
base_dir = "/data4/dxw/checkpoints/ZUH"
clip.load(os.path.join(base_dir, "CT_CLIP_zeroshot.pt"))
clip.to(device)
# # 获取图像编码器的权重
# image_encoder_weights = clip.visual_transformer.state_dict()
#
# # 保存图像编码器的权重
# torch.save(image_encoder_weights, os.path.join(base_dir, 'CTViT_enc_zeroshot.pt'))

# # 获取文本编码器的权重
# text_encoder_weights = clip.text_transformer.state_dict()
#
# # 保存文本编码器的权重
# torch.save(text_encoder_weights, os.path.join(base_dir, 'Bert_enc_zeroshot.pt'))
"""
CTCLIP(
  (text_transformer): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.25, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.25, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.25, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.25, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (visual_transformer): CTViT(
    (spatial_rel_pos_bias): ContinuousPositionBias(
      (net): ModuleList(
        (0): Sequential(
          (0): Linear(in_features=2, out_features=512, bias=True)
          (1): LeakyReLU(negative_slope=0.1)
        )
        (1): Sequential(
          (0): Linear(in_features=512, out_features=512, bias=True)
          (1): LeakyReLU(negative_slope=0.1)
        )
        (2): Linear(in_features=512, out_features=8, bias=True)
      )
    )
    (to_patch_emb_first_frame): Sequential(
      (0): Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)', p1=30, p2=30)
      (1): LayerNorm((900,), eps=1e-05, elementwise_affine=True)
      (2): Linear(in_features=900, out_features=512, bias=True)
      (3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (to_patch_emb): Sequential(
      (0): Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)', p1=30, p2=30, pt=15)
      (1): LayerNorm((13500,), eps=1e-05, elementwise_affine=True)
      (2): Linear(in_features=13500, out_features=512, bias=True)
      (3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (enc_spatial_transformer): Transformer(
      (layers): ModuleList(
        (0-3): 4 x ModuleList(
          (0): PEG(
            (dsconv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), groups=512)
          )
          (1): Attention(
            (attn_dropout): Dropout(p=0.0, inplace=False)
            (norm): LayerNorm()
            (context_norm): LayerNorm()
            (to_q): Linear(in_features=512, out_features=256, bias=False)
            (to_kv): Linear(in_features=512, out_features=512, bias=False)
            (to_out): Linear(in_features=256, out_features=512, bias=False)
          )
          (2): None
          (3): Sequential(
            (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (1): Linear(in_features=512, out_features=2730, bias=False)
            (2): GEGLU()
            (3): Dropout(p=0.0, inplace=False)
            (4): Linear(in_features=1365, out_features=512, bias=False)
          )
        )
      )
      (norm_out): LayerNorm()
    )
    (enc_temporal_transformer): Transformer(
      (layers): ModuleList(
        (0-3): 4 x ModuleList(
          (0): PEG(
            (dsconv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), groups=512)
          )
          (1): Attention(
            (attn_dropout): Dropout(p=0.0, inplace=False)
            (norm): LayerNorm()
            (context_norm): LayerNorm()
            (to_q): Linear(in_features=512, out_features=256, bias=False)
            (to_kv): Linear(in_features=512, out_features=512, bias=False)
            (to_out): Linear(in_features=256, out_features=512, bias=False)
          )
          (2): None
          (3): Sequential(
            (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (1): Linear(in_features=512, out_features=2730, bias=False)
            (2): GEGLU()
            (3): Dropout(p=0.0, inplace=False)
            (4): Linear(in_features=1365, out_features=512, bias=False)
          )
        )
      )
      (norm_out): LayerNorm()
    )
    (vq): VectorQuantize(
      (project_in): Identity()
      (project_out): Identity()
      (_codebook): CosineSimCodebook()
    )
    (to_pixels_first_frame): Sequential(
      (0): Linear(in_features=512, out_features=900, bias=True)
      (1): Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)', p1=30, p2=30)
    )
    (to_pixels): Sequential(
      (0): Linear(in_features=512, out_features=13500, bias=True)
      (1): Rearrange('b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)', p1=30, p2=30, pt=15)
    )
  )
  (to_text_latent): Linear(in_features=768, out_features=512, bias=False)
  (to_visual_latent): Linear(in_features=2097152, out_features=512, bias=False)
  (to_text_latent_extra): Linear(in_features=768, out_features=512, bias=False)
  (to_visual_latent_extra): Linear(in_features=2097152, out_features=512, bias=False)
)
"""
#
# inference = CTClipInference(
#     clip,
#     data_folder='path_to_preprocessed_validation_folder',
#     reports_file="path_to_validation_reports_csv",
#     labels="path_to_validation_labels_csv",
#     batch_size=1,
#     results_folder="inference_zeroshot/",
#     num_train_steps=1,
# )
#
# inference.infer()
