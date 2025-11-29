import torch
from PIL import Image
import open_clip
from pathlib import Path

# ===== 設定 =====
MODEL_NAME = "ViT-B-16"
# ルートに置いた open_clip_model.safetensors へのパス
CKPT_PATH = Path("open_clip_model.safetensors")
IMAGE_PATH = Path("test.png")  # 適当なテスト画像に差し替え

# ===== モデル読み込み =====
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

model, preprocess, _ = open_clip.create_model_and_transforms(
    model_name=MODEL_NAME,
    pretrained=str(CKPT_PATH),
)
model = model.to(device)
model.eval()

# ===== 画像前処理 =====
img = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = preprocess(img).unsqueeze(0).to(device)  # (1, 3, H, W)

# ===== 埋め込み計算 =====
with torch.no_grad():
    image_features = model.encode_image(image_tensor)  # (1, D)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

print("embedding shape:", image_features.shape)
print("first 5 dims:", image_features[0, :5].cpu().numpy())
