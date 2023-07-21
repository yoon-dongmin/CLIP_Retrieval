import torch
from torchvision.models import resnet50
import clip


# 임의의 입력 데이터 생성
input_data = torch.randn(1, 3, 224, 224).cuda()

# ResNet50 모델 불러오기
# model = resnet50()
clip_model, preprocess = clip.load("RN50")
model = clip_model.visual
# print(model)

# 중간 계층의 출력을 저장하기 위한 리스트
intermediate_outputs = []

# hook 함수 정의
def hook(module, input, output):
    intermediate_outputs.append(output)

# 모델의 각 계층에 대해 hook 등록
for name, layer in model.named_children():
    layer.register_forward_hook(hook)

# Forward pass 수행
output = model(input_data)

# 각 계층에서의 출력 크기 출력
for i, out in enumerate(intermediate_outputs):
    print(f"Output of layer {i} is {out.shape}")
