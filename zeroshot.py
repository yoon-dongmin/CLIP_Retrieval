import numpy as np
import torch
from torch import nn
import clip
from tqdm.notebook import tqdm
from pkg_resources import packaging
import torchvision
from torch.utils.data import DataLoader, Dataset # 데이터를 모델에 사용할 수 있도록 정리해 주는 라이브러리
import torchvision.transforms as tr
import csv
from PIL import Image
import os
import torch.nn.functional as F

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class MyDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.image_path = []
        for f in os.listdir(x_data): #랜덤하게 읽어옴 수정 필요
            self.image_path.append(os.path.join(x_data, f))
        
        self.image_path.sort()
            
            #input_image = Image.open(os.path.join(dir, f))
        #print(Image_path)
        
        with open(y_data, 'r') as f:
            text_data = list(csv.reader(f))
        self.y_data = text_data # 넘파이 배열이 들어온다.
        #print(self.y_data,123123)
        self.transform = transform
        self.len = len(y_data)


    def __getitem__(self, index):
        input_image = Image.open(self.image_path[index])
        label = int(self.y_data[index][0])
        label = label - 1
        #sample = input_image, label
        
        if self.transform:
            input_image = self.transform(input_image) #self.transform이 None이 아니라면 전처리를 작업한다.
        
        return input_image, label # 3.3과 다르게 넘파이 배열로 출력 되는 것에 유의 하도록 한다.
    
    def __len__(self):
        return self.len       


model, preprocess = clip.load("RN50")
saved_state_dict = torch.load("RN50_results(epoch100)/best.pt")
model.load_state_dict(saved_state_dict)
model.eval().float()

input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

# print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
# print("Input resolution:", input_resolution)
# print("Context length:", context_length)
# print("Vocab size:", vocab_size)

file_name = 'data/map_clsloc.txt'

objects = []
with open(file_name) as filename:
    for line in filename:
        obj = str(line.split()[2]).split('_')
        object = ''
        for o in obj:
            object += o + "" # _ 대신 ' '
            
        objects.append(object)
# print(objects)


imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]
# print(f"{len(objects)} classes, {len(imagenet_templates)} templates")




dataset1 = MyDataset("data/ILSVRC2012_img_val/Images/imagenet","data/ILSVRC2012_img_val/Labels/labels.txt",transform=preprocess)
#print(dataset1[0])

loader = torch.utils.data.DataLoader(dataset1, batch_size=32)

# dataiter1 = iter(loader)
# images1, labels1 = dataiter1.next()
# print(images1.size()) # 배치 및 이미지 크기 확인




def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #템플릿 80개에 대해서
            # print(texts,1111)
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            # print(class_embeddings,1111)   
            # print(type(class_embeddings))
                
            class_embedding = class_embeddings.mean(dim=0) #임베딩의 평균값을 구함 # 80x512 -> 1x512
            # print(class_embeddings.size(),2222)
            class_embedding /= class_embedding.norm() #norm값으로 나눔
            # print(class_embeddings.size(),3333) 
            zeroshot_weights.append(class_embedding)
            # print(zeroshot_weights[0].size(),22222) # [512,512,512,...,512]
      
        #print(zeroshot_weights,111111)
        # print(len(zeroshot_weights),2222)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        # print(zeroshot_weights.size(),333333333333)    
        
    return zeroshot_weights


zeroshot_weights = zeroshot_classifier(objects, imagenet_templates)
# print(zeroshot_weights.size(),123123) #512 x 1000 

# def accuracy(output, target, topk=(1,)):
#     pred = output.topk(max(topk), 1, True, True)[1].t()
#     # print(pred,1111111111)
#     # print(pred.size(),22222222222)
#     # print(target.view(1, -1).size(),99999999999999)
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#     return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]



with torch.no_grad():
    top1, top2, n = 0., 0., 0.
    corret, correct2 = 0.0, 0.0
    for i, (images, target) in enumerate(tqdm(loader)):
        # print(images, target,3333333333333)
    
        images = images.cuda()
        target = target.cuda()
        #print(target,123123)
        # print(target.size(),88888888888)
        # print(images.size())

        #target = F.one_hot(target, num_classes=1000)
        #print(target)
        #print(target.size())
       
        # predict
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # print(image_features.size())
        # print(zeroshot_weights.size())
        logits = 100. * image_features @ zeroshot_weights



        top_probs, top_labels = logits.topk(1, dim=-1)
        top_probs2, top_labels2 = logits.topk(2, dim=-1)
        # print(top_probs, top_labels,00000)
        # print(top_probs2, top_labels2,11111)

        pred = logits.topk(max((1,5)), 1, True, True)[1].t() #[[1~5],[1~5],...,[1~5]]

        # print(pred,1111111111)

        # print(target.view(1, -1),222222222222)
        
        # print(target.view(1, -1).expand_as(pred),3333333333333)
     

        correct = pred.eq(target.view(1, -1).expand_as(pred)) #True or False
        # print(correct,444444444444444444)

        acc1 = float(correct[:1].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        acc2 = float(correct[:2].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())

        # print(float(correct[:1].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()))
        # print(float(correct[:2].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()))

        # # measure accuracy

        top1 += acc1
        top2 += acc2
        n += images.size(0)

top1 = (top1 / n) * 100
top2 = (top2 / n) * 100 

print(f"Top-1 accuracy: {top1:.2f}")
print(f"Top-2 accuracy: {top2:.2f}")