from argparse import ArgumentParser
from datetime import datetime
import json
import clip
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from data_utils import CustomDataset #verb-object
import data_utils as data_util
import utils.helpers as utils
import matplotlib.pyplot as plt
import random
import os

# CUDA_VISIBLE_DEVICES=3 python model4.py 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
random.seed(100)


FTensor = torch.cuda.FloatTensor


def vo_eval(clip_model,testloader):
    top1, top2, top3, top4, top5, n = 0., 0., 0., 0., 0., 0.
    for idx, (test_embedding, test_command) in enumerate(tqdm(testloader)):
        with torch.no_grad():   
            image_features = test_embedding.to('cuda').to(torch.float32)
            # print(image_features.size(0)) #80
           
            text_inputs = clip.tokenize(test_command, context_length=77, truncate=True).to('cuda',
                                                                                        non_blocking=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)                                                                       
            text_features = clip_model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) 
            # print(image_features.dtype)
            # print(text_features.dtype)
      
            logits = 100 * image_features @ text_features.T          
            # top_probs, top_labels = logits.topk(1, dim=-1)
            # top_probs2, top_labels2 = logits.topk(2, dim=-1)
            # print(top_probs, top_labels,00000)
            # print(top_probs2, top_labels2,11111)
      
            ground_truth = torch.arange(image_features.size(0), dtype=torch.long, device='cuda')      
   
            ###ACC 측정###
            pred = logits.topk(max((1,5)), 1, True, True)[1].t() #[[1~5],[1~5],...,[1~5]]
            # print(pred,123123)
            
            correct = pred.eq(ground_truth.expand_as(pred)) #True or False
            # print(correct)
            
            acc1 = float(correct[:1].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            acc2 = float(correct[:2].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            acc3 = float(correct[:3].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            acc4 = float(correct[:4].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            acc5 = float(correct[:5].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            # print(acc1,acc2,acc3,acc4,acc5)

            top1 += acc1
            top2 += acc2
            top3 += acc3
            top4 += acc4
            top5 += acc5

            n += image_features.size(0) #80

    # print(n)
    top1 = (top1 / n) * 100
    top2 = (top2 / n) * 100 
    top3 = (top3 / n) * 100 
    top4 = (top4 / n) * 100 
    top5 = (top5 / n) * 100 

    
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-2 accuracy: {top2:.2f}")
    print(f"Top-3 accuracy: {top3:.2f}")
    print(f"Top-4 accuracy: {top4:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")


def object_eval(clip_model,testloader):
    top1, top2, top3, top4, top5, n = 0., 0., 0., 0., 0., 0.
    for idx, (test_embedding, test_command) in enumerate(tqdm(testloader)):
        with torch.no_grad():   
            image_features = test_embedding.to('cuda').to(torch.float32)
            # print(image_features.size(0)) #80
           
            text_inputs = clip.tokenize(test_command, context_length=77, truncate=True).to('cuda',
                                                                                        non_blocking=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)        
            # image_features = image_features.to('cuda').to(torch.float16) #보류                                                              
            text_features = clip_model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) 
            # print(image_features.dtype)
            # print(text_features.dtype)
      
            logits = 100 * image_features @ text_features.T          
            # top_probs, top_labels = logits.topk(1, dim=-1)
            # top_probs2, top_labels2 = logits.topk(2, dim=-1)
            # print(top_probs, top_labels,00000)
            # print(top_probs2, top_labels2,11111)
      
            ground_truth = torch.arange(image_features.size(0), dtype=torch.long, device='cuda')      
   
            ###ACC 측정###
            pred = logits.topk(max((1,5)), 1, True, True)[1].t() #[[1~5],[1~5],...,[1~5]]
            # print(pred,123123)
            
            correct = pred.eq(ground_truth.expand_as(pred)) #True or False
            # print(correct)
            
            acc1 = float(correct[:1].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            acc2 = float(correct[:2].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            acc3 = float(correct[:3].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            acc4 = float(correct[:4].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            acc5 = float(correct[:5].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            # print(acc1,acc2,acc3,acc4,acc5)

            top1 += acc1
            top2 += acc2
            top3 += acc3
            top4 += acc4
            top5 += acc5

            n += image_features.size(0) #80

    # print(n)
    top1 = (top1 / n) * 100
    top2 = (top2 / n) * 100 
    top3 = (top3 / n) * 100 
    top4 = (top4 / n) * 100 
    top5 = (top5 / n) * 100 

    
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-2 accuracy: {top2:.2f}")
    print(f"Top-3 accuracy: {top3:.2f}")
    print(f"Top-4 accuracy: {top4:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")



def test(clip_model_name: str, batch_size: int):
                       
    """
    param clip_model_name: CLIP model you want to use: "RN101", "RN101", "RN101x4"...
    param batch_size: batch size
    """

    ###Data loading###
    with open('data/RN50/label_embedding.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            label_embedding = json.loads(line) #v-o dictionary를 한줄씩 읽음

    with open('data/RN50/test_embedding.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            test_embedding = json.loads(line) #v-o dictionary를 한줄씩 읽음

    with open('data/ov_dict.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            ov_pair = json.loads(line) #v-o dictionary를 한줄씩 읽음

    with open('data/vo_dict.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            vo_dict = json.loads(line) #v-o dictionary를 한줄씩 읽음

    # print(len(label_embedding))
    # print(len(test_embedding))

    label_list = list(label_embedding.keys())
    test_list = list(test_embedding.keys())
    train_embedding_list = list(label_embedding.values())
    test_embedding_list = list(test_embedding.values())

    label_embedding.update(test_embedding)
    # print(len(label_embedding))


    label_list = label_list + test_list
    embedding_list = train_embedding_list + test_embedding_list    
    # print(label_list)
    # print(len(label_list))
 
    # for verb,objs in c_vo_dict.items():
    #     for obj in objs:
    #         if obj in label_list:
    #             if verb not in vo_dict: #verb가 없으면 
    #                 vo_dict[verb] = []
    #             vo_dict[verb].append(obj)

    # print(len(vo_dict))

    clip_model, preprocess = clip.load("RN50")
    saved_state_dict = torch.load("RN50_neg60(3)/best.pt")
    clip_model.load_state_dict(saved_state_dict)

    clip_model.eval().float()


    ###verb-object accuracy###
    test_data = CustomDataset(label_list, label_embedding, ov_pair, vo_dict, train_val_mode='test', o_ov_mode = 'o')
    testloader = torch.utils.data.DataLoader(test_data, batch_size=594)
    # print(len(testloader)) #29700



    ###model evaluation###
    ###1.verb###
    object_eval(clip_model, testloader)

    ###2.verb-object###
    # vo_eval(clip_model,testloader)






if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clip-model-name", default="RN50", type=str, help="CLIP model to use, e.g 'RN101', 'RN101x4'")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    args = parser.parse_args()

    training_hyper_params = {
        "clip_model_name": args.clip_model_name,
        "batch_size": args.batch_size,
    }



    test(**training_hyper_params)






