from argparse import ArgumentParser
from datetime import datetime
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from data_utils3 import CustomDataset3
import data_utils3 as data_util
import rnn
# from data_utils import CustomDataset #verb-object
# import data_utils as data_util
import utils.helpers as utils
import matplotlib.pyplot as plt
import random
import os

# CUDA_VISIBLE_DEVICES=3 python model4.py 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
random.seed(100)


FTensor = torch.cuda.FloatTensor


def verb_eval(model, testloader,num_comm=5):
    model.eval()
    sum_loss, top1, top2,top3, top4, top5, n = 0., 0., 0., 0., 0., 0., 0.
    for idx, (command, ret_objs) in enumerate(tqdm(testloader)):
        # print(command.shape)

        empty_batch = torch.zeros((num_comm,1), dtype=torch.float16).cuda()
        ground_truth = torch.zeros((1), dtype=torch.float16).cuda()
        # print(empty_batch.size())
        # print(ground_truth.size())
        with torch.no_grad():
            text_features = model(command)
            # text_features = text_features.to('cuda').to(torch.float16)
            # print(text_features.shape,11111)
            sims = [] 
            for sub_idx, (obj_name, affordances) in enumerate(ret_objs):
                affordances = affordances.to('cuda').to(torch.float16).view(1, 2048)
                # print(affordances.shape)
                # affordances = affordances / affordances.norm(dim=-1, keepdim=True) 
                # print(affordances.size(),2222)
                ####loss 측정###
                #positive:1,negative:1
                # if sub_idx == 0:
                #     val = val.cuda()
                #     sum_loss += abs(F.cosine_embedding_loss(text_features,affordances,val).item())

                ###ACC 측정###
                sim = F.cosine_similarity(text_features, affordances)
                # print(affordances.shape)
                # print(text_features.shape)
                # print(sim.shape)
                # print(sim)
                
                empty_batch[sub_idx,:]= sim
            
                # sims = torch.stack(())
                # sims.append(sim.item())

            # print(empty_batch)
            # print(empty_batch.T)
            logits = empty_batch.T
            
            # top_probs, top_labels = logits.topk(1, dim=-1)
            # print(top_labels)

            ###ACC 측정###
            pred = logits.topk(max((1,5)), 1, True, True)[1].t()         
            # print(pred)
            correct = pred.eq(ground_truth.expand_as(pred))
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

            # acc1 = float(correct[:1].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            # acc2 = float(correct[:2].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
 
        
            # top1 += acc1
            # top2 += acc2
            n += 1
    
    Eval_loss = sum_loss / len(testloader)
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


def test(clip_model_name: str, batch_size: int):
                       
    """
    param clip_model_name: CLIP model you want to use: "RN101", "RN101", "RN101x4"...
    param batch_size: batch size
    """

    ###Data loading###
    with open('data/Resnet101/test_embedding5.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            label_embedding = json.loads(line) #v-o dictionary를 한줄씩 읽음

    with open('data/ov_dict.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            ov_pair = json.loads(line) #v-o dictionary를 한줄씩 읽음

    with open('data/vo_dict.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            c_vo_dict = json.loads(line) #v-o dictionary를 한줄씩 읽음

    with open("data/Resnet101/word2id.json") as f:
        for line in f:
            word2id = json.loads(line)

    label_list = list(label_embedding.keys())
    embedding_list = list(label_embedding.values())
    # print(label_list)
    # print(len(label_list))
 
    vo_dict = {}
    for verb,objs in c_vo_dict.items():
        for obj in objs:
            if obj in label_list:
                if verb not in vo_dict: #verb가 없으면 
                    vo_dict[verb] = []
                vo_dict[verb].append(obj)



    ###load model###
    vocab_size = len(word2id)
    rnn_input = 128
    rnn_output = 2048
    hidden_dim = 64
    num_layers = 1
    dropout = 0.0
    device = 'cuda'


    model = nn.Sequential(
    nn.Embedding(vocab_size, rnn_input),
    rnn.RNNModel(rnn_input, rnn_output, hidden_dim, num_layers,
                 dropout, device)).to(device)


    saved_state_dict = torch.load("RNN_result5/checkpoint.pt")
    model.load_state_dict(saved_state_dict)


    # ###verb-object accuracy###
    # test_data = CustomDataset(label_list,embedding_list,ov_pair,vo_dict)
    # testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    ###verb accuracy###
    test_data = CustomDataset3(label_embedding,vo_dict,word2id,train_val_mode='test')
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1)
    
    ###verb-object###
    # print(len(testloader)) #4000

    ###verb###
    # print(len(testloader)) #14000


    ###model evaluation###
    ###1.verb###
    verb_eval(model, testloader)

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






