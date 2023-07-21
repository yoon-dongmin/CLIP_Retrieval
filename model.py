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
from data_utils2 import CustomDataset2
import data_utils2 as data_util
import utils.helpers as utils
import matplotlib.pyplot as plt
import random
import os

# CUDA_VISIBLE_DEVICES=3 python model4.py 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.cuda.empty_cache()
random.seed(100)


FTensor = torch.cuda.FloatTensor

class EarlyStopping:
    """조기 종료 (Early stopping)를 위한 클래스"""
    def __init__(self, file_name, patience=5, verbose=False,  delta=0):
        """
        Args:
            patience (int): Improvement가 이루어지지 않은 epoch 수가 patience를 초과하면 학습을 멈춘다. Default: 7
            verbose (bool): True일 경우 각 단계에 대한 메시지를 출력. Default: False
            delta (float): Improvement가 delta보다 큰 값일 경우에만 Improvement로 인정. Default: 0
        """
        self.file_name = file_name
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0
        self.delta = delta

    def __call__(self, val_acc, model):

        score = val_acc
        
        ###가장 높은 acc가지는 모델 저장###
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        '''validation accuracy가 증가하면 모델을 저장한다'''
        name = self.file_name
        if self.verbose:
            print(f'Test Accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), name + '/best.pt')
        # torch.save({
        # 'epoch': epoch,
        # 'model_state_dict': clip_model.state_dict(),
        # }, f"models2/model" + str(epoch+1) +".pt")
        self.val_acc_max = val_acc

def eval(clip_model, testloader, eval_loss, ret_acc1, ret_acc2,num_comm=5,mode='val'):
    sum_loss, top1, top2, n = 0., 0., 0., 0.

    for idx, (sentence, ret_objs, val) in enumerate(tqdm(testloader)):
        # print(len(sentence))
        empty_batch = torch.zeros((num_comm,len(sentence)), dtype=torch.float16).cuda()
        ground_truth = torch.zeros((len(sentence)), dtype=torch.float16).cuda()
        # print(empty_batch.size())
        # print(ground_truth.size())
        with torch.no_grad():
            text_inputs = clip.tokenize(sentence, context_length=77, truncate=True).to('cuda',
                                                                                        non_blocking=True)                                               
            text_features = clip_model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # print(text_features.size(),1111) 
            sims = [] 
            for sub_idx, (obj_name, affordances) in enumerate(ret_objs):
                affordances = affordances.to('cuda').to(torch.float16)
                affordances = affordances / affordances.norm(dim=-1, keepdim=True) 
                # print(affordances.size(),2222)
                
                ####loss 측정###
                #positive:1,negative:1
                if sub_idx == 0:
                    val = val.cuda()
                    sum_loss += abs(F.cosine_embedding_loss(text_features,affordances,val).item())

                ###ACC 측정###
                sim = F.cosine_similarity(text_features, affordances)
                # print(sim.shape)
         
                empty_batch[sub_idx,:]= sim
              
                # sims = torch.stack(())
               
                # sims.append(sim.item())
            # print(empty_batch)
            # print(empty_batch.T)
            logits = empty_batch.T
            
            # top_probs, top_tests = logits.topk(1, dim=-1)
            # print(top_tests)

            ###ACC 측정###
            pred = logits.topk(max((1,2)), 1, True, True)[1].t()         
            # print(pred)
            correct = pred.eq(ground_truth.expand_as(pred))
            # print(correct)

            acc1 = float(correct[:1].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            acc2 = float(correct[:2].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            # print(acc1,acc2)    
        
            top1 += acc1
            top2 += acc2
            n += len(sentence)

    Eval_loss = sum_loss / len(testloader)
    top1 = (top1 / n) * 100
    top2 = (top2 / n) * 100 


    if mode == 'val':
        print(f"Eval loss: {Eval_loss:.2f}")
        print(f"Eval Top-1 accuracy: {top1:.2f}")
        print(f"Eval Top-2 accuracy: {top2:.2f}")
    else:
        print(f"Test loss: {Eval_loss:.2f}")
        print(f"Test Top-1 accuracy: {top1:.2f}")
        print(f"Test Top-2 accuracy: {top2:.2f}")

    eval_loss.append(Eval_loss)
    ret_acc1.append(top1)
    ret_acc2.append(top2)

    return top1

def clip_finetune(num_epochs: int, clip_model_name: str, learning_rate: float, batch_size: int
                        ,encoder: str, plot_fig: bool, file_name: str):
                       
    """
    Fine-tune CLIP on the ImageNet val dataset using affordance based commad
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param learning_rate: fine-tuning learning rate
    :param batch_size: batch size
    :param encoder: which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio`    :return:
    """

    ###Data loading###
    with open('data/RN50/label_embedding.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            label_embedding = json.loads(line) #v-o dictionary를 한줄씩 읽음

    with open('data/RN50/test_embedding.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            test_embedding = json.loads(line) #v-o dictionary를 한줄씩 읽음    

    with open('data/vo_dict.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            c_vo_dict = json.loads(line) #v-o dictionary를 한줄씩 읽음


    train_list = list(label_embedding.keys())
    test_list = list(test_embedding.keys())
    # print(len(train_list)) #512
    # print(len(test_list)) #82

    vo_dict = {}
    for verb,objs in c_vo_dict.items():
        for obj in objs:
            if obj in train_list:
                if verb not in vo_dict: #verb가 없으면 
                    vo_dict[verb] = []
                vo_dict[verb].append(obj)

    # print(len(vo_dict))
    # i = 0
    # for key in vo_dict.keys():
    #     i += len(vo_dict[key])
    # print(i) #1785
    # print(i*40)


    vo_dict2 = {}
    for verb,objs in c_vo_dict.items():
        for obj in objs:
            if obj in test_list:
                if verb not in vo_dict2: #verb가 없으면 
                    vo_dict2[verb] = []
                vo_dict2[verb].append(obj)

    # print(len(vo_dict2))
    # i = 0
    # for key in vo_dict2.keys():
    #     i += len(vo_dict2[key])
    # print(i) #1785
    # print(i*50)
    # exit()


    # print(len(test_embedding['dishwasher']))
  
    # print(len(train_list[0]))
    # print(train_list[0])
    # print(len(embedding_list[0]))

    ###test/train setting###
    
    train_embedding = {}
    val_embedding = {}

    for obj, embedding in label_embedding.items():
        train_embedding[obj] = embedding[:40]
        val_embedding[obj] = embedding[40:]


    train_obj = list(train_embedding.keys())
    # print(len(train_obj)) #512


    # print(len(train_embedding['dishwasher']))
    # print(len(val_embedding['dishwasher']))



    clip_model, preprocess = clip.load(clip_model_name, device='cuda', jit=False)
    # saved_state_dict = torch.load("template40.pt")
    # clip_model.load_state_dict(saved_state_dict)


    ###Train data###
    train_data = CustomDataset2(train_embedding,vo_dict,train_val_mode='train')
    val_data = CustomDataset2(val_embedding,vo_dict,train_val_mode='val')
    test_data = CustomDataset2(test_embedding,vo_dict2,train_val_mode='test')


    ###Data loading###
    ###text encoder 혹은 image encoder fine-tuning###
    if encoder == 'text':
        print('Only the CLIP text encoder will be fine-tuned')
        for param in clip_model.visual.parameters():
            param.requires_grad = False
    elif encoder == 'image':
        print('Only the CLIP image encoder will be fine-tuned')
        for param in clip_model.parameters():
            param.requires_grad = False
        for param in clip_model.visual.parameters():
            param.requires_grad = True
    elif encoder == 'both':
        print('Both CLIP encoders will be fine-tuned')
    else:
        raise ValueError("encoder parameter should be in ['text', 'image', both']")


    clip_model.eval().float()

    # Define the optimizer, the loss and the grad scaler

    optimizer = optim.AdamW(clip_model.parameters(), lr=2e-6,betas=(0.9, 0.999),eps=1e-7,weight_decay=0.1)
    crossentropy_criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=512)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=512)
    # print(len(trainloader)) #71400
    # print(len(valloader))#17850
    # print(len(testloader)) #13850

    plot_loss, plot_acc = [], []
    train_loss, val_loss, test_loss = [], [], []
    val_ret_acc1, val_ret_acc2, test_ret_acc1, test_ret_acc2 = [], [], [], []

    ###model evaluation###
    #validation
    _ = eval(clip_model, valloader, val_loss, val_ret_acc1, val_ret_acc2,mode='val')
    #test
    _ = eval(clip_model, testloader, test_loss, test_ret_acc1, test_ret_acc2,mode='test')


    ###model train###
    early_stopping = EarlyStopping(file_name, patience=5, verbose=True, delta=0)
    for epoch in range(num_epochs): #tqdm module을 이용하면 iteration 진행 상황을 progress bar로 나타낼 수 있다
        sum_loss = 0.0
        print('EPOCH', epoch + 1)
        for idx, (command, ret_embedds) in enumerate(tqdm(trainloader)):
            # print(command,obj)

            optimizer.zero_grad()
     
            text_inputs = clip.tokenize(command, context_length=77, truncate=True).to('cuda',
                                                                                        non_blocking=True)
                                                                               
            text_features = clip_model.encode_text(text_inputs)                                                            
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) 
            # print(text_features.size())
            # print(ret_embedds.size())
            # print(ret_embedds.to(torch.float32).size())
            # print(text_features.unsqueeze(1).size())

            """
            text : [b,1024] => [b,1,1024]
            image : [b,1024,100]
            [4,1,100] => [4,100]
            """
            logits = 100 * torch.bmm(text_features.unsqueeze(1), ret_embedds.to(torch.float32).to('cuda')).squeeze(1)
            logits = logits.to('cuda').to(torch.float16)
            ground_truth = torch.zeros(len(command), dtype=torch.long).cuda() #0번째 index가 정답
            # print(logits.size())
            # print(ground_truth.size())

            loss = crossentropy_criterion(logits, ground_truth)
            sum_loss += loss.item()
            
            # Backpropagate and update the weights
            loss.backward()
            optimizer.step()
        # scheduler.step()
        print(f"Train loss: {sum_loss / len(trainloader):.2f}")
        train_loss.append(sum_loss / len(trainloader))
        print()

        #validation
        _ = eval(clip_model, valloader, val_loss, val_ret_acc1, val_ret_acc2,mode='val')
        #test
        top1_acc = eval(clip_model, testloader, test_loss, test_ret_acc1, test_ret_acc2,mode='test')
        print()
        early_stopping(top1_acc, clip_model)

        if early_stopping.early_stop:
            print("Early stopping")
            break


    # torch.save(clip_model.state_dict(), 'RN50_model2/last_checkpoint.pt')


    with open(file_name +'/val_ret_acc1.json', 'w+') as f: 
        f.write(json.dumps(val_ret_acc1)) #json 문자열로 변환

    with open(file_name +'/test_ret_acc1.json', 'w+') as f: 
        f.write(json.dumps(test_ret_acc1)) #json 문자열로 변환

    with open(file_name +'/train_loss.json', 'w+') as f: 
        f.write(json.dumps(train_loss)) #json 문자열로 변환

    with open(file_name +'/val_loss.json', 'w+') as f: 
        f.write(json.dumps(val_loss)) #json 문자열로 변환

    with open(file_name +'/test_loss.json', 'w+') as f: 
        f.write(json.dumps(test_loss)) #json 문자열로 변환


    if plot_fig:
        plot_loss.append((train_loss,test_loss))
        plot_acc.append((test_ret_acc1, test_ret_acc2))
        
        utils.loss_plot(plot_loss,'Train & Eval Loss',
                file_name + '/loss.png')

        utils.acc_plot(plot_acc, 'Top1 & Top2 Accuracy',
                file_name + '/accuracy.png')



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--num_epochs", default=50, type=int, help="number training epochs")
    parser.add_argument("--clip_model_name", default="RN50", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--encoder", default='text', type=str,
                    help="Which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']")
    parser.add_argument("--learning_rate", default=1e-6, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--plot_fig", type=bool, default=True, help='')
    parser.add_argument("--file_name", type=str, default='RN50_neg60', help='')
    args = parser.parse_args()

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "clip_model_name": args.clip_model_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "encoder": args.encoder,
        "plot_fig": args.plot_fig,
        "file_name": args.file_name
    }


    clip_finetune(**training_hyper_params)






