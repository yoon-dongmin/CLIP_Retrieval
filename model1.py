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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
random.seed(100)


FTensor = torch.cuda.FloatTensor

class EarlyStopping:
    """조기 종료 (Early stopping)를 위한 클래스"""
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): Improvement가 이루어지지 않은 epoch 수가 patience를 초과하면 학습을 멈춘다. Default: 7
            verbose (bool): True일 경우 각 단계에 대한 메시지를 출력. Default: False
            delta (float): Improvement가 delta보다 큰 값일 경우에만 Improvement로 인정. Default: 0
        """
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
        if self.verbose:
            print(f'Validation Accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'RN50_model/best.pt')
        # torch.save({
        # 'epoch': epoch,
        # 'model_state_dict': clip_model.state_dict(),
        # }, f"models2/model" + str(epoch+1) +".pt")
        self.val_acc_max = val_acc



def eval(clip_model, testloader, eval_loss, ret_acc1, ret_acc2,bsz,num_comm=5):
    sum_loss, top1, top2, n = 0., 0., 0., 0.
    bsz = 512
    for idx, (sentence, ret_objs, val) in enumerate(tqdm(testloader)):
        # print(len(sentence))
        empty_batch = torch.zeros((num_comm,bsz), dtype=torch.float16).cuda()
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
                empty_batch[sub_idx,:]= sim
              
                
                # sims = torch.stack(())
               
                # sims.append(sim.item())
            # print(empty_batch)
            # print(empty_batch.T)
            logits = empty_batch.T
            
            # top_probs, top_labels = logits.topk(1, dim=-1)
            # print(top_labels)

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
            n += bsz
    

    Eval_loss = sum_loss / len(testloader)
    top1 = (top1 / n) * 100
    top2 = (top2 / n) * 100 

    print(f"Eval loss: {Eval_loss:.2f}")
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-2 accuracy: {top2:.2f}")


    eval_loss.append(Eval_loss)
    ret_acc1.append(top1)
    ret_acc2.append(top2)

    return top1

def clip_finetune(num_epochs: int, clip_model_name: str, learning_rate: float, batch_size: int
                        ,encoder: str, save_best: bool, plot_fig: bool):
                       
    """
    Fine-tune CLIP on the ImageNet val dataset using affordance based commad
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param learning_rate: fine-tuning learning rate
    :param batch_size: batch size
    :param encoder: which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']
    :param save_best: when True save only the weights of the best Combiner wrt three different averages of the metrics
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio`    :return:
    """

    ###Data loading###
    with open('data/RN50/train1_embedding.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            label_embedding = json.loads(line) #v-o dictionary를 한줄씩 읽음

    with open('data/vo_dict.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            c_vo_dict = json.loads(line) #v-o dictionary를 한줄씩 읽음

    label_list = list(label_embedding.keys())
    # print(label_list)
    
    vo_dict = {}
    for verb,objs in c_vo_dict.items():
        for obj in objs:
            if obj in label_list:
                if verb not in vo_dict: #verb가 없으면 
                    vo_dict[verb] = []
                vo_dict[verb].append(obj)

    # print(vo_dict)
    # print(len(vo_dict))
    # i = 0
    # for key in vo_dict.keys():
    #     i += len(vo_dict[key])
    # print(i) #1788



    # print(len(label_embedding['dishwasher']))
  
    # print(len(label_list[0]))
    # print(label_list[0])
    # print(len(embedding_list[0]))

    ###test/train setting###
    
    train_embedding = {}
    test_embedding = {}

    for obj, embedding in label_embedding.items():
        train_embedding[obj] = embedding[:40]
        test_embedding[obj] = embedding[40:]

    # print(len(train_embedding['dishwasher']))
    # print(len(test_embedding['dishwasher']))



    clip_model, preprocess = clip.load(clip_model_name, device='cuda', jit=False)
    # saved_state_dict = torch.load("template40.pt")
    # clip_model.load_state_dict(saved_state_dict)


    ###Train data###
    train_data = CustomDataset2(train_embedding,vo_dict,train_val_mode='train')
    test_data = CustomDataset2(test_embedding,vo_dict,train_val_mode='val')



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
    # optimizer = optim.Adam(clip_model.parameters(), lr=1e-6,betas=(0.9,0.98),eps=1e-6,weight_decay=0.1)
    optimizer = optim.AdamW(clip_model.parameters(), lr=2e-6,betas=(0.9, 0.999),eps=1e-7,weight_decay=0.1)
    # optimizer = optim.Adam(clip_model.parameters(), lr=0.0001)
    # scaler = torch.cuda.amp.GradScaler()
   
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    crossentropy_criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=512)
    # print(len(trainloader)) #143040
    # print(len(testloader)) #17880


    plot_loss, plot_acc = [], []
    train_loss, eval_loss,\
    ret_acc1, ret_acc2 = [], [], [], []

    ###model evaluation###
    _ = eval(clip_model, testloader, eval_loss, ret_acc1, ret_acc2, bsz=512)


    ###model train###
    early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in range(num_epochs): #tqdm module을 이용하면 iteration 진행 상황을 progress bar로 나타낼 수 있다
        sum_loss = 0.0
        print('EPOCH', epoch + 1)
        for idx, (command, ret_embedds) in enumerate(tqdm(trainloader)):
            optimizer.zero_grad()
            text_inputs = clip.tokenize(command, context_length=77, truncate=True).to('cuda',
                                                                                        non_blocking=True)
                                                                               
            text_features = clip_model.encode_text(text_inputs)                                                            
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) 
            # print(text_features.size())
            # print(ret_embedds.size())


            """
            text : [b,1024] => [b,1,1024]
            image : [b,1024,100]
            [4,1,100] => [4,100]
            """
            logits = 100 * torch.bmm(text_features.unsqueeze(1), ret_embedds).squeeze(1)
            ground_truth = torch.zeros(command.size(0), dtype=torch.long).cuda() #0번째 index가 정답
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
        top1_acc = eval(clip_model,testloader, eval_loss, ret_acc1, ret_acc2, bsz=512)
        print()
        early_stopping(top1_acc, clip_model)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    torch.save(clip_model.state_dict(), 'RN50_model/last_checkpoint.pt')


    with open('RN50_results/ret_acc1.json', 'w+') as f: 
        f.write(json.dumps(ret_acc1)) #json 문자열로 변환

    with open('RN50_results/train_loss.json', 'w+') as f: 
        f.write(json.dumps(train_loss)) #json 문자열로 변환

    with open('RN50_results/eval_loss.json', 'w+') as f: 
        f.write(json.dumps(eval_loss)) #json 문자열로 변환



    if plot_fig:
        plot_loss.append((train_loss,eval_loss))
        plot_acc.append((ret_acc1, ret_acc2))
        
        utils.loss_plot(plot_loss,'Train & Eval Loss',
                'RN50_results/loss.png')

        utils.acc_plot(plot_acc, 'Top1 & Top2 Accuracy',
                'RN50_results/accuracy.png')




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--num-epochs", default=50, type=int, help="number training epochs")
    parser.add_argument("--clip-model-name", default="RN50", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--encoder", default='text', type=str,
                    help="Which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']")
    parser.add_argument("--learning-rate", default=1e-6, type=float, help="Learning rate")
    parser.add_argument("--batch-size", default=128, type=int, help="Batch size")
    parser.add_argument("--validation-frequency", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")
    parser.add_argument("--plot_fig", type=bool, default=True, help='')
    args = parser.parse_args()

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "clip_model_name": args.clip_model_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "encoder": args.encoder,
        "save_best": args.save_best,
        "plot_fig": args.plot_fig
    }



    clip_finetune(**training_hyper_params)






