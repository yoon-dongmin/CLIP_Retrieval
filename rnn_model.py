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
from data_utils3 import CustomDataset3
import data_utils3 as data_util
import utils.helpers as utils
import matplotlib.pyplot as plt
import random
import os
import rnn

# CUDA_VISIBLE_DEVICES=3 python model4.py 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
random.seed(100)

Tensor = torch.cuda.LongTensor
FTensor = torch.cuda.FloatTensor


def process_command(command, word2id):
    sentence = []
    s = command.lower().split()
    for word in s:
        if word in word2id:
            sentence.append(word2id[word])
        else:
            sentence.append(word2id['UNK'])
    return sentence

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
            print(f'Validation Accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), name + '/best.pt')
        # torch.save({
        # 'epoch': epoch,
        # 'model_state_dict': clip_model.state_dict(),
        # }, f"models2/model" + str(epoch+1) +".pt")
        self.val_acc_max = val_acc

def eval(model, testloader, eval_loss, ret_acc1, ret_acc2,num_comm=5,mode='val'):
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

    
    # print(f"Top-1 accuracy: {top1:.2f}")
    # print(f"Top-2 accuracy: {top2:.2f}")
    # print(f"Top-3 accuracy: {top3:.2f}")
    # print(f"Top-4 accuracy: {top4:.2f}")
    # print(f"Top-5 accuracy: {top5:.2f}")

    
    # top1 = (top1 / n) * 100
    # top2 = (top2 / n) * 100 

    # print(f"Eval loss: {Eval_loss:.2f}")
    # print(f"Top-1 accuracy: {top1:.2f}")
    # print(f"Top-2 accuracy: {top2:.2f}")


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
 
def train_rnn(num_epochs: int, learning_rate: float, batch_size: int
                        ,save_best: bool, plot_fig: bool, file_name: str):
                       

    ###Data loading###
    with open('data/Resnet101/label_embedding5.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            label_embedding = json.loads(line) #v-o dictionary를 한줄씩 읽음

    with open('data/Resnet101/test_embedding5.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            test_embedding = json.loads(line) #v-o dictionary를 한줄씩 읽음

    with open('data/vo_dict.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            c_vo_dict = json.loads(line) #v-o dictionary를 한줄씩 읽음


    with open("data/Resnet101/word2id.json") as f:
        for line in f:
            word2id = json.loads(line)


    ###make tokenizer###
    """
    with open('data/Resnet101/train_command.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            train_command = json.loads(line) #v-o dictionary를 한줄씩 읽음

    with open('data/Resnet101/test_command.json') as f: #vo_dict.json is generated by gen_data.py
        for line in f:
            test_command = json.loads(line) #v-o dictionary를 한줄씩 읽음


   
    word2id = {'UNK':0}
    id2word = {0:'UNK'}

    for row in train_command:
        s = row.lower().split()
        for word in s:
            if word not in word2id: # build word vocab
                word2id[word] = len(word2id) #
                id2word[word2id[word]] = word


    for row in test_command:
        s = row.lower().split()
        for word in s:
            if word not in word2id: # build word vocab
                word2id[word] = len(word2id) #
                id2word[word2id[word]] = word

    with open('data/Resnet101/word2id.json', 'w+') as f: # w+ => 파일이 없으면 새로 만든다. word2id.json 이름으로 저장
        f.write(json.dumps(word2id))
    with open('data/Resnet101/id2word.json', 'w+') as f: # id2word.json 이름으로 저장
        f.write(json.dumps(id2word))

"""



    label_list = list(label_embedding.keys())
    test_list = list(test_embedding.keys())
    # print(len(label_list))

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
    # print(i) #1767


    vo_dict2 = {}
    for verb,objs in c_vo_dict.items():
        for obj in objs:
            if obj in test_list:
                if verb not in vo_dict2: #verb가 없으면 
                    vo_dict2[verb] = []
                vo_dict2[verb].append(obj)





    # print(len(label_embedding['dishwasher']))

    # print(len(label_list[0]))
    # print(label_list[0])
   

    ###test/train setting###
    
    train_embedding = {}
    val_embedding = {}

    for obj, embedding in label_embedding.items():
        train_embedding[obj] = embedding[:40]
        val_embedding[obj] = embedding[40:]

    # print(len(train_embedding['dishwasher']))
    # print(len(val_embedding['dishwasher']))
    


    ###Train data###
    train_data = CustomDataset3(train_embedding,vo_dict,word2id, train_val_mode='train')
    val_data = CustomDataset3(val_embedding,vo_dict,word2id, train_val_mode='val')
    test_data = CustomDataset3(test_embedding,vo_dict2,word2id, train_val_mode='test')



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

    # Define the optimizer, the loss and the grad scaler
    # optimizer = optim.Adam(clip_model.parameters(), lr=1e-6,betas=(0.9,0.98),eps=1e-6,weight_decay=0.1)
    # optimizer = optim.AdamW(model.parameters(), lr=2e-6,betas=(0.9, 0.999),eps=1e-7,weight_decay=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # scaler = torch.cuda.amp.GradScaler()
   
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=1)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=1)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1)

    # print(len(trainloader)) #142800
    # print(len(valloader)) #17850
    # print(len(testloader)) #13850
  
    ###Data check###
    # #train_data
    # dataiter = iter(trainloader)
    # image_embedding, command, obj, verb = next(dataiter) #images, labels = dataiter.next()

    # print(image_embedding, command, obj, verb)

    # #test_data
    # dataiter = iter(testloader)
    # image_embedding, command = next(dataiter) #images, labels = dataiter.next()
    # print(image_embedding, command)


    plot_loss, plot_acc = [], []
    train_loss, val_loss, test_loss = [], [], []
    val_ret_acc1, val_ret_acc2, test_ret_acc1, test_ret_acc2 = [], [], [], []

    ###model evaluation###
    _ = eval(model, valloader, val_loss, val_ret_acc1, val_ret_acc2,mode='val')
    _ = eval(model, testloader, test_loss, test_ret_acc1, test_ret_acc2,mode='test')
 

    ###model train###
    early_stopping = EarlyStopping(file_name,patience=10, verbose=True)
    for epoch in range(num_epochs): #tqdm module을 이용하면 iteration 진행 상황을 progress bar로 나타낼 수 있다
        model.train()
        sum_loss = 0.0
        print('EPOCH', epoch + 1)
        for idx, (command, image_embedding, val) in enumerate(tqdm(trainloader)):
            # print(image_embedding)ll
            # print(command)
            # print(val)
            optimizer.zero_grad()
            image_features = image_embedding.to('cuda').to(torch.float16).view(1,2048)   
            # print(command)
            # print(type(command))
     
            text_features = model(command)       
            loss = F.cosine_embedding_loss(image_features, text_features, val.cuda())
            sum_loss += abs(loss.item())

            # Backpropagate and update the weights
            loss.backward()
            optimizer.step()
        print(f"Train loss: {sum_loss / len(trainloader):.2f}")
        train_loss.append(sum_loss / len(trainloader))
        print()
        #validation
        _ = eval(model, valloader, val_loss, val_ret_acc1, val_ret_acc2,mode='val')
        #test
        top1_acc = eval(model, testloader, test_loss, test_ret_acc1, test_ret_acc2,mode='test')
        print()
        early_stopping(top1_acc, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open(file_name +'/val_ret_acc1.json', 'w+') as f: 
        f.write(json.dumps(val_ret_acc1)) #json 문자열로 변환

    with open(file_name +'/test_ret_acc1.json', 'w+') as f: 
        f.write(json.dumps(test_ret_acc1)) #json 문자열로 변환

    with open(file_name +'/train_loss.json', 'w+') as f: 
        f.write(json.dumps(train_loss)) #json 문자열로 변환




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
    parser.add_argument("--learning_rate", default=1e-6, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--save_best", dest="save_best", action='store_true',
                        help="Save only the best model during training")
    parser.add_argument("--plot_fig", type=bool, default=True, help='')
    parser.add_argument("--file_name", type=str, default='RNN_result5', help='')
    args = parser.parse_args()

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "save_best": args.save_best,
        "plot_fig": args.plot_fig,
        "file_name": args.file_name,
    }

    train_rnn(**training_hyper_params)

