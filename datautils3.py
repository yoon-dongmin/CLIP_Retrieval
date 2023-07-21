###verb###
import json
from pathlib import Path
from typing import List
import random
import csv
import os
import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import torch
import utils.helpers as utils


random.seed(100)
FTensor = torch.cuda.FloatTensor
Tensor = torch.cuda.LongTensor


###v-o pair를 이용하여 데이터 생성###
###verb로 구성된 template###
class CustomDataset3(Dataset):
    def __init__(self, label_embedding, vo_dict, word2id, train_val_mode='train'):
        self.train_val_mode = train_val_mode 

        vo_templates = ['An item that can {}.', 'An object that can {}.',
           'Give me something that can {}.', 'Give me an item that can {}.',
           'Hand me something with which I can {}.',
           'Give me something with which I can {}.',
           'Hand me something to {}.', 'Give me something to {}.',
           'I want something to {}.', 'I need something to {}.',
           'Give me something so I can {}.',
            'Handover something to {}.', #수정됨
            'I must use something to {}.',
            'I have to use something to {}.',
            'Bring me something so I can {}',
            'Can you give me something to {}?',
            'Can you give me something so I can {}?',
            'Can you get me something to {}?',
            'Can you get me something so I can {}?',
            'Can you hand me something to {}?',
            'Can you bring me something to {}?',
            'Could you pass me something so I can {}?',
            'Could you bring me something so I can {}?',
            'Would you mind giving me something to {}?',
            'Do you have something I can {}?',
            'Without something, I am unable to {}.',
            'Please get something for me so I can {}.',
            'Please give me something so I can {}.',
            'Please bring me something so I can {}.',
            'Please hand me something so I can {}.',
            'Please get something so I can {}.',
            'Get me something so I can {}.',
            'Get something for me so I can {}.',
            'Would you use something to {}.',
            'Please use something to {}.',
            'Show me something I can {}.', #Object Detection
            'Show me something so I can {}.',
            'Can you show me something to {}?',
            'Could you show me something I can {}?',
            'Please show me something so I can {}.',
           ]
       
        objs = []
        verbs = []
        all_texts = []
        images = []
        for verb, obj in vo_dict.items():
            for o in obj: #random.choice(aff_dict[o])
                for i in range(len(list(label_embedding.values())[0])): #train:40 test:10
                    template = random.choice(vo_templates)
                    texts = template.format(verb)
                    texts = self.process_command(texts, word2id)
                    texts = Tensor(texts)
                    image = torch.tensor(label_embedding[o][i],dtype=torch.float)
                    objs.append(o)
                    verbs.append(verb)
                    all_texts.append(texts)
                    images.append(image)
 


        if train_val_mode == 'train':
            ###1 positve & 1 negative###
            self.train_data = utils.gen_examples(verbs,objs,all_texts,images,vo_dict)
            self.len = len(self.train_data)
            # print(self.train_data[:3])
            # print(self.len)

            ###1 positve & n negatives###
            # self.train_data = utils.gen_neg_samples(verbs,objs,all_texts,images,vo_dict)
            # self.len = len(self.train_data)
            # print(self.len)
       
        elif train_val_mode == 'val':
            self.val_data = utils.gen_examples2(verbs,objs,all_texts,images,vo_dict)
            self.len = len(self.val_data)

        else:
            self.test_data = utils.gen_examples2(verbs,objs,all_texts,images,vo_dict)
            self.len = len(self.test_data)
                


                
        # print(len(self.train_data)) #71400
        # print(len(self.val_data))  #17850
        # print(len(self.test_data)) #14000


 
    def __getitem__(self, index):
        if self.train_val_mode == 'train':

            ###1 negative###
            verb, obj, command, image_embedding, val = self.train_data[index]
            return command, image_embedding, val

            ###n negative###
            # sentence, ret_objs = self.train_data[index]
            
            # return sentence, ret_objs

        elif self.train_val_mode == 'val':
            sentence, ret_objs, _  = self.val_data[index]

            return sentence, ret_objs   # 3.3과 다르게 넘파이 배열로 출력 되는 것에 유의 하도록 한다.
    
        else:
            sentence, ret_objs, _  = self.test_data[index]
            
            return sentence, ret_objs


    def __len__(self):
        return self.len


    def process_command(self, command, word2id):
        sentence = []
        s = command.lower().split()
        for word in s:
            if word in word2id:
                sentence.append(word2id[word])
            else:
                sentence.append(word2id['UNK'])
        return sentence

