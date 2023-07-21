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



###v-o pair를 이용하여 데이터 생성###
###verb로 구성된 template###
class CustomDataset2(Dataset):
    def __init__(self, label_embedding, vo_dict, train_val_mode='train', template_mode='v'):
        self.train_val_mode = train_val_mode 
        if template_mode == 'ov':
            templates = [
            'Give me the {} to {}.',
            'Hand me the {} to {}.',
            'Pass me the {} to {}.',
            'Fetch the {} to {}.',
            'Get the {} to {}.',
            'Bring the {} to {}.',
            'Bring me the {} to {}.',
            'I need the {} to {}.',
            'I want the {} to {}.',
            'I need a {} to {}.',
            'I want a {} to {}.',
            'I need a {} so I can {}.', #여기서부터 추가
            'Give me the {} so I can {}.',
            'I must use the {} to {}.',
            'I have to use the {} to {}.',
            'Bring me a {} so I can {}.',
            'Grab a {} for me so I can {}.',
            'Can you give me the {} to {}?',
            'Can you get me a {} so I can {}?',
            'Can you hand me the {} to {}?',
            'Can you bring me the {} to {}?',
            'May I have the {} to {}?',
            'Could you pass me the {} so I can {}?',
            'Could you bring me the {} so I can {}?',
            'Could you get me the {} so I can {}?',
            'Would you mind giving me the {} to {}?',
            'Do you have a {} I can {}?',
            'The {} is necessary for me to be able to {}.',
            'Please get a {} for me so I can {}.',
            'Please give me a {} so I can {}.',
            'Please bring me the {} so I can {}.',
            'Please hand me the {} so I can {}.',
            'Please fetch me the {} so I can {}.',
            'Please get a {} so I can {}.',
            'Get me a {} so I can {}.',
            'Get the {} for me so I can {}.',
            'Show me a {} so I can {}.', #Object Detection
            'Show me the {} with which I can {}.',
            'Can you show me the {} to {}?',
            'Could you show me the {} I can {}?',
            ]
        
        elif template_mode == 'o':
            #40개 #아래2개 추가
            templates = [
            'Give me the {}.',
            'Hand me the {}.',
            'Pass me the {}.',
            'Fetch the {}.',
            'Get the {}.',
            'Bring the {}.',
            'Bring me the {}.',
            'I need the {}',
            'I want the {}.',
            'I need a {}.',
            'I want a {}.',
            'I must use the {}.',
            'I have to use the {}.',
            'Bring me a {}.',
            'Grab a {} for me.',
            'Can you give me the {}?',
            'Can you get me a {}?',
            'Can you hand me the {}?',
            'Can you bring me the {}?',
            'May I have the {}?',
            'Could you pass me the {}?',
            'Could you bring me the {}?',
            'Could you get me the {}?',
            'Would you mind giving me the {}?',
            'Do you have a {}?',
            'The {} is necessary for me.',
            'Please get a {} for me.',
            'Please give me a {}.',
            'Please bring me the {}.',
            'Please hand me the {}.',
            'Please fetch me the {}.',
            'Please get a {}.',
            'Get me a {}.',
            'Get the {} for me.',
            'Show me a {}.', #Object Detection
            'Show me the {}.',
            'Can you show me the {}?',
            'Could you show me the {}?',
            'Point me the {}.',
            "Please point me the {}.",
            ]

        else:
            templates = ['An item that can {}.', 'An object that can {}.',
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
                    template = random.choice(templates)
                    if template_mode == 'ov':
                        texts = template.format(o,verb)
                    elif template_mode == 'o':
                        texts = template.format(o)
                    else:
                        texts = template.format(verb)
                    image = torch.tensor(label_embedding[o][i],dtype=torch.float)
                    objs.append(o)
                    verbs.append(verb)
                    all_texts.append(texts)
                    images.append(image)


        ###command 저장###
        # with open('data/Resnet101/test_command.json', 'w+') as f: 
        #     f.write(json.dumps(all_texts)) #json 문자열로 변환

        # print(len(objs))
        # print(len(all_texts))



        ###실험###
        # self.objs = []
        # self.verbs = []
        # self.all_texts = []
        # self.images = []
        # for verb, obj in vo_dict.items(): #verb에 해당하는 object40장 다 넣어서 학습
        #     for o in obj: #random.choice(aff_dict[o])
        #         for i in range(len(list(label_embedding.values())[0])): #train:40 test:10
        #             template = random.choice(vo_templates)
        #             texts = template.format(verb)
        #             image = torch.tensor(label_embedding[o][i],dtype=torch.float)
        #             self.objs.append(o)
        #             self.verbs.append(verb)
        #             self.all_texts.append(texts)
        #             self.images.append(image)


        # print(len(self.objs)) #71400
        # print(len(verbs))
        # print(len(all_texts))
        # print(len(images))

        # self.len = len(self.objs)
        ###실함 end###


        if train_val_mode == 'train':
             ###1 positve & n negatives###
            self.train_data = utils.gen_neg_samples(verbs,objs,all_texts,images,vo_dict,n=61)
            self.len = len(self.train_data)
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
            # verb, obj, sentence, affordance = self.verbs[index], self.objs[index], self.all_texts[index], self.images[index]
            # return verb, obj, sentence, affordance

            ###n negative###
            sentence, ret_objs = self.train_data[index]
            return sentence, ret_objs

        elif self.train_val_mode == 'val':
            sentence, ret_objs, val  = self.val_data[index]
            return sentence, ret_objs, val  # 3.3과 다르게 넘파이 배열로 출력 되는 것에 유의 하도록 한다.
    
        else:
            sentence, ret_objs, val = self.test_data[index]
            return sentence, ret_objs, val


    def __len__(self):
        return self.len



