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


###ov-pair를 이용하여 데이터 생성###
###verb로 구성된 template###
# class CustomDataset(Dataset):
#     def __init__(self, label_list, train_embedding, ov_pair, vo_dict, train_val_mode='train'):
#         self.train_val_mode = train_val_mode 
#         vo_templates = ['An item that can {}.', 'An object that can {}.',
#            'Give me something that can {}.', 'Give me an item that can {}.',
#            'Hand me something with which I can {}.',
#            'Give me something with which I can {}.',
#            'Hand me something to {}.', 'Give me something to {}.',
#            'I want something to {}.', 'I need something to {}.']

#         n_objs = []
#         n_verbs = []
#         n_all_texts = []
#         for obj in label_list: #512
#             objs = []
#             verbs = []
#             all_texts = []
#             for i in range(len(train_embedding[0])):
#                 template = random.choice(vo_templates)
#                 verb = random.choice(ov_pair[obj])
#                 texts = template.format(verb)
#                 objs.append(obj)
#                 verbs.append(verb)
#                 all_texts.append(texts)
#             n_objs.append(objs)
#             n_verbs.append(verbs)
#             n_all_texts.append(all_texts)      


#         commands = []
#         images = []
#         objs = []
#         verbs = []
#         for i in range(len(train_embedding[0])): #40
#             for j in range(len(label_list)): #512
#                 n_embedding = train_embedding[j][i]
#                 n_text = n_all_texts[j][i]
#                 n_obj = n_objs[j][i]
#                 n_verb = n_verbs[j][i]
#                 images.append(torch.tensor(n_embedding,dtype=torch.float))
#                 commands.append(n_text)
#                 objs.append(n_obj)
#                 verbs.append(n_verb)

#         # print(len(images))
#         # print(len(commands))
#         # print(len(objs))
#         # print(len(verbs))
   
#         if train_val_mode == 'train':
#             self.train_data = utils.gen_examples(verbs,objs,commands,images,vo_dict)

#         elif train_val_mode == 'val':
#             self.val_data = utils.gen_examples2(verbs,objs,commands,images,vo_dict)


                
#         # print(len(self.train_data)) #40960
#         # print(self.train_data[0])
 

#         self.images = images



#     def __getitem__(self, index):
#         if self.train_val_mode == 'train':
#             verb, obj, sentence, affordance, val = self.train_data[index]
    

#             return affordance, sentence, val

#         elif self.train_val_mode == 'val':
#             sentence, ret_objs, val  = self.val_data[index]

#             return sentence, ret_objs, val  # 3.3과 다르게 넘파이 배열로 출력 되는 것에 유의 하도록 한다.
    




#     def __len__(self):
        
        
#         return len(self.images)





###v-o pair를 이용하여 데이터 생성###
###verb로 구성된 template###
class CustomDataset2(Dataset):
    def __init__(self, label_embedding, vo_dict, train_val_mode='train'):
        self.train_val_mode = train_val_mode 

        vo_templates = ['An item that can {}.', 'An object that can {}.',
           'Give me something that can {}.', 'Give me an item that can {}.',
           'Hand me something with which I can {}.',
        #    'Give me something with which I can {}.',
           'Hand me something to {}.', 'Give me something to {}.',
           'I want something to {}.', 'I need something to {}.',
           'Give me something so I can {}.',
            'Handover something to {}.', #수정됨
            # 'I must use something to {}.',
            'I have to use something to {}.',
            'Bring me something so I can {}',
            # 'Grab something for me so I can {}.',
            'Can you give me something to {}?',
            'Can you give me something so I can {}?',
            'Can you get me something to {}?',
            'Can you get me something so I can {}?',
            'Can you hand me something to {}?',
            'Can you bring me something to {}?',
            # 'May I have something to {}?',
            'Could you pass me something so I can {}?',
            'Could you bring me something so I can {}?',
            # 'Could you get me something so I can {}?',
            'Would you mind giving me something to {}?',
            'Do you have something I can {}?',
            'Without something, I am unable to {}.',
            # 'Without something, I cannot to {}.',
            'something is necessary for me to be able to {}.',
            'Please get something for me so I can {}.',
            'Please give me something so I can {}.',
            'Please bring me something so I can {}.',
            'Please hand me something so I can {}.',
            # 'Please fetch me something so I can {}.',
            'Please get something so I can {}.',
            'Get me something so I can {}.',
            'Get something for me so I can {}.',
            # 'Use something to {}.',
            'Would you use something to {}.',
            'Please use something to {}.',
            'Show me something I can {}.', #Object Detection
            'Show me something so I can {}.',
            # 'Show me something with which I can {}.',
            'Can you show me something to {}?',
            'Could you show me something I can {}?',
            'Please show me something so I can {}.',
            # 'point me something so I can {}.',
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
                    image = torch.tensor(label_embedding[o][i],dtype=torch.float)
                    objs.append(o)
                    verbs.append(verb)
                    all_texts.append(texts)
                    images.append(image)

   
        # print(len(objs)) #71520
        # print(len(verbs))
        # print(len(all_texts))
        # print(len(images))

        if train_val_mode == 'train':
            ###1 positve & 1 negative###
            # self.train_data = utils.gen_examples(verbs,objs,all_texts,images,vo_dict)

             ###1 positve & n negatives###
            self.train_data = utils.gen_neg_samples(verbs,objs,all_texts,images,vo_dict)
            self.len = len(self.train_data)
            # print(self.len)
       
       
        elif train_val_mode == 'val':
            self.val_data = utils.gen_examples2(verbs,objs,all_texts,images,vo_dict)
            self.len = len(self.val_data)

        else:
            self.test_data = utils.gen_examples2(verbs,objs,all_texts,images,vo_dict)
            self.len = len(self.test_data)
                
        # print(len(self.train_data)) #71520
        # print(len(self.val_data))  #17880
        # print(len(self.test_data)) #14000


 

    def __getitem__(self, index):
        if self.train_val_mode == 'train':

            ###1 negative###
            verb, obj, sentence, affordance, val = self.train_data[index]
            return affordance, sentence, val
            ###100 negative###
            # sentence, ret_objs = self.train_data[index]
            
            # return sentence, ret_objs

        elif self.train_val_mode == 'val':
            sentence, ret_objs, val  = self.val_data[index]

            return sentence, ret_objs, val  # 3.3과 다르게 넘파이 배열로 출력 되는 것에 유의 하도록 한다.
    
        else:
            sentence, ret_objs, _  = self.test_data[index]
            
            return sentence, ret_objs


    def __len__(self):
        return self.len



