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
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch

random.seed(100)



class Train_Dataset(Dataset):
    def __init__(self, label_list, train_embedding, ov_pair):       

        # print(len(train_embedding[0]))
        vo_templates = [
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
        'I need the {} so I can {}.',
        'Give me the {} so I can {}.',
        # 'To {}, I need the {}.', #이건 삭제
        'I must use the {} to {}.',
        'I have to use the {} to {}.',
        'Bring me a {} so I can {}',
        'Grab a {} for me so I can {}.',
        'Can you give me the {} to {}?',
        # 'Can you give me a {} so I can {}?',
        # 'Can you get me the {} to {}?',
        'Can you get me a {} so I can {}?',
        'Can you hand me the {} to {}?',
        'Can you bring me the {} to {}?',
        'May I have the {} to {}?',
        'Could you pass me the {} so I can {}?',
        'Could you bring me the {} so I can {}?',
        'Could you get me the {} so I can {}?',
        'Would you mind giving me the {} to {}?',
        'Do you have a {} I can {}?',
        # 'Without the {}, I am unable to {}.',
        # 'Without the {}, I cannot to {}.',
        # 'The {} is necessary for me to be able to {}.',
        'Please get a {} for me so I can {}.',
        'Please give me a {} so I can {}.',
        'Please bring me the {} so I can {}.',
        'Please hand me the {} so I can {}.',
        'Please fetch me the {} so I can {}.',
        'Please get a {} so I can {}.',
        'Get me a {} so I can {}.',
        'Get the {} for me so I can {}.',
        # 'Show me the {} I can {}.', #Object Detection
        'Show me a {} so I can {}.',
        'Show me the {} with which I can {}.',
        'Can you show me the {} to {}?',
        'Could you show me the {} I can {}?',
        # 'Please show me a {} so I can {}.',
        # 'Point me a {} I can {}.',
        # 'Please point me the {} so I can {}.',
        ]
        

        n_objs = []
        n_verbs = []
        n_all_texts = []
        for obj in label_list: #512
            objs = []
            verbs = []
            all_texts = []
            for template in vo_templates:
                verb = random.choice(ov_pair[obj])
                texts = template.format(obj,verb)
                objs.append(obj)
                verbs.append(verb)
                all_texts.append(texts)
            n_objs.append(objs)
            n_verbs.append(verbs)
            n_all_texts.append(all_texts)            
          

        # print(len(n_objs))
        # print(len(n_verbs))
        # print(len(n_all_texts))
        # print(len(n_objs[0]))
        # print(len(n_verbs[0]))
        # print(len(n_all_texts[0]))
        # print(n_objs[0])
        # print(n_verbs[0])
        # print(n_all_texts[0])

        

        # for i in range(len(self.embedding_list[0][:40])): #40개 image사용
        #     for obj in self.label_list: #512
        #         verb = random.choice(ov_pair[obj])
        #         texts = [template.format(obj, verb) for template in self.vo_templates]
        #         self.all_texts.append(texts)
        #         self.objs.append(obj)
        #         self.verbs.append(verb)

        #     for j in range(len(self.label_list)):
        #         y = self.embedding_list[j][i]
        #         self.images.append(y)


        self.commands = []
        self.images = []
        self.objs = []
        self.verbs = []
        for i in range(len(train_embedding[0])): #40
            for j in range(len(label_list)): #512
                n_embedding = train_embedding[j][i]
                n_text = n_all_texts[j][i]
                n_obj = n_objs[j][i]
                n_verb = n_verbs[j][i]
                self.images.append(n_embedding)
                self.commands.append(n_text)
                self.objs.append(n_obj)
                self.verbs.append(n_verb)
                # self.images.append(torch.tensor(y, dtype=torch.float))
                
                # print(len(all_texts))

        self.len = len(self.images)

        
        # print(len(self.images))
        # print(len(self.commands))
        # print(len(self.images))
        # print(len(self.commands))
        # exit()

    def __getitem__(self, index):
        image_embedding = torch.tensor(self.images[index],dtype=torch.float)
        command = self.commands[index]
        obj = self.objs[index]
        verb = self.verbs[index]
   
        
        return image_embedding, command, obj, verb # 3.3과 다르게 넘파이 배열로 출력 되는 것에 유의 하도록 한다.
    
    def __len__(self):
        return self.len 
    
    
class Val_Dataset(Dataset):
    def __init__(self, label_list, test_embedding, ov_pair):       
        # print(len(test_embedding[0]))
       
        vo_templates = [
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
        ]
        

        n_all_texts = []
        for obj in label_list: #512
            all_texts = []
            for template in vo_templates:
                verb = random.choice(ov_pair[obj])
                texts = template.format(obj,verb)
                all_texts.append(texts)
            n_all_texts.append(all_texts)            
          

        self.commands = []
        self.images = []
        for i in range(len(test_embedding[0])): #10
            for j in range(len(label_list)): #512
                n_embedding = test_embedding[j][i]
                n_text = n_all_texts[j][i]
                self.images.append(n_embedding)
                self.commands.append(n_text)



        self.len = len(self.images)



    def __getitem__(self, index):
        image_embedding = torch.tensor(self.images[index],dtype=torch.float)
        command = self.commands[index]

   
        
        return image_embedding, command
    
    def __len__(self):
        return self.len 



# Generate retrieval tasks (from positive test examples) to test the model
def genRet(test_pos, vo_dict):
    ret_set = []
    for verb, obj, sentence, affordances, img, val in test_pos:
        l = [sentence]
        #the object included in the current test example
        #is the first candidate object for this retrieval task
        ret_objs = [[obj, affordances, img]]
        all_o = [obj]
        #each retrieval task includes ret_num (5) candidate objects
        #for the model to select from
        while len(ret_objs) < opt.ret_num: #나머지 4개는 다른것 추가
            sample = random.choice(test_pos) #test_pos에서 random하게 하나 가져옴
            #only sample objects that cannot be paired with the current verb
            #and make sure that all objects in the retrieval set are unique
            #(from different object classes)
            if (sample[1] not in vo_dict[verb]) and (sample[1] not in all_o):
                ret_objs.append([sample[1], sample[3], sample[4]])
                all_o.append(sample[1])
        l.append(ret_objs)
        ret_set.append(l)
    return ret_set


class Test_Dataset(Dataset):
    def __init__(self, label_list, train_embedding, ov_pair):       

        # print(len(train_embedding[0]))
        vo_templates = [
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
        'I need the {} so I can {}.',
        'Give me the {} so I can {}.',
        'To {}, I need the {}.', #이건 삭제
        'I must use the {} to {}.',
        'I have to use the {} to {}.',
        'Bring me a {} so I can {}',
        'Grab a {} for me so I can {}.',
        'Can you give me the {} to {}?',
        'Can you give me a {} so I can {}?',
        'Can you get me the {} to {}?',
        'Can you get me a {} so I can {}?',
        'Can you hand me the {} to {}?',
        'Can you bring me the {} to {}?',
        'May I have the {} to {}?',
        'Could you pass me the {} so I can {}?',
        'Could you bring me the {} so I can {}?',
        'Could you get me the {} so I can {}?',
        'Would you mind giving me the {} to {}?',
        'Do you have a {} I can {}?',
        'Without the {}, I am unable to {}.',
        'Without the {}, I cannot to {}.',
        'The {} is necessary for me to be able to {}.',
        'Please get a {} for me so I can {}.',
        'Please give me a {} so I can {}.',
        'Please bring me the {} so I can {}.',
        'Please hand me the {} so I can {}.',
        'Please fetch me the {} so I can {}.',
        'Please get a {} so I can {}.',
        'Get me a {} so I can {}.',
        'Get the {} for me so I can {}.',
        'Show me the {} I can {}.', #Object Detection
        'Show me a {} so I can {}.',
        'Show me the {} with which I can {}.',
        'Can you show me the {} to {}?',
        'Could you show me the {} I can {}?',
        'Please show me a {} so I can {}.',
        'Point me a {} I can {}.',
        'Please point me the {} so I can {}.',
        ]
        

        n_objs = []
        n_verbs = []
        n_all_texts = []
        for obj in label_list: #512
            objs = []
            verbs = []
            all_texts = []
            for template in vo_templates:
                verb = random.choice(ov_pair[obj])
                texts = template.format(obj,verb)
                objs.append(obj)
                verbs.append(verb)
                all_texts.append(texts)
            n_objs.append(objs)
            n_verbs.append(verbs)
            n_all_texts.append(all_texts)            
          
        self.commands = []
        self.images = []
        self.objs = []
        self.verbs = []
        for i in range(len(train_embedding[0])): #40
            for j in range(len(label_list)): #512
                n_embedding = train_embedding[j][i]
                n_text = n_all_texts[j][i]
                n_obj = n_objs[j][i]
                n_verb = n_verbs[j][i]
                self.images.append(n_embedding)
                self.commands.append(n_text)
                self.objs.append(n_obj)
                self.verbs.append(n_verb)
                # self.images.append(torch.tensor(y, dtype=torch.float))
                
                # print(len(all_texts))

        self.len = len(self.images)

        
        # print(len(self.images))
        # print(len(self.commands))
        # print(len(self.images))
        # print(len(self.commands))
        # exit()

    def __getitem__(self, index):
        image_embedding = torch.tensor(self.images[index],dtype=torch.float)
        command = self.commands[index]
        obj = self.objs[index]
        verb = self.verbs[index]
   
        
        return image_embedding, command, obj, verb # 3.3과 다르게 넘파이 배열로 출력 되는 것에 유의 하도록 한다.
    
    def __len__(self):
        return self.len 









def get_testdata(label_embedding, ov_pair):
     
    label_list = list(label_embedding.keys())
    embedding_list = list(label_embedding.values())
    # print(self.label_list)
    # print(self.embedding_list[0])
    # print(len(self.embedding_list))
    # print(len(self.embedding_list[0]))
    # print(len(self.embedding_list[0][:40]))
    

    vo_templates = [
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
    ]
    
    n_all_texts = []
    for obj in label_list: #512
        all_texts = []
        for template in vo_templates:
            verb = random.choice(ov_pair[obj])
            texts = template.format(obj,verb)
            all_texts.append(texts)
        n_all_texts.append(all_texts)            
    
    commands = []
    images = []
    for i in range(len(embedding_list[0][-10:])): #10
        for j in range(len(label_list)): #512
            n_embedding = torch.tensor(embedding_list[j][i],dtype=torch.float) 
            n_text = n_all_texts[j][i]
            images.append(n_embedding)
            commands.append(n_text)
 
    return images, commands    