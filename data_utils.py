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



class CustomDataset(Dataset):
    def __init__(self, label_list, label_embedding, ov_pair, vo_pair, train_val_mode='train', o_ov_mode = 'o'):       
        if o_ov_mode == 'ov':
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
        
        else:
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






            # templates = [
            # 'a bad photo of a {}.',
            # 'a photo of many {}.',
            # 'a sculpture of a {}.',
            # 'a photo of the hard to see {}.',
            # 'a low resolution photo of the {}.',
            # 'a rendering of a {}.',
            # 'graffiti of a {}.',
            # 'a bad photo of the {}.',
            # 'a cropped photo of the {}.',
            # 'a tattoo of a {}.',
            # 'the embroidered {}.',
            # 'a photo of a hard to see {}.',
            # 'a bright photo of a {}.',
            # 'a photo of a clean {}.',
            # 'a photo of a dirty {}.',
            # 'a dark photo of the {}.',
            # 'a drawing of a {}.',
            # 'a photo of my {}.',
            # 'the plastic {}.',
            # 'a photo of the cool {}.',
            # 'a close-up photo of a {}.',
            # 'a black and white photo of the {}.',
            # 'a painting of the {}.',
            # 'a painting of a {}.',
            # 'a pixelated photo of the {}.',
            # 'a sculpture of the {}.',
            # 'a bright photo of the {}.',
            # 'a cropped photo of a {}.',
            # 'a plastic {}.',
            # 'a photo of the dirty {}.',
            # 'a jpeg corrupted photo of a {}.',
            # 'a blurry photo of the {}.',
            # 'a photo of the {}.',
            # 'a good photo of the {}.',
            # 'a rendering of the {}.',
            # 'a {} in a video game.',
            # 'a photo of one {}.',
            # 'a doodle of a {}.',
            # 'a close-up photo of the {}.',
            # 'a photo of a {}.',
            # 'the origami {}.',
            # 'the {} in a video game.',
            # 'a sketch of a {}.',
            # 'a doodle of the {}.',
            # 'a origami {}.',
            # 'a low resolution photo of a {}.',
            # 'the toy {}.',
            # 'a rendition of the {}.',
            # 'a photo of the clean {}.',
            # 'a photo of a large {}.',
            # 'a rendition of a {}.',
            # 'a photo of a nice {}.',
            # 'a photo of a weird {}.',
            # 'a blurry photo of a {}.',
            # 'a cartoon {}.',
            # 'art of a {}.',
            # 'a sketch of the {}.',
            # 'a embroidered {}.',
            # 'a pixelated photo of a {}.',
            # 'itap of the {}.',
            # 'a jpeg corrupted photo of the {}.',
            # 'a good photo of a {}.',
            # 'a plushie {}.',
            # 'a photo of the nice {}.',
            # 'a photo of the small {}.',
            # 'a photo of the weird {}.',
            # 'the cartoon {}.',
            # 'art of the {}.',
            # 'a drawing of the {}.',
            # 'a photo of the large {}.',
            # 'a black and white photo of a {}.',
            # 'the plushie {}.',
            # 'a dark photo of a {}.',
            # 'itap of a {}.',
            # 'graffiti of the {}.',
            # 'a toy {}.',
            # 'itap of my {}.',
            # 'a photo of a cool {}.',
            # 'a photo of a small {}.',
            # 'a tattoo of the {}.',
            # ]

        # print(len(list(label_embedding.values())[0]))
        # print(len(ov_pair))
  

        self.commands = []
        self.images = []
        for i in range(len(list(label_embedding.values())[0])): #test:50
            for obj, v in ov_pair.items(): #594
                template = random.choice(templates) #40개중 random
                if o_ov_mode == 'ov':
                    verb = random.choice(v)
                    texts = template.format(obj,verb)
                else:
                    texts = template.format(obj)
                

                image = torch.tensor(label_embedding[obj][i],dtype=torch.float)
                self.commands.append(texts)
                self.images.append(image)


        self.len = len(self.commands)


        # n_objs = []
        # n_verbs = []
        # n_all_texts = []
        # for obj in label_list: #512
        #     objs = []
        #     verbs = []
        #     all_texts = []
        #     for i in range(50):
        #         template = random.choice(vo_templates) #40개 템플릿중 아무거나 가져옴
        #         verb = random.choice(ov_pair[obj])
        #         texts = template.format(obj,verb)
        #         objs.append(obj)
        #         verbs.append(verb)
        #         all_texts.append(texts)
        #     n_objs.append(objs)
        #     n_verbs.append(verbs)
        #     n_all_texts.append(all_texts)            
          

        # # print(len(n_objs))
        # # print(len(n_verbs))
        # # print(len(n_all_texts))
        # # print(len(n_objs[0]))
        # # print(len(n_verbs[0]))
        # # print(len(n_all_texts[0]))
 

   
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


        # self.commands = []
        # self.images = []
        # self.objs = []
        # self.verbs = []
        # for i in range(len(train_embedding[0])): #40
        #     for j in range(len(label_list)): #512
        #         n_embedding = train_embedding[j][i]
        #         n_text = n_all_texts[j][i]
        #         n_obj = n_objs[j][i]
        #         n_verb = n_verbs[j][i]
        #         self.images.append(n_embedding)
        #         self.commands.append(n_text)
        #         self.objs.append(n_obj)
        #         self.verbs.append(n_verb)
        #         # self.images.append(torch.tensor(y, dtype=torch.float))
                
        #         # print(len(all_texts))

        # self.len = len(self.images)

        
        # print(len(self.images))
        # print(len(self.commands))
        # print(len(self.images))
        # print(len(self.commands))
  
###
    def __getitem__(self, index):
        image_embedding = self.images[index]
        command = self.commands[index]

        
        return image_embedding, command # 3.3과 다르게 넘파이 배열로 출력 되는 것에 유의 하도록 한다.
    
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





