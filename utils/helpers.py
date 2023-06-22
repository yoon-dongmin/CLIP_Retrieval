import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
# Helper functions

'''
Process the input data (build word vocab, map each word in the language command
to its ID number, etc.)
Each output datapoint has the form [verb, object, command, embedding, image]
where 'verb' and 'object' are strings, 'command' is a list of word IDs,
'embedding' is a numpy array representing the image/object,
and 'image' is a string containing the image file name
'''

'''
language command에 있는 각 word를 ID number에 mapping
Image embedding => numpy array
image => a string containing the image file name 
'''
random.seed(100)

def read_data(train, test):
    word2id = {'UNK':0}
    id2word = {0:'UNK'}
    with open(train, 'r') as train_file:
        with open(test, 'r') as test_file:
            train_data = list(csv.reader(train_file))
            test_data = list(csv.reader(test_file))
            train_dt, test_dt = [], []
            for row in train_data:
                affordances = np.fromstring(row[3][1:-1], dtype=float, sep=',') #[]제외
                sentence = []
                s = row[2].lower().split()
                for word in s:
                    if word not in word2id: # build word vocab
                        word2id[word] = len(word2id) #
                        id2word[word2id[word]] = word
                    sentence.append(word2id[word]) # map each word to its ID number
                train_dt.append([row[0], row[1], sentence, affordances, row[4]])
            for row in test_data:
                affordances = np.fromstring(row[3][1:-1], dtype=float, sep=',')
                sentence = []
                s = row[2].split()
                for word in s:
                    if word not in word2id:
                        word2id[word] = len(word2id)
                        id2word[word2id[word]] = word
                    sentence.append(word2id[word])
                test_dt.append([row[0], row[1], sentence, affordances, row[4]])
    return (train_dt, test_dt, word2id, id2word)


def read_data2(train, test):
    with open(train, 'r') as train_file:
        with open(test, 'r') as test_file:
            train_data = list(csv.reader(train_file))
            test_data = list(csv.reader(test_file))
            train_dt, test_dt = [], []
            for row in train_data:
                affordances = np.fromstring(row[3][1:-1], dtype=float, sep=',') #[]제외
                train_dt.append([row[0], row[1], row[2], affordances, row[4]])
            for row in test_data:
                affordances = np.fromstring(row[3][1:-1], dtype=float, sep=',') 
                s = row[2].split()
                test_dt.append([row[0], row[1], row[2], affordances, row[4]])
    return (train_dt, test_dt)


'''
Generate positive and negative examples from the data.
Each output datapoint has the form [verb, object, command, embedding, image, value],
with 'value' being 1.0 for a positive example and -1.0 for a negative example
'''

'''
data를 이용하여 positive와 negative example 만들기
[verb, object, command, embedding, image, value]형태
1.0이면 positive -1.0이면 negative
'''
# def gen_examples(image_embeddings, obj, vo_dict):
#     pos = [(row[0], row[1], row[2], row[3], row[4], 1.0) for row in data]
#     neg = []
#     for row in pos:
#         while True:
#             neg_candidate = random.choice(pos) #pos의 값 중 아무거나 하나 가져옴
#             #only sample examples with objects that cannot be paired
#             #with the current verb to use as a negative example
#             if (neg_candidate[1] not in vo_dict[row[0]]): #neg_candidate의 object가 vo_dict의 verb에 해당하는 object에 없다면
#                 neg += [(row[0], neg_candidate[1], row[2], neg_candidate[3], neg_candidate[4], -1.0)]
#                 break
#     all_data = pos + neg
#     return all_data, pos

def gen_examples(verb,obj,command,image_embedding,vo_dict):
    pos = [(verb[i], obj[i], command[i], image_embedding[i], 1.0) for i in range(len(image_embedding))]
    neg = []
    for row in pos:
        # print(123123)
        while True:
            neg_candidate = random.choice(pos) #pos의 값 중 아무거나 하나 가져옴
            #only sample examples with objects that cannot be paired
            #with the current verb to use as a negative example
            if (neg_candidate[1] not in vo_dict[row[0]]): #neg_candidate의 object가 vo_dict의 verb에 해당하는 object에 없다면
                neg += [(row[0], neg_candidate[1], row[2], neg_candidate[3], -1.0)]
                break
    all_data = pos + neg
    return all_data


def gen_examples2(verbs,objs,commands,image_embeddings,vo_dict):
    ret_set = []
    for i in range(len(image_embeddings)):
        verb, obj, command, image_embedding = verbs[i], objs[i], commands[i], image_embeddings[i]
        l = [command]
        # n = [1.0]
        ret_objs = [[obj, image_embedding]]
        all_o = [obj]
       
        while len(ret_objs) < 5:
            num = random.randrange(len(objs))
            if (objs[num] not in vo_dict[verb]) and (objs[num] not in all_o):
                # print(objs[num])
                ret_objs.append([objs[num], image_embeddings[num]])
                all_o.append(objs[num])
        # print(len(ret_objs))
    
        l.append(ret_objs)
        l.append(1.0)
        ret_set.append(l) #[[command,[o,e][o,e][o,e][o,e][o,e]],...,]
        # print(ret_set)

    return ret_set




def gen_hundred_examples(verbs,objs,commands,image_embeddings,vo_dict):
    ret_set = []
    for i in range(len(image_embeddings)):
        verb, obj, command, image_embedding = verbs[i], objs[i], commands[i], image_embeddings[i]
        l = [command]
        ret_embedds = [image_embedding] 
        all_o = [obj]
       
        while len(ret_embedds) < 100:
            num = random.randrange(len(objs))
            if (objs[num] not in vo_dict[verb]) and (objs[num] not in all_o):
                # print(objs[num])
                image_embeddings[num] /= image_embeddings[num].norm(dim=-1, keepdim=True) ##split에 추가할 예정
                ret_embedds.append(image_embeddings[num]) #, objs[num]
                all_o.append(objs[num])
        # print(len(ret_embedds))
        if len(ret_embedds) == 100:
            ret_embedds = torch.stack(ret_embedds, dim=1).cuda()
            # print(ret_embedds.size())
            l.append(ret_embedds)
            ret_set.append(l) #[[command,e,e,e,,..],...,]
            # print(ret_set)

    return ret_set



def loss_plot(loss, y_label, file_name):
    plt.figure()
    for train_loss, eval_loss in loss:
        plt.plot(train_loss,'r-', label='train loss')
        plt.plot(eval_loss,'b-', label='eval loss')

    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(file_name)


def acc_plot(accuracy, y_label, file_name):
    plt.figure()
    for top1,top2 in accuracy:
        plt.plot(top1,'r-', label='top1 acc')
        plt.plot(top2,'b-', label='top2 acc')

    # for acc in acc2:
    #     line_label = 'top2 accuracy'
    #     plt.plot(acc,'b-', label=line_label)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(file_name)

