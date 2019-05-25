# -*- coding:utf-8 -*-
import tensorflow as tf
import os, codecs
from tqdm import tqdm
import numpy as np
import nltk
import re
import pprint
import random
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler


class Markov(object):
    def __init__(self, file):
        self.table=defaultdict(defaultdict)
        self.file=file

    def read_file(self):
        with open(self.file) as file:
            for line in file:
                tokens=line.strip().split()
                former=None
                pole=0
                # Add a special key with just beginning words
                # 记录句子中开头的单词
                self.table['#BEGIN#']['UN_kown']=0
                self.table['#BEGIN#'][tokens[0]]=self.table['#BEGIN#'].setdefault(tokens[0],0)+1

                # loop through each word, and if we have enough to add dictionary item, then add
                for item in tokens:
                    # not enough items
                    if pole < 1:
                        pole+=1
                        former=tokens[0]
                        self.table[former]['UN_kown']=0
                        continue
                    # If we already have the item, then add it, otherwise add to empty list
                    self.table[former][item]=self.table[former].setdefault(item,0)+1
                    former=item
        for key in self.table:
            store=list(zip(*self.table[key].items()))
            tmp=tuple(1-MinMaxScaler(feature_range=(0.1, 1)).fit_transform(np.array(list(store[1])).reshape(-1, 1)).reshape(1,-1)[0])
            store[1]=tmp
            self.table[key]=dict(set(zip(*store)))

    def save(self,filename):
        # /home1/bqw/att_gen/data/mar_dict
        markovDictFile = open(filename, 'w')
        pprint.pprint(self.table, markovDictFile)

    def load(self, filename):
        with open(filename, 'r') as inf:
            self.table = eval(inf.read())

    def get_reward(self,text,iw_dict,batch_size,sequence_length):
        reward=np.zeros((batch_size,sequence_length))
        eof_code = str(int(len(iw_dict)))
        for i in range(batch_size):
            for j in range(sequence_length):
                if j==0:
                    # text是数字要转换为文本
                    former=str(int(text[i][j]))
                    # print(former)
                    # 第一个单词word
                    # print(iw_dict.get(former,'unkown'))
                    print('the first word {}.'.format(former))

                    # print('the first word {}.'.format(iw_dict.get(former,'unkown'),0.1))
                    # 第一个单词的概率
                    # print('the first word {}.'.format(self.table['#BEGIN#'].get(iw_dict.get(former,'unkown'),0.1)))
                    print('the first word {}.'.format(self.table['#BEGIN#'].get(iw_dict.get(former,'unkown'),1-0)))
                    # reward[i][j]=self.table['#BEGIN#'].get(iw_dict.get(former,'unkown'),0.1)
                    reward[i][j]=self.table['#BEGIN#'].get(iw_dict.get(former,'unkown'),1-0)
                latter = str(int(text[i][j]))
                if former==eof_code:
                    if latter==eof_code:
                        reward[i][j] = 1
                        former = latter
                        continue
                    else:
                        reward[i][j] = 0.1
                        former = latter
                        continue
                # reward[i][j] = self.table[former].get(iw_dict.get(latter, 'unkown'),0.1)
                reward[i][j] = self.table[former].get(iw_dict.get(latter, 'unkown'),1-0)
                former=latter
        reward=reward.reshape(-1)
        return reward

# table=defaultdict(defaultdict)
# table['1']['wo']=1
# table['1']['shi']=2
# table['2']['zou']=4
# table['2']['ni']=5
# markov = Markov('/home1/bqw/att_gen/data/image_coco.txt')
# markov.read_file()

