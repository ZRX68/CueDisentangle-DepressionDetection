

import torch.utils.data as udata
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch
# from pydub import AudioSegment
import random
import logging
from torchvision import transforms
from PIL import Image

#定义了视频和音频数据的标准形状
normalVideoShape = 915
normalAudioShape = 186

class temporalMask():                                                                  #随机丢弃部分帧，以增加数据的多样性
    def __init__(self, drop_ratio):
        self.ratio = drop_ratio                                                        #丢弃率
    def __call__(self, frame_indices):
        frame_len = frame_indices.shape[0]                                             #计算行数
        sample_len = int(self.ratio*frame_len)                                         #丢弃样本的数量
        sample_list = random.sample([i for i in range(0, frame_len)], sample_len)      #随机生成丢弃的样本
        frame_indices[sample_list,:] = 0                                               #将指定行的所有列设置为0
        return frame_indices

#test = temporalMask()

class TemporalRandomCrop(object):         #随机裁剪视频帧，并根据预设大小和下采样比例返回帧索引

    def __init__(self, size, downsample): #初始化
        self.size = size                  #裁剪后视频的长度
        self.downsample = downsample      #下采样比例


    def __call__(self, frame_indices):
        downsample = max(min(int(self.vid_duration/self.size), self.downsample),1)  #设置合适的下采样率

        end_index = min(self.begin_index + self.clip_duration, self.vid_duration)   #确定裁剪视频的结束索引
        out = frame_indices[self.begin_index:end_index]                             #从视频帧索引数组中提取一部分帧索引
        while len(out) < self.clip_duration:                                        #通过不断重复out数组中的元素来扩展out，使其长度达到clip_duration
            for index in out:
                if len(out) >= self.clip_duration:
                    break
                out.append(index)
        selected_frames = [i for i in range(self.begin_index, self.clip_duration+self.begin_index, downsample)]#每隔downsample个帧选择一个帧作为裁剪后的结果
        return selected_frames
    
    def randomize_parameters(self, frame_indices):
        self.vid_duration  = len(frame_indices)                                #视频的总帧数
        downsample = max(min(int(self.vid_duration/self.size), self.downsample),1)#计算下采样率
        self.clip_duration = self.size * downsample                               #计算裁剪后的视频长度

        rand_end = max(0, self.vid_duration - self.clip_duration-1)               #计算裁剪的上限位置
        self.begin_index = random.randint(0, rand_end)                            #确定开始裁剪的索引
    
class AffectnetSampler(torch.utils.data.sampler.Sampler):   #通过指定的权重采样数据，确保每种表达都有公平的抽取机会，处理标签不平衡问题

    def __init__(self, dataset):

        self.indices = list(range(len(dataset)))                     #代表了数据集中每个样本的索引
        self.num_samples = len(self.indices)                         #记录总采样数量

        expression_count = [0] * 63                                  #用于统计每种特征的出现次数
        for idx in self.indices:                                     #
            label = dataset.label[idx]
            expression_count[int(label)] += 1

        self.weights = torch.zeros(self.num_samples)                 #创建一个长度为num_samples的全零张量
        for idx in self.indices:
            label = dataset.label[idx]
            self.weights[idx] = 1. / expression_count[int(label)]    #出现次数较少的标签将拥有更大的权重，处理标签不平衡的问题


    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))#返回采样后对应的样本索引

    def __len__(self):
        return self.num_samples  #返回样本数量

def chouzhen(_feature):  #从输入特征数组中每隔 6 个元素抽取一个元素，并将他们垂直堆叠起来
    flag = 0
    for i in range(0, len(_feature), 6):
        if flag == 0:
            feature = _feature[i]
            flag = 1
        else:
            feature = np.vstack((feature, _feature[i]))
    return feature


class MyDataLoader(udata.Dataset):        #主要的数据加载类，负责读取视频和音频文件，以及对应的标签
    def __init__(self, videoFileName, AudioFileName, Kfolds, labelPath, type) -> None:
        super().__init__()
        if type == "train":
            self.temp = temporalMask(0.25)      #仅训练集进行数据增强
        else :
            self.temp = None
        
        self.transform = TemporalRandomCrop(size=16, downsample=4)
        self.videoList = []
        self.audioList = []
        self.label = []
        self.type = type

        for file in Kfolds:
            file = str(file)
            id = file.split('.')[0]
            self.videoList.append(os.path.join(videoFileName, file))
            # file_audio = file.replace(".csv", ".npy")
            # print(file_audio, file)
            self.audioList.append(os.path.join(AudioFileName, file))
            
                
            file_csv = pd.read_csv(os.path.join(labelPath, file.replace(".npy", "_Depression.csv")))#修改

            
            bdi = int(file_csv.columns[0])
            self.label.append(bdi)
        # print('flag')
            
    def __getitem__(self, index: int):     #固定输入数据形状（915帧视频，186帧音频）


        videoData, audioData, label, = np.load(self.videoList[index], allow_pickle=True), np.load(self.audioList[index], allow_pickle=True), self.label[index]
        label = np.array(label)
        
        if self.temp is not None:
            videoData = self.temp(videoData)
        label = torch.from_numpy(label).type(torch.float)
        # print(videoData.dtype, audioData.dtype)
        if videoData.dtype == object:
            # videoData = pd.Series(videoData)
            # videoData = pd.to_numeric(videoData, errors='coerce').values.astype(np.float64)
            videoData = videoData.astype(float)
        # if audioData.dtype == object:
        #     audioData = pd.Series(audioData)
        #     audioData = pd.to_numeric(audioData, errors='coerce').values.astype(np.float32)
        videoData = torch.from_numpy(videoData)
        audioData = torch.from_numpy(audioData)

        if audioData.shape[0] > normalAudioShape:
            audioData = audioData[:180,:]
        if videoData.shape[0] > normalVideoShape:
            videoData = videoData[:915,]
        assert videoData.shape[0] <= normalVideoShape
        assert audioData.shape[0] <= normalAudioShape
        assert videoData.shape[0] > 0
        assert audioData.shape[0] > 0

        if videoData.shape[0] < normalVideoShape:
            zeroPadVideo = nn.ZeroPad2d(padding=(0,0,0,normalVideoShape-videoData.shape[0]))
            videoData = zeroPadVideo(videoData)
        if audioData.shape[0] < normalAudioShape:
            zeroPadAudio = nn.ZeroPad2d(padding=(0,0,0,normalAudioShape-audioData.shape[0]))
            audioData = zeroPadAudio(audioData)

        videoData = videoData.type(torch.float)
        audioData = audioData.type(torch.float)
        
        if self.type == "train":
            return videoData, audioData, label
        if self.type == "dev":
            return videoData, audioData, label

    def __len__(self) -> int:     #返回数据集的大小
        return len(self.videoList)

