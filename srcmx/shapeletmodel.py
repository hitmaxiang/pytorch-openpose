from os import dup
import time
from numpy.lib.function_base import _copy_dispatcher, copy
import torch
from torch._C import dtype, float32
import utilmx

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ShapeletNetModel(nn.Module):
    def __init__(self, shape, query=None, lr=0.01):
        super().__init__()
        m, d = shape
        # 因为 query 最后要与 sequence 做卷积， 所以尺寸应该为 1 x D x m
        if query is None:
            query = torch.randn(1, d, m, dtype=torch.float32)
        self.query = nn.Parameter(data=query, requires_grad=True)
        self.weight = nn.Parameter(data=torch.ones(1, d, m), requires_grad=False)
        # self.weight = torch.ones(1, d, m)
        self.CLS = nn.Linear(1, 2)
        self.optimer = torch.optim.Adam([self.query, self.CLS.weight, self.CLS.bias], lr=lr)
        # self.optimer = torch.optim.SGD([self.query, self.CLS.weight, self.CLS.bias], lr=lr, momentum=0.9)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def __call__(self, X, Y):
        # X 的 shape 应该是 （Batch x D x T）
        with torch.no_grad():
            DIS = self.slidingdistance(X)
            DIS, loc = torch.min(DIS, dim=1, keepdim=True)
            _, predicts = torch.max(self.CLS(DIS), dim=1)
            score = int((predicts == Y).sum())/Y.shape[0]
        
        return DIS[:, 0], loc[:, 0], score

    def forward(self, X, label):
        # X 的 shape 应该是 （Batch x D x T）
        DIS = self.slidingdistance(X)
        DIS = torch.min(DIS, dim=1, keepdim=True)[0]
        meadis = torch.mean(DIS[label == 1]) - torch.mean(DIS[label == 0])
        Y = self.CLS(DIS)
        return Y, meadis

    def train(self, X, Y, epochs=10000):
        temploss = float('inf')
        counter = 0
        for i in range(epochs):
            Y_e, meadis = self.forward(X, Y)
            # loss = self.loss_fn(Y_e, Y) + meadis
            loss = self.loss_fn(Y_e, Y)
            # if loss.item() > temploss:
            #     if counter > 100:
            #         break
            #     else:
            #         counter += 1
            # else:
            #     temploss = loss.item()
            #     counter = 0

            self.optimer.zero_grad()
            loss.backward()
            self.optimer.step()
            with torch.no_grad():
                _, predics = torch.max(Y_e, dim=1)
                correct = int((predics == Y).sum())
            if i % 1000 == 0:
                print('epoch: %d, loss %f, accuracy: %f' % (i, loss, correct/Y.shape[0]))
        
        print('\n finish the train:\n')
        print('epoch: %d, loss %f, accuracy: %f' % (i, loss, correct/Y.shape[0]))
    
    def localizeshape(self, X):
        with torch.no_grad():
            DIS = self.slidingdistance(X)
            dis, loc = torch.min(DIS, dim=1)
        return loc, dis
    
    def getshapelet(self):
        shapelet = self.query.detach().squeeze(0).permute(1, 0)
        return shapelet
    
    def slidingdistance(self, X):
        QQ = torch.sum(torch.square(self.query))
        XX = F.conv1d(torch.square(X), weight=self.weight)
        QX = F.conv1d(X, weight=self.query)
        DIS = QQ + XX - 2 * QX
        # DIS = torch.sqrt(DIS.squeeze(dim=1))
        DIS = DIS.squeeze(dim=1)
        # DIS[DIS == float('nan')] = float('inf')
        return DIS


class ShapeletMatrixModel():
    def __init__(self):
        pass
    
    def train(self, X, labels, m_len):
        # X shape batchsize x D x T 或者是 list T x D
        if isinstance(X, list):
            samples = [torch.from_numpy(x).float() for x in X]
            lengths = [len(x) for x in X]
            catsamples = torch.cat(samples, dim=0)
            N, T = len(lengths), max(lengths)
        else:
            N, D, T = X.shape
            catsamples = torch.reshape(X.permute(0, 2, 1), (-1, D))
            lengths = [T] * N
        
        bestscore = 0
        bestloss = float('inf')
        cumlength = np.cumsum([0] + lengths)
        shrink = - m_len + 1
        offset = [x + shrink for x in lengths]

        # 为了避免每次都要重新构建数据库，所以这里提前按照最大的sample构建距离矩阵
        DISMAT_pre = torch.zeros(T-m_len+1, cumlength[-1]+shrink, dtype=torch.float32)

        # 同样也会准备存放处理结果数据的
        MinDis = torch.zeros(N, T-m_len+1, dtype=torch.float32)
        MinLoc = torch.zeros(N, T-m_len+1, dtype=torch.int16)

        if torch.cuda.is_available():
            catsamples = catsamples.cuda()
            DISMAT_pre = DISMAT_pre.cuda()
            MinDis = MinDis.cuda()
            MinLoc = MinLoc.cuda()

        Dis_sample = np.zeros((N, T-m_len+1))
        Dis_loc = np.zeros(Dis_sample.shape, dtype=np.int16)
        dis = np.zeros((N,))
        locs = np.zeros((N,), dtype=np.int16)
        for i in range(N):
            if labels[i] == 0:
                continue

            # time_0 = time.time()
            begin, end = cumlength[i:i+2]
            # end = T * (i+1)
            with torch.no_grad():
                DISMAT_pre[:offset[i]] = utilmx.matrixprofile_torch(
                    catsamples[begin:end], 
                    catsamples, 
                    m_len, 
                    DISMAT_pre[:offset[i]])

                # time_1 = time.time()

                for j in range(N):
                    b_loc = cumlength[j]
                    e_loc = cumlength[j+1] + shrink
                    tempdis = DISMAT_pre[:offset[i], b_loc:e_loc]
                    MinDis[j, :offset[i]], MinLoc[j, :offset[i]] = torch.min(tempdis, dim=-1)

                Dis_sample[:] = MinDis.cpu().numpy()
                Dis_loc[:] = MinLoc.cpu().numpy()

            for candin_index in range(lengths[i]-m_len+1):
                
                score = self.Bipartition_score(Dis_sample[:, candin_index], labels.numpy(), bestscore)
                loss = np.mean(Dis_sample[:, candin_index][labels == 1])

                if score > bestscore:
                    bestscore = score
                    shapeindex = i
                    bestloss = loss
                    dis[:] = Dis_sample[:, candin_index]
                    locs[:] = Dis_loc[:, candin_index]
                elif score == bestscore and loss < bestloss:
                    shapeindex = i
                    bestloss = loss
                    dis[:] = Dis_sample[:, candin_index]
                    locs[:] = Dis_loc[:, candin_index]
            # time_2 = time.time()
            # print('%f----%f' % (time_1-time_0, time_2-time_1))
            # print('%d/%d--->loss: %f, accuracy: %f' % (i, N, bestloss, bestscore))
        self.shapeindex = shapeindex
        self.locs = locs
        self.dis = dis
        self.score = bestscore
    
    def Bipartition_score(self, distances, labels, bestscore):
        '''
        description: 针对一个 distances 的 二分类的最大分类精度
        param: 
            pos_num: 其中 distance 的前 pos_num 个 的标签为 1, 其余都为 0
        return: 最高的分类精度, 以及对应的分割位置
        author: mario
        '''     
        dis_sort_index = np.argsort(distances)
        correct = len(distances) - labels.sum()
        Bound_correct = len(distances)
        maxcorrect = correct

        for i, index in enumerate(dis_sort_index):
            if labels[index] == 1:  # 分对的
                correct += 1
                if correct > maxcorrect:
                    maxcorrect = correct
            else:
                correct -= 1
                Bound_correct -= 1
            if correct == Bound_correct:
                break
            if (Bound_correct/len(distances)) < bestscore:
                break
        
        score = maxcorrect/len(distances)

        return score


# 在这种情况下，手语数据可能要分为左右手的形式
class ShapeletMatrixModel_LR():
    def __init__(self, **args):
        # x-mirror, y-mirror 分别是对奇数和偶数特征维度乘以 -1
        self.datamodeoptions = ['normal', 'x-mirror', 'y-mirror']
        self.datamode = self.datamodeoptions[0]
        # 表示在得到距离之后，二分类的使用方法
        self.clsoptions = ['fastbipa', 'linear', 'svm']
        self.clsmethod = self.clsoptions[0]
        # 表示dis 特征的类型
        self.disnumoptions = ['global', 'trunk', 'joints']
        self.disnum = self.disnumoptions

        self.alloptionnames = {'datamode': [self.datamodeoptions, self.datamode], 
                               'clsoptions': [self.clsoptions, self.clsmethod],
                               'disnum': [self.disnumoptions, self.disnum]}
        # 对输入的参数进行解析
        self.argsparse(args)
    
    def argsparse(self, argdict):
        for key in argdict.keys():
            if key not in self.alloptionnames.keys():
                raise ValueError('the input name %s is not correct' % key)
            
            value = argdict[key]
            if isinstance(value, int):
                if value < 0 or value >= len(self.alloptionnames[key][0]):
                    self.alloptionnames[key][1] = self.alloptionnames[key][0][value]
                else:
                    raise IndexError('the input value %s %d is out of index' % (key, value))
            
            elif value in self.alloptionnames[key][0]:
                self.alloptionnames[key][1] = value
            
            else:
                raise ValueError
    
    def preparedata(self, X):
        # X shape batchsize x D x T 或者是 list T x D
        if isinstance(X, list):
            # 默认list下，数据的类型是 numpy.arrary
            samples = [torch.from_numpy(x).float() for x in X]
            lengths = [len(x) for x in X]
            catsamples = torch.cat(samples, dim=0)
            N, T = len(lengths), max(lengths)
        else:
            # 如果已经是batch的格式的话，则认为已经是 tensor 的类型了
            N, D, T = X.shape
            catsamples = torch.reshape(X.permute(0, 2, 1), (-1, D))
            lengths = [T] * N
        # 现在数据catsamples的格式是为 NT x D

        # 下面将根据datamode对数据进行下一步的处理
        if self.datamode in ['x-mirror', 'y-mirror']:
            copysamples = torch.clone(catsamples)
            for i in range(D):
                if i % 2 == 0 and self.datamode == 'x-mirror':
                    copysamples[:, i] *= -1
                elif i % 2 == 1 and self.datamode == 'y-mirror':
                    copysamples[:, i] *= -1

            catsamples = torch.cat((catsamples, copysamples), dim=0)
            lengths = lengths * 2
        
        return catsamples, lengths
    
    def train(self, X, labels, m_len, **options):
        # 首先要对模型的参数进行配置
        self.argsparse(options)
        catsamples, lengths = self.preparedata(X)
        N, T = len(lengths), max(lengths)
        
        bestscore, miniloss = 0, float('inf')
        cumlength = np.cumsum([0] + lengths)
        shrink = - m_len + 1
        validlengths = [x + shrink for x in lengths]

        # 为了避免每次都要重新构建数据库，所以这里提前按照最大的sample构建距离矩阵, 存储每个sample与整体的 matrixprofile
        DISMAT_pre = torch.zeros(T+shrink, cumlength[-1]+shrink, dtype=torch.float32)

        # 同样也会准备存放处理结果数据的矩阵
        MinDis = torch.zeros(N, T+shrink, dtype=torch.float32)
        MinLoc = torch.zeros(N, T+shrink, dtype=torch.int16)

        if torch.cuda.is_available():
            catsamples = catsamples.cuda()
            DISMAT_pre = DISMAT_pre.cuda()
            MinDis = MinDis.cuda()
            MinLoc = MinLoc.cuda()

        Dis_sample = np.zeros((N, T-m_len+1))
        Dis_loc = np.zeros(Dis_sample.shape, dtype=np.int16)
        dis = np.zeros((N,))
        locs = np.zeros((N,), dtype=np.int16)
        for i in range(len(labels)):
            if labels[i] == 0:
                continue

            # time_0 = time.time()
            begin, end = cumlength[i:i+2]
            # end = T * (i+1)
            with torch.no_grad():
                DISMAT_pre[:validlengths[i]] = utilmx.matrixprofile_torch(
                    catsamples[begin:end], 
                    catsamples, 
                    m_len, 
                    DISMAT_pre[:validlengths[i]])

                # time_1 = time.time()

                for j in range(N):
                    b_loc = cumlength[j]
                    e_loc = cumlength[j+1] + shrink
                    tempdis = DISMAT_pre[:validlengths[i], b_loc:e_loc]
                    MinDis[j, :validlengths[i]], MinLoc[j, :validlengths[i]] = torch.min(tempdis, dim=-1)

                Dis_sample[:] = MinDis.cpu().numpy()
                Dis_loc[:] = MinLoc.cpu().numpy()

            for candin_index in range(lengths[i]-m_len+1):
                
                score = self.Bipartition_score(Dis_sample[:, candin_index], labels.numpy(), bestscore)
                loss = np.mean(Dis_sample[:, candin_index][labels == 1])

                if score > bestscore:
                    bestscore = score
                    shapeindex = i
                    bestloss = loss
                    dis[:] = Dis_sample[:, candin_index]
                    locs[:] = Dis_loc[:, candin_index]
                elif score == bestscore and loss < bestloss:
                    shapeindex = i
                    bestloss = loss
                    dis[:] = Dis_sample[:, candin_index]
                    locs[:] = Dis_loc[:, candin_index]
            # time_2 = time.time()
            # print('%f----%f' % (time_1-time_0, time_2-time_1))
            # print('%d/%d--->loss: %f, accuracy: %f' % (i, N, bestloss, bestscore))
        self.shapeindex = shapeindex
        self.locs = locs
        self.dis = dis
        self.score = bestscore
    
    def Bipartition_score(self, distances, labels, bestscore):
        '''
        description: 针对一个 distances 的 二分类的最大分类精度
        param: 
            pos_num: 其中 distance 的前 pos_num 个 的标签为 1, 其余都为 0
        return: 最高的分类精度, 以及对应的分割位置
        author: mario
        '''     
        dis_sort_index = np.argsort(distances)
        correct = len(distances) - labels.sum()
        Bound_correct = len(distances)
        maxcorrect = correct

        for i, index in enumerate(dis_sort_index):
            if labels[index] == 1:  # 分对的
                correct += 1
                if correct > maxcorrect:
                    maxcorrect = correct
            else:
                correct -= 1
                Bound_correct -= 1
            if correct == Bound_correct:
                break
            if (Bound_correct/len(distances)) < bestscore:
                break
        
        score = maxcorrect/len(distances)

        return score