# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:51:31 2019

@author: tis05
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:26:39 2018

@author: Brian.Chiu
"""

#load .mat data __ data preprocess

import time, cv2, os
from scipy import io
import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
import mxnet.ndarray as nd
from mxnet.gluon import nn 
from mxnet import gluon
from mxnet import autograd
from mxnet import init
from mxnet import image
from mxnet.gluon.model_zoo import vision as models
import utils


def vgg16_net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(channels=64, kernel_size=3, activation='relu'),
                nn.Conv2D(channels=64, kernel_size=3, activation='relu'),
                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=128, kernel_size=3, activation='relu'),
                nn.Conv2D(channels=128, kernel_size=3, activation='relu'),
                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=256, kernel_size=3, activation='relu'),
                nn.Conv2D(channels=256, kernel_size=3, activation='relu'),
                nn.Conv2D(channels=256, kernel_size=3, activation='relu'),
#                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=512, kernel_size=3, activation='relu'),
                nn.Conv2D(channels=512, kernel_size=3, activation='relu'),
                nn.Conv2D(channels=512, kernel_size=3, activation='relu'),
#                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=512, kernel_size=3, activation='relu'),
                nn.Conv2D(channels=512, kernel_size=3, activation='relu'),
                nn.Conv2D(channels=512, kernel_size=3, activation='relu'),
#                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Dense(4096, activation = 'relu'),
                nn.Dropout(.5),
                nn.Dense(4096, activation = 'relu'),
                nn.Dropout(.5),
                nn.Dense(2)
        )
    net.initialize()
    return net


def get_transform(augs):
    def transform(data, label):
        # data: sample x height x width x channel
        # label: sample
        data = data.astype('float32')
        if augs is not None:
            # apply to each sample one-by-one and then stack
            data = nd.stack(*[
                apply_aug_list(d, augs) for d in data])
        data = nd.transpose(data, (0,3,1,2))
        return data, label.astype('float32')
    return transform


def image_aug_transform(data,label):
    img_augs = [image.HorizontalFlipAug(1)]
    array_rate = 8
    data_array = np.zeros((data.shape[0]*array_rate,data.shape[1],data.shape[2]))
    label_array = np.ones(data_array.shape[0])*label
    data_array[0:data.shape[0],:,:] = data
    for i, d in enumerate(data):
        for j in img_augs:
            for k in range(1,array_rate):
                data_array[data.shape[0]*k+i,:,:] = j(nd.array(d)).asnumpy()
                
        
    
    return data_array, label_array

                
def data_class_process(pre_data,pre_label,data_num,train_augs,class_value = [0,3],augs_value = 1):
    #將Not sperm的歸類於Abnormal
    new_label = np.zeros((0,0))
    new_data = np.zeros((0,128,128))
    for i, v in enumerate(class_value):
        if i <len(class_value)-1:
            for j in range(class_value[i+1]-class_value[i]):
                pre_label[np.where(pre_label == v+j)] = i
            index = np.random.permutation(np.where(pre_label == i)[0])
            if i == augs_value:
                data_array, label_array = image_aug_transform(pre_data[index[0:data_num[i]]],augs_value)
                new_label = np.append(new_label,label_array)
                new_data = np.append(new_data,data_array,axis = 0)
            else:
                new_label = np.append(new_label,pre_label[index[0:data_num[i]]])
                new_data = np.append(new_data,pre_data[index[0:data_num[i]]],axis = 0)            
            
        else:
            pre_label[np.where(pre_label == v)] = i
            index = np.random.permutation(np.where(pre_label == i)[0])
            if i == augs_value:
                data_array, label_array = image_aug_transform(pre_data[index[0:data_num[i]]],augs_value)
                new_label = np.append(new_label,label_array)
                new_data = np.append(new_data,data_array,axis = 0)
            else:
                new_label = np.append(new_label,pre_label[index[0:data_num[i]]])
                new_data = np.append(new_data,pre_data[index[0:data_num[i]]],axis = 0)
    index = np.random.permutation(np.where(pre_label == 4)[0])
    new_label = np.append(new_label,np.zeros((data_num[2])))
    new_data = np.append(new_data,pre_data[index[0:data_num[2]]],axis = 0)
            
    return pre_data, pre_label,new_label, new_data


def get_data(file_path,rand_seed,batch_size,train_augs):
    #從 mat 檔獲得 data資訊
    sperm_data = io.loadmat(file_path)['cnn_data']
    pre_data = []
    pre_label = []
    for i in sperm_data:
        for j in i[0]:
            pre_data.append(np.float64(j[0]))
            pre_label.append(np.int32(j[1][0][0]))
#            tic = time.time()
#            pre_data = np.append(pre_data,nd.array(cv2.resize(j[0],(94,94))).expand_dims(axis=0).asnumpy(),axis = 0)
#            pre_label = np.append(pre_label,np.int32(j[1][0][0]))
#            print(time.time()-tic)
    pre_data = np.asarray([cv2.resize(pre_data,(128,128)) for pre_data in pre_data])
    pre_data = pre_data.astype(np.float32, copy=False)/(255.0/2) - 1.0
    pre_label = np.array(pre_label)
    
    class_value = [0,3]  #所需要流下的種類編號
    data_num = [1770,list(pre_label).count(3),414] #
    pre_data, pre_label,new_label, new_data = data_class_process(pre_data,pre_label,data_num,train_augs,class_value)
#    plt.imshow(pre_data[0])
    pre_data = nd.array(new_data,ctx = mx.gpu(0)).expand_dims(axis= 1)
    pre_data = nd.tile(pre_data,(1,3,1,1))
    
    pre_label = nd.array(new_label,mx.gpu(0))
    np.random.seed(rand_seed)
    rand_index = np.random.permutation(np.shape(pre_data)[0])
    
#    plt.figure(2)
#    plt.imshow(((pre_data[0][0]+1)*(255/2)).asnumpy())
    
    pre_train = [pre_data[rand_index[0:-np.int32(rand_index.shape[0]/4)]],pre_label[rand_index[0:-np.int32(rand_index.shape[0]/4)]]]
    pre_test = [pre_data[rand_index[-np.int32(rand_index.shape[0]/4):]],pre_label[rand_index[-np.int32(rand_index.shape[0]/4):]]]
    train_iter = mx.io.NDArrayIter(data = pre_train[0], label= pre_train[1], batch_size=batch_size)
    test_iter = mx.io.NDArrayIter(data = pre_test[0], label= pre_test[1], batch_size=batch_size)
    
    return train_iter, test_iter, data_num
    
def apply_aug_list(img, augs):
    for f in augs:
        img = f(img)
    return img

def evaluate_accuracy(data_iterator, net, ctx=[mx.gpu()]):
    acc = nd.array([0])
    n = 0.
    data_iterator.reset()
    for batch in data_iterator:
        data = batch.data
        label = batch.label
        for X, y in zip(data, label):
            y = y.astype('float32')
            acc += nd.sum(net(X).argmax(axis=1)==y).copyto(mx.cpu())
            n += y.size
        acc.wait_to_read() # don't push too many operators into backend
    return acc.asscalar() / n

def show_evaluate_value(data_iterator, net, data_num,code_path,result_path, ctx = mx.gpu(0)):
    data_iterator.reset()
    testv = data_iterator.label[0][1]
    if (testv.shape[0]%64) != 0:
        netv = nd.zeros(((testv.shape[0]//64)+1)*64,ctx = ctx)
    else:
        netv = nd.zeros(testv.shape[0],ctx = ctx)
    true_num = nd.zeros((2,),ctx = ctx)
    fail_test = []
    fail_net = []
    error_list = []
    for i, batch in enumerate(data_iterator):
        data = batch.data
        label = batch.label
        for X, y in zip(data, label):
            ty = net(X).argmax(axis = 1)
#            testv[i*64:i*64+64] = y
            netv[i*64:i*64+64] = ty
            for ind,n in enumerate(ty):
                if n == y[ind]:
                    true_num[n] += 1
                else:
                    fail_test.append([i,ind,y[ind].asnumpy()[0]])
                    fail_net.append([i,ind,n.asnumpy()[0]])
                    error_list.append(y[ind].asnumpy()[0]*10+n.asnumpy()[0])
                    plt.figure((i*64+ind)*10+np.int32(y[ind].asnumpy()[0]))
                    plt.imshow((X[ind].transpose((1,2,0))+125)[:,:,0].asnumpy(),cmap='gray')
                    os.chdir(result_path)
                    plt.savefig('image%d'% ((i*64+ind)*10+np.int32(y[ind].asnumpy()[0])))
                    os.chdir(code_path)
#            for i in range(len(ty))
#            print(netv)
#            print('No.%d net value ='%i,ty)
#            print('No.%d test value ='% i,y)
    print('Abnormal data num: %d , data rate: %.2f%%, Abnormal acc : %.2f%%' % (list(testv).count(0), list(testv).count(0)/(data_num[0]+data_num[2])*100, (true_num[0]/list(testv).count(0)*100).asnumpy()[0]))
    print('Normal data num: %d , data rate: %.2f%%, Normal acc : %.2f%%' % (list(testv).count(1),  list(testv).count(1)/(data_num[1]*8)*100, (true_num[1]/list(testv).count(1)*100).asnumpy()[0]))
    print('Error 0->1 : %d' % error_list.count(1))
    print('Error 1->0 : %d' % error_list.count(10))

    return testv, netv, true_num, fail_test, fail_net,error_list
            
            


train_augs = image.HorizontalFlipAug(1)

test_augs = [
    image.CenterCropAug((32,32))
]



batch_size = 64
learning_rate=.1
num_epochs = 50
randseed = 4
ctx = utils.try_all_gpus()

loss = gluon.loss.SoftmaxCrossEntropyLoss()
train_data, test_data, data_num = get_data('D:/lagBear/SEMEN/finally_data/cnn_data.mat',randseed,batch_size=batch_size,train_augs=train_augs)


pre_net = models.resnet18_v1(ctx = ctx,pretrained=True,prefix = 'sperm_3t2class_')
pre_net.output
pre_net.features[0].weight.data()[0][0]

net = models.resnet18_v1(classes=2,prefix = 'sperm_3t2class_',ctx = ctx)
net.features = pre_net.features
net.initialize(ctx=ctx)
net.output.initialize(init.Xavier())
net.hybridize()




trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': learning_rate})

print("Start training on ", ctx)
print_batches=None

if not os.path.isdir('spermdata'):
    os.mkdir('spermdata')
if not os.path.isdir('spermdata/new3t2/test1'):
    os.mkdir('spermdata/new3t2/test1')

code_path = os.getcwd()
result_path = 'D:\\code\\spermdata\\new3t2\\test1'
    
for epoch in range(num_epochs):
    train_loss, train_acc, n, m = 0.0, 0.0, 0.0, 0.0
    start = time.time()
    train_data.reset()
    for i, batch in enumerate(train_data):
        data = batch.data
        label = batch.label
        losses = []
        with autograd.record():
            outputs = [net(X) for X in data]
            losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
        for l in losses:
            l.backward()
        train_acc += sum([(yhat.argmax(axis=1)==y).sum().asscalar()
                          for yhat, y in zip(outputs, label)])  
        train_loss += sum([l.sum().asscalar() for l in losses])
        trainer.step(batch_size)
        n += batch_size
        m += sum([y.size for y in label])
        if print_batches and (i+1) % print_batches == 0:
            
            print("Batch %d. Loss: %f, Train acc %f" % (
                n, train_loss/n, train_acc/m
            ))

    test_acc = evaluate_accuracy(test_data, net, ctx)
    print("Epoch %d. Loss: %.3f, Train acc %.2f, Test acc %.2f, Time %.1f sec" % (
        epoch, train_loss/n, train_acc/m, test_acc, time.time() - start
    ))
    if train_loss/n <= 0.0008 and epoch >= 20:
        break
    
print('Sperm 3 class to 2 class predict \nTest 1, Rand seed %d, '% randseed)
tv,nv,tn,ft,fn,el = show_evaluate_value(test_data,net,data_num,code_path,result_path,ctx = ctx)
with open('Sperm_3to2class_result.txt','a+') as f :
    f.write('Test 10, Rand seed %d\n\n' % randseed)
    f.write('Sperm 3 class to 2 class predict, Rand seed %d, test acc %.2f%%\n'% (randseed, test_acc*100))
    f.write('Abnormal data num: %d , data rate: %.2f%%, Abnormal acc : %.2f%% \n' % (list(tv).count(0), list(tv).count(0)/(data_num[0]+data_num[2])*100, (tn[0]/list(tv).count(0)*100).asnumpy()[0]))
    f.write('Normal data num: %d , data rate: %.2f%%, Normal acc : %.2f%% \n\n\n' % (list(tv).count(1),  list(tv).count(1)/(data_num[1]*8)*100, (tn[1]/list(tv).count(1)*100).asnumpy()[0]))
        