'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv

from densenets import DenseNet70

import mnist_fashion_reader
from utils import progress_bar, mixup_data,mixup_data_test, mixup_criterion
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='mixup_default', type=str, help='session id')
parser.add_argument('--seed', default=0, type=int, help='rng seed')
parser.add_argument('--alpha', default=1., type=float, help='interpolation strength (uniform=1., ERM=0.)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay (default=1e-4)')
args = parser.parse_args()

torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 128
base_learning_rate = 0.1
if use_cuda:
    # data parallel
    n_gpu = torch.cuda.device_count()
    print(n_gpu)
    batch_size *= n_gpu
    base_learning_rate *= n_gpu

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

c_trainset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=True, download=True, transform=transform_train)
c_train = torch.utils.data.DataLoader(c_trainset, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)

c_testset = torchvision.datasets.CIFAR10(root='../../data/cifar10', train=False, download=True, transform=transform_test)
c_test = torch.utils.data.DataLoader(c_testset, batch_size=100, shuffle=False, num_workers=1,drop_last=True)

C_trainset = torchvision.datasets.CIFAR100(root='../../data/cifar100', train=True, download=True, transform=transform_train)
C_train = torch.utils.data.DataLoader(C_trainset, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)

C_testset = torchvision.datasets.CIFAR100(root='../../data/cifar100', train=False, download=True, transform=transform_test)
C_test = torch.utils.data.DataLoader(C_testset, batch_size=100, shuffle=False, num_workers=1,drop_last=True)


m_transform_train = transform=transforms.Compose([
                      transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(), 
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))
                   ])

m_transform_test = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

fashion_train = mnist_fashion_reader.MNIST_FISHION(root='../../data/fashion', train=True, download=False, transform=m_transform_train)
fashion_test = mnist_fashion_reader.MNIST_FISHION(root='../../data/fashion', train=False, download=False, transform=m_transform_test)

f_train = torch.utils.data.DataLoader(fashion_train, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
f_test = torch.utils.data.DataLoader(fashion_test, batch_size=100, shuffle=False, num_workers=1,drop_last=True)

mnist_train = torchvision.datasets.MNIST('../../data/mnist', train=True, download=True,transform=m_transform_train)
mnist_test = torchvision.datasets.MNIST('../../data/mnist', train=False, transform=m_transform_test)

m_train = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=1,drop_last=True)
m_test = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False, num_workers=1,drop_last=True)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7.' + args.sess + '_' + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
else:
    print('==> Building model..')
    net = DenseNet70()
    # net = PreActResNet18()

result_folder = './results/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

logname = result_folder + net.__class__.__name__ + '_' + args.sess + '_' + str(args.seed) + '.csv'

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=base_learning_rate, momentum=0.9, weight_decay=args.decay)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    p1=0
    p2=0
    p3=0
    p4=0
    correct = 0
    total = 0
    for batch_idx, ((m_in, m_tar),(f_in, f_tar),(c_in,c_tar),(C_in,C_tar)) in enumerate(zip(m_train,f_train,c_train,C_train)):
        # generate mixed inputs, two one-hot label vectors and mixing coefficient
        lam=0
        inputs = torch.cat((m_in,f_in,c_in,C_in),1)
        if use_cuda:
            inputs, targets_a, targets_b, targets_c, targets_d = inputs.cuda(), m_tar.type(torch.LongTensor).cuda(),f_tar.type(torch.LongTensor).cuda(), c_tar.type(torch.LongTensor).cuda(), C_tar.type(torch.LongTensor).cuda()
        optimizer.zero_grad()
        inputs, targets_a, targets_b, targets_c, targets_d = Variable(inputs), Variable(targets_a), Variable(targets_b), Variable(targets_c), Variable(targets_d)
        outputs,outputs2,outputs3, outputs4 = net(inputs, targets_a, targets_b, targets_c, targets_d)

        #rand = np.random.uniform(1, 1)
        loss_func = mixup_criterion(targets_a, targets_b,targets_c, targets_d, lam)
        loss = loss_func(criterion, outputs,outputs2,outputs3,outputs4)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        _, predicted2 = torch.max(outputs2.data, 1)
        _, predicted3 = torch.max(outputs3.data, 1)
        _, predicted4 = torch.max(outputs4.data, 1)
        total += targets_a.size(0)
        # --------------
        prec1=predicted.eq(targets_a.data).cpu().sum()
        prec2=predicted2.eq(targets_b.data).cpu().sum()
        prec3=predicted3.eq(targets_c.data).cpu().sum()
        prec4=predicted4.eq(targets_d.data).cpu().sum()
        p1+=prec1
        p2+=prec2
        p3+=prec3
        p4+=prec4        

        
        correct +=  (prec1 +  prec2 + prec3 + prec4)/3
        progress_bar(batch_idx, len(m_train), 'Loss: %.3f |P1: %.3f |P2: %.3f |P3: %.3f| P4: %.3f| Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1),100.*p1/total,100.*p2/total,100.*p3/total,100.*p4/total,100.*correct/total, correct, total))
    return (train_loss/batch_idx, 100.*correct/total,100.*p1/total,100.*p2/total,100.*p3/total,100.*p4/total,lam)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    correct2 = 0
    correct3 = 0
#    correct4 = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(c_test):
        #inputs = mixup_data_test(inputs)
        inputs=torch.cat((inputs[:,0:2,:,:],inputs,inputs),1)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.type(torch.LongTensor).cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs,outputs2,outputs3,outputs4 = net(inputs)
        loss = criterion(outputs, targets)
        #loss2 = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        _, predicted2 = torch.max(outputs2.data, 1)
        correct2 += predicted2.eq(targets.data).cpu().sum()

        _, predicted3 = torch.max(outputs3.data, 1)
        correct3 += predicted3.eq(targets.data).cpu().sum()
#        _, predicted4 = torch.max(outputs4.data, 1)
#        correct4 += predicted4.eq(targets.data).cpu().sum()
        
             

        progress_bar(batch_idx, len(c_test), 'Loss: %.3f |Acc1: %.3f |Acc2: %.3f | Acc3: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total,100.*correct2/total,100.*correct3/total,correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        checkpoint(acc, epoch)
    return (test_loss/batch_idx, 100.*correct/total,100.*correct2/total,100.*correct3/total)

def test2(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    total = 0
    for batch_idx, ((m_in, m_tar),(f_in,f_tar),(c_in,c_tar),(C_in,C_tar)) in enumerate(zip(m_test,f_test,c_test,C_test)):

        inputs = torch.cat((m_in,f_in,c_in,C_in),1)
        if use_cuda:
            inputs, targets_a, targets_b,targets_c, targets_d = inputs.cuda(), m_tar.cuda(),f_tar.cuda(), c_tar.cuda(),C_tar.cuda()
        inputs, targets_a, targets_b,targets_c,targets_d = Variable(inputs, volatile=True), Variable(targets_a),Variable(targets_b),Variable(targets_c),Variable(targets_d)
        outputs,outputs2,outputs3,outputs4 = net(inputs)
        loss = criterion(outputs, targets_a)
        
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        _, predicted2 = torch.max(outputs2.data, 1)
        _, predicted3 = torch.max(outputs3.data, 1)
        _, predicted4 = torch.max(outputs4.data, 1)
        total += targets_a.size(0)
        
        correct += predicted.eq(targets_a.data).cpu().sum()
        correct2 += predicted2.eq(targets_b.data).cpu().sum()
        correct3 += predicted3.eq(targets_c.data).cpu().sum()
        correct4 += predicted4.eq(targets_d.data).cpu().sum()
        # --------------
        
        progress_bar(batch_idx, len(m_test), 'Loss: %.3f |Acc_a: %.3f | Acc_b: %.3f|Acc_c: %.3f|  Acc_d: %.3f%% (%d)'
            % (test_loss/(batch_idx+1),100.*correct/total,100.*correct2/total, 100.*correct3/total, 100.*correct4/total,  total))

    # Save checkpoint.
#    acc = 100.*correct/total
#    if acc > best_acc:
#        best_acc = acc
#        checkpoint(acc, epoch)
    return (test_loss/batch_idx, 100.*correct/total,100.*correct2/total,100.*correct3/total, 100.*correct4/total)

def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7.' + args.sess + '_' + str(args.seed))

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = base_learning_rate
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (base_learning_rate - 0.1) * epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss','lam','prec1','prec2', 'prec3','train acc', 
        'test loss', 'test acc_1','test acc_2','test acc_3', 'test loss2','test acc1','test acc2','test acc3'])

for epoch in range(start_epoch, 200):
    adjust_learning_rate(optimizer, epoch)
    train_loss, train_acc,prec1,prec2,prec3,prec4,lam = train(epoch)
    test_loss, test_acc_1,test_acc_2,test_acc_3 = test(epoch)
    test_loss2, t_acc1,t_acc2,t_acc3,t_acc4 = test2(epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss,lam,prec1,prec2,prec3,prec4,train_acc,
                            test_loss, test_acc_1,test_acc_2,test_acc_3,test_loss2,t_acc1,t_acc2,t_acc3,t_acc4])
print('best acc:',best_acc)
