import time
import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from config import load_args
from data_read import readdata
from util import tr_acc, test_batch, record_output
from model import SSFIN
import os
day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')
args = load_args()
import torch
import sys
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

num_of_ex = 10
dataset = args.dataset
windowsize = args.windowsize
type = args.type
train_num = args.train_num
lr = args.lr
epoch = args.epoch
batch_size = args.batch_size
drop_rate = args.drop_rate
gamma = args.lr_decay
spe_windowsize = args.spe_windowsize
spa_patch_size = args.spa_patch_size
spe_patch_size = args.spe_patch_size
embed_dim = args.embed_dim
hid_chans = args.hid_chans
depth = args.depth
use_bi = args.use_bi
use_global = args.use_global
use_cls = args.use_cls
halfsize = int((windowsize-1)/2)
import numpy as np

AC, OA, AA, KA, CM,  TRAINING_TIME, TESTING_TIME = [], [], [], [], [], [], []
for num in range(0, num_of_ex):
    print('num:', num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    train_image, train_label, validation_image, validation_label, nTrain_perClass, nvalid_perClass, index, image, gt, s, train_imageX, validation_imageX, imageX, test_index = readdata(
        args, num)
    nclass = np.max(gt)
    nclass = int(nclass)
    result = np.zeros([nclass + 3, num_of_ex])
    nband = train_image.shape[-1]
    xnband = args.Xband
    print(train_image.shape,train_imageX.shape)
    train_image = np.transpose(train_image,(0,3,1,2))
    train_imageX = np.transpose(train_imageX, (0, 3, 1, 2))
    validation_image = np.transpose(validation_image,(0,3,1,2))
    validation_imageX=np.transpose(validation_imageX,(0,3,1,2))
    nvalid_perClass = np.zeros_like(nvalid_perClass)
    model = SSFIN(spa_img_size=(windowsize, windowsize),spe_img_size=(spe_windowsize,spe_windowsize), spa_patch_size=spa_patch_size, spe_patch_size=spe_patch_size, in_chans=nband, xband=xnband ,hid_chans = hid_chans, embed_dim=embed_dim, drop_path=drop_rate, nclass=nclass, depth=depth, bi=use_bi, norm_layer=nn.LayerNorm, global_pool=use_global, cls = use_cls, fu = args.use_fu)


    optimizer = optim.Adam(model.parameters(),lr = lr, weight_decay = 1e-4)
    print('the number of training samples:', train_image.shape[0])

    train_dataset = TensorDataset(torch.tensor(train_image), torch.tensor(train_label), torch.tensor(train_imageX))
    train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True, drop_last=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [50, 110, 140], gamma = gamma, last_epoch=-1)

    # training
    best_checkpoint = {'epoch': 0, 'OA_TE': 5, 'AA_TE': 5, 'each_class': [0],
                       'Kappa_TE': 5, 'true_class': [0], 'cm': [0], 'test_pre': [0]}

    toc2 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ce_loss = torch.nn.CrossEntropyLoss()
    model.to(device)
    for i in range(epoch+1):
        model.train()
        train_loss = 0
        for idx, (label_x, label_y, label_xX) in enumerate(train_loader):
            label_x, label_y, label_xX = label_x.squeeze(1).to(device), label_y.to(device), label_xX.to(device)
            outputs = model(label_x, label_xX)
            loss = ce_loss(outputs, label_y.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()*label_x.shape[0]
        train_loss = train_loss/train_image.shape[0]
        scheduler.step()

        if (i+1) % 10 == 0:
            train_acc, train_loss = tr_acc(model.eval(), train_image, train_label, train_imageX)
            print('epoch:', i, 'loss:%.4f' % train_loss,'train_acc:%.4f'%train_acc.item())
    tic1 = time.time()
    true_cla, overall_accuracy, average_accuracy, kappa, cm, test_pred = test_batch(model.eval(), image, imageX,
                                                                                                index, 400,
                                                                                                nTrain_perClass,
                                                                                                nvalid_perClass, halfsize)
    best_checkpoint = {'epoch': i, 'OA_TE': overall_accuracy, 'AA_TE': average_accuracy, 'Kappa_TE': kappa,
                                       'each_class':true_cla, 'cm':cm, 'test_pre':test_pred}

    print("OA: {:.2f} | AA: {:.2f} | Kappa: {:.4f}".format(best_checkpoint['OA_TE'], best_checkpoint['AA_TE'],
                                                           best_checkpoint['Kappa_TE']))
    print(best_checkpoint['each_class'])
    tic2 = time.time()
    true_cla, overall_accuracy, average_accuracy, kappa, cm, test_pred = best_checkpoint['each_class'], best_checkpoint[
        'OA_TE'], best_checkpoint['AA_TE'], best_checkpoint['Kappa_TE'], best_checkpoint['cm'], best_checkpoint[
                                                                             'test_pre']
    result[:nclass, num] = true_cla
    result[nclass, num] = overall_accuracy
    result[nclass + 1, num] = average_accuracy
    result[nclass + 2, num] = kappa
    
    AC.append(true_cla)
    OA.append(overall_accuracy)
    AA.append(average_accuracy)
    KA.append(kappa)
    CM.append(cm)
    TRAINING_TIME.append(tic1 - toc2)
    TESTING_TIME.append(tic2-tic1)
    ELEMENT_ACC = np.array(AC)
os.makedirs('./record', exist_ok=True)
record_output(OA, AA, KA, ELEMENT_ACC, CM, TRAINING_TIME, TESTING_TIME, './record/' +'_'+ day_str + '_' +dataset+'_'+str(train_image.shape[0])+ '_end.txt')


