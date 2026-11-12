import argparse
def load_args():
    parser = argparse.ArgumentParser()

    # Pre training
    parser.add_argument('--dataset', type=str, default='HK')
    parser.add_argument('--train_num', type=int, default=10)
    parser.add_argument('--windowsize', type=int, default=11) #augsburg--7,hk--11,yellow--13,houston-11
    parser.add_argument('--type', type=str, default='none')
    parser.add_argument('--Xband', type=int,default=4)
    # Houston 1 YELLOW 3 Augsburg 4 hk-4

    # training parameter
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=320)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--lr_decay', type=float, default=0.5)

    # model parameter
    parser.add_argument('--spe_windowsize', type=int, default=3)
    parser.add_argument('--spa_patch_size', type=int, default=3)
    parser.add_argument('--spe_patch_size', type=int, default=4)
    parser.add_argument('--hid_chans', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--depth', type=int, default=2)

    parser.add_argument('--use_bi', default=True, type=lambda x: (str(x).lower() == 'true'), help='use bidirection or not' )  
    parser.add_argument('--use_global', default=True, type=bool,
                        help='use token meaning or not') 
    parser.add_argument('--use_cls', default=True, type=bool,
                        help='use class tken or not') 
    parser.add_argument('--use_fu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='use center augmentation fusion or not') 
      
    args = parser.parse_args()
    return args

