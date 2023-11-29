import argparse
import datetime as dt
import platform

from utils.common import show_lg

# 
def parse_opt(known=False):
    parser = argparse.ArgumentParser(prog='PROG')
    # train-args and Shared parameters
    parser.add_argument('--resume', type=str, default='', help="预训练模型")
    # parser.add_argument("--net",type=str,required=True,help='选择训练的网络')
    parser.add_argument('--input_size', type=int, nargs='+', default=[224,224], help="输入图像大小")
    parser.add_argument('--epochs', type=int, default=100, help="训练epochs")
    parser.add_argument('--batch_size', type=int, default=64, help='batch-size大小')
    parser.add_argument('--n_cuda', type=str, default=0, help='gpu id')  #!^
    parser.add_argument('--data2index', type=int, nargs='+', default=[-2], help='数据源对应的label')  #!^ 
    parser.add_argument('--n_cls', type=int, default=3, help='mulit-heads: num_classes')  #!^
    parser.add_argument('--worker', type=int, default=8, help='训练模型', choices=[0,1,2,3])

    parser.add_argument('--init_method', type=int,default=0, help='learning ratio')
    parser.add_argument('--workers', type=int,default=4, help='worker core')   #!^ 
    parser.add_argument('--lr0', type=float,default=1e-3, help='learning ratio')
    parser.add_argument('--lr1', type=float,default=1e-3, help='learning ratio') 
    parser.add_argument('--momentum', type=float,default=0.937, help='momentum')
    parser.add_argument('--weight_decay', type=float,default=0.0005, help='weight_decay')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    

    parser.add_argument('--data_path', type=str, default='pth_1', required=True, help='1-head 数据路径')
    parser.add_argument('--resume_weights', type=str, default='', help='1-head 保存的模型')
    parser.add_argument('--log_name', type=str, default='train/sh', help='保存训练日志文件夹')
  
    # default
    # parser.add_argument('--flag', type=int, default=0, help='训练模型', choices=[0, 1, 2, 3])
    opts = parser.parse_known_args()[0] if known else parser.parse_args()

    opts.mode_Lst = ['train', 'val']
    # opts.netIndx = opts.net.split("_")[1]    # [0, 1, 2]  small middle large
    # opts.net = int(opts.net.split("_")[0])
    opts.early_lst = {"acc":0.99,'loss':0.002} # '早停阈值'
    opts.log_pth = opts.log_name + f"_{dt.datetime.now().strftime('%Y_%m_%d-%H%M%S')}"
    if 'win' in platform.platform().lower():
        show_lg('using window and worker be 0')
        opts.workers=0
    return opts

if __name__ == '__main__':
    print(parse_opt())
