import getopt
import sys
import argparse
from colorama import Fore

from models.gsgan.Gsgan import Gsgan
from models.leakgan.Leakgan import Leakgan
from models.maligan_basic.Maligan import Maligan
from models.mle.Mle import Mle
from models.rankgan.Rankgan import Rankgan
from models.seqgan.Seqgan import Seqgan
from models.textGan_MMD.Textgan import TextganMmd
from models.att_leak.Leakgan import Att_leakgan
import os

def set_gan(gan_name,pre=80,dis=100,ad=1,re=1,step=100,dis_dim=64):
    gans = dict()
    gans['seqgan'] = Seqgan
    gans['gsgan'] = Gsgan
    gans['textgan'] = TextganMmd
    gans['leakgan'] = Leakgan
    gans['rankgan'] = Rankgan
    gans['maligan'] = Maligan
    gans['mle'] = Mle
    gans['att_leak'] = Att_leakgan
    try:
        Gan = gans[gan_name.lower()]
        gan = Gan(pre,dis,ad,re,step,dis_dim)
        gan.vocab_size = 5000
        gan.generate_num = 10000
        return gan
    except KeyError:
        print(Fore.RED + 'Unsupported GAN type: ' + gan_name + Fore.RESET)
        sys.exit(-2)



def set_training(gan, training_method):
    try:
        if training_method == 'oracle':
            gan_func = gan.train_oracle
        elif training_method == 'cfg':
            gan_func = gan.train_cfg
        elif training_method == 'real':
            gan_func = gan.train_real
        else:
            print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
            sys.exit(-3)
    except AttributeError:
        print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
        sys.exit(-3)
    return gan_func


def parse_cmd():
    try:
        # pre=80
        # dis=100
        # ad=100
        # re=0.3
        parser = argparse.ArgumentParser()
        # parser.add_argument('--help', help='path to dataset')
        parser.add_argument('-g', type=str, default='seqgan', help='type of gan')
        parser.add_argument('-t', type=str, default='real',help='data type')
        parser.add_argument('-d', type=str, default='data/image_coco.txt',help='location of data')
        parser.add_argument('-pre', type=int, default=80,help='pretrain epoch')
        parser.add_argument('-dis', type=int, default=100,help='discriminator epoch')
        parser.add_argument('-ad', type=int, default=1,help='adversial training epoch')
        parser.add_argument('-re', type=float, default=1,help='reward coefficient')
        parser.add_argument('-step', type=int, default=100,help='step coefficient')
        parser.add_argument('-dis_dim', type=int, default=64,help='disdim coefficient')

        # opts, args = getopt.getopt(argv, "hg:t:d:")
        args = parser.parse_args()
        # # opt_arg = dict(opts)
        # if '-pre' in opt_arg.keys():
        #     pre=opt_arg['-pre']
        # if '-dis' in opt_arg.keys():
        #     dis=opt_arg['-dis']
        # if '-ad' in opt_arg.keys():
        #     ad=opt_arg['-ad']
        # if '-re' in opt_arg.keys():
        #     re=opt_arg['-re']

        # if '-h' in opt_arg.keys():
        #     print('usage: python main.py -g <gan_type>')
        #     print('       python main.py -g <gan_type> -t <train_type>')
        #     print('       python main.py -g <gan_type> -t realdata -d <your_data_location>')
        #     sys.exit(0)

        # if not '-g' in opt_arg.keys():
        #     print('unspecified GAN type, use MLE training only...')
        #     gan = set_gan('mle')
        # else:
        # print((args.pre,args.dis,args.ad,args.re))
        gan = set_gan(args.g, args.pre, args.dis, args.ad, args.re, args.step, args.dis_dim)

        # if not '-t' in opt_arg.keys():
        #     gan.train_oracle()
        # else:
        gan_func = set_training(gan, args.t)
        gan_func(args.d)
        # if opt_arg['-t'] == 'real' and '-d' in opt_arg.keys():
        #     gan_func(opt_arg['-d'])
        # else:
        #     gan_func()
    except getopt.GetoptError:
        print('invalid arguments!')
        print('`python main.py -h`  for help')
        sys.exit(-1)
    pass


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gan = None
    parse_cmd()
    # parse_cmd(sys.argv[1:])
