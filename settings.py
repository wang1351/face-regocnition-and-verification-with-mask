import argparse
import cv2

#
#
def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str,
                        default='real2real_using_ResNet18',
                        help='Name of experiment')
    parser.add_argument("--log_out_dir", type=str,
                        default='logs',
                        help='log')
    parser.add_argument("--check_point_out_dir", type=str,
                        default='check_points',
                        help='checkout')
    parser.add_argument("--runargs_out_dir", type=str,
                        default='runargs',
                        help='runargs')
    parser.add_argument("--lr", type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument("--moment", type=float,
                        default=0.9,
                        help='momentum')
    parser.add_argument("--log_freq", type=int,
                        default=10,
                        help='logging frequency')
    parser.add_argument("--n_classes", type=int,
                        default=2,
                        help='number of classes')
    parser.add_argument("--batch_size", type=int,
                        default=128,
                        help='batch size')
    parser.add_argument("--num_epoch", type=int,
                        default=100,
                        help='number of training epochs')
    parser.add_argument("--lr_step1", type=int,
                        default=20,
                        help='when to decrease lr')
    parser.add_argument("--lr_gamma", type=float,
                        default=10,
                        help='the amount of decreas')
    return vars(parser.parse_args())
