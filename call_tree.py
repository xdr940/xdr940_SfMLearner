"""prepare_train_data.py"""
#1. 对数据预处理

main
	|
	|--KittiRawLoader#kitti_raw_loader.py
	|--dunp_example

"""kitti_raw_loader.py"""

KittiRawLoader#主要是读文件的，这里函数比较多先略过，大部分是属于file_read_utils
	|--
	|


"""train.py"""
#2.
main
	|
	|--SequenceFolder
	|--PoseExpNet
	|--DispNet
	|--train
	|--save_checkpoint

train# for one epoch
	|--photometric_reconstruction_loss
	|	|
	|	|--inverse_warp(ref_img,depth,intrinsic,pose)
	|
	|--explainability_loss
	|--smooth_loss