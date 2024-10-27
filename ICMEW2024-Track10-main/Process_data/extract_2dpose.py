import numpy as np

x_train = np.load('data/train_joint.npy',mmap_mode='r')
x_test = np.load('data/test_joint_A.npy',mmap_mode='r')

y_train = np.load('data/train_label.npy')
y_test = np.load('data/test_label_A.npy')


np.savez('./save_2d_pose/V2.npz', x_train = x_train[:,[0,2],:,:,:].transpose(0, 2, 4, 3, 1), y_train = y_train, x_test = x_test[:,[0,2],:,:,:].transpose(0, 2, 4, 3, 1), y_test = y_test)