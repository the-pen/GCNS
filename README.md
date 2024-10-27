# MIX_GCN

##  一，模型介绍

模型基于仓库：ICMEW2024-Track10 仓库地址：https://github.com/liujf69/ICMEW2024-Track10

​                           TE-[GCN] 仓库地址：https://github.com/xieyulai/TE-GCN

## 二，数据介绍与处理

​    我们采用joint，bone，joint_motion，bone_motion模态进行训练和测试

（1），把赛题数据放到预处理文件"Process_data"中的"data"目录下；

└───data
        ├───train_joint.npy
        ├───train_label.npy

​        ├───test_label_A.npy

​        ├───test_joint_A.npy

​        └───...

  （2），使用赛题数据train_joint.npy通过运行官方脚本ICMEW2024-Track10-main/Process_data/gen_modal.py生成train_bone.npy，train_joint_motion.npy，train_bone_motion.npy，同时参考仓库https://github.com/happylinze/UAV-SAR，生成angel模态数据train_angel.npy与test_angel_A.npy

```
python gen_modal.py --modal bone --use_mp True

python gen_modal.py --modal jmb --use_mp True

python gen_modal.py --modal motion
###
python gen_angle_data.py
```

（3）从训练数据集中提取2d姿势，运行以下代码

```
cd Process_data
python extract_2dpose.py
```

运行此代码后，我们将在**Process_data/save_2d_pose**文件夹中生成名为V2.npz的文件。

## 三，复现过程

####   1，导入数据集

HDBN：将save_2d_pose文件夹移动到ICMEW2024-Track10-main/Model_inference/Mix_GCN/dataset与ICMEW2024-Track10-main/Model_inference/Mix_Former/dataset目录下

TE-GCN:将train_joint.npy，train_label.npy，test_joint_A.npy，test_label_A.npy移动到TE-GCN-main\data目录下

####   2，Mix_GCN

```
cd ./Model_inference/Mix_GCN
###
python main.py --config ./config/ctrgcn_V2_J.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn_V2_B.yaml --phase train --save-score True --device 0
###
python main.py --config ./config/ctrgcn_V2_J_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn_V2_B_3d.yaml --phase train --save-score True --device 0
###
python main.py --config ./config/tdgcn_V2_J.yaml --phase train --save-score True --device 0
python main.py --config ./config/tdgcn_V2_B.yaml --phase train --save-score True --device 0
###
python main.py --config ./config/tdgcn_V2_J_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/tdgcn_V2_B_3d.yaml --phase train --save-score True --device 0
###
python main.py --config ./config/tdgcn_V2_A.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn_V2_A.yaml --phase train --save-score True --device 0
```

#### 3，Mix_Former

```
cd ./Model_inference/Mix_Former
###
python main.py --config ./config/mixformer_V2_J.yaml --phase train --save-score True --device 0 
python main.py --config ./config/mixformer_V2_B.yaml --phase train --save-score True --device 0 
python main.py --config ./config/mixformer_V2_k2.yaml --phase train --save-score True --device 0 
python main.py --config ./config/mixformer_V2_k2M.yaml --phase train --save-score True --device 0 
```

#### 4，TE-GCN

```
python main.py --config ./config/uav-cross-subjectv2/train.yaml --work-dir work_dir/2101 -model_saved_name runs/2101 --device 0 --batch-size 56 --test-batch-size 56 --warm_up_epoch 5 --only_train_epoch 60 --seed 777
```

## 四，

将生成的pkl置信度文件放到Ensemble文件夹下，运行ensemble.py，生成pred.npy文件# GCNS
