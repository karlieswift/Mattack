#!/bin/bash
#SBATCH --job-name=pytorch #任务名称
#SBATCH --nodes=1 #节点数，根据计算需要灵活配置
#SBATCH --ntasks-per-node=4 #每个节点分配进程（任务）数量，根据计算需要灵活配置
#SBATCH --ntasks=4 #任务（进程）数量，根据计算需要灵活配置
#SBATCH --mem=80G
#SBATCH --partition=normal #队列名称，需根据所用账号所属队列的权限填写
#SBATCH --gres=dcu:4 #单计算节点需要 DCU 加速卡张数
##软件环境变量设置
#module load apps/PyTorch/1.6.0a0/hpcx-2.4.1-gcc-7.3.1-rocm3.3 #加载 PyTorch 环境变量
##赋予 Python 测试脚本可执行权限
##通过 PyTorch 环境或通过 anaconda 安装虚拟 python 环境并激活，查找 python3 解释器/命令


#python adv_vitgpt2.py 
python  -m torch.distributed.launch --nproc_per_node=4 ddp_train_coco.py 
#python3 ./train_vit_gpt2_coco.py 
#python3 ./train_vit_gpt2_flicker.py 
#python3 ./train_vit_gpt2_pokemon.py 
#bash finetune/finetune_visualglm_qlora.sh
