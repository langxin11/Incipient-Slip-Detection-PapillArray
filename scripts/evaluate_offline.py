#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 00:14:18 2022

@author: qiang
"""

from papillarray.networks import SlipDetectGlobalGru as Net
import torch
import numpy as np
from copy import copy
from papillarray import utils
import argparse
from papillarray import constants as CONSTANTS
import matplotlib.pyplot as plt
import warnings
import time
import os
from papillarray import data_handler as dataset_handler
warnings.filterwarnings("ignore")

def run(args):
    print(f'The are/is {args.model_num} model(s) in the ensemble model')
    print(args.stack_samples)
    # 预训练模型为 8 柱（P1-P8）×滤波 FX/FY = 16 特征；本仓库训练流水线产出的模型为 9 柱 = 18 特征，
    # 以参考权重的真实输入宽度构造网络并自适应选柱组，两种模型都能评估。
    ref = torch.load(os.path.join(args.model_load_path, f"ckpt_{args.model_load_num}_model_0.pth"),
                     weights_only=True)
    args.input_dim = ref['projector.0.weight'].shape[1]
    n_features = args.input_dim // args.stack_samples
    pillar_ids = range(9) if n_features == 18 else range(1, 9)
    models = []
    for i in range(args.model_num):
        m = Net(args).to(args.device)
        m.load_state_dict(torch.load(os.path.join(args.model_load_path, f"ckpt_{args.model_load_num}_model_{i}.pth"),
                                     weights_only=True))
        models.append(m)

    feature_start_num = 4
    feature_end_num = 6
    t_total = 0
    count = 0
    h_space = 0.37
    v_space = 0
    tick_font_size = 14
    title_size = 16
    legend_font_size = 15
    label_size = 16
    
    slip_path = args.slip_path  #For instance: "./datasets/all_cases/z=0.9-v=1-XY.csv"
    df, labels = dataset_handler.reshape_sequence_raw(slip_path)
    labels_modified = dataset_handler.one_hot_label_2cats(labels)
    features = np.concatenate(
        [np.array(df[CONSTANTS.pillars[p][feature_start_num:feature_end_num]]) for p in pillar_ids],
        axis=1)

    features_len = features.shape[0] - (features.shape[0] % args.stack_samples)
    features = features[:features_len]
    labels_modified = labels_modified[:features_len]
    t = []
    for i in range(features_len):
        if i % args.stack_samples == 0:
            t.append(df["time"].iloc[i])
    features = features.reshape(int(features_len/args.stack_samples), int(n_features * args.stack_samples))
    labels_modified = labels_modified.reshape(int(features_len/args.stack_samples), int(args.categories * args.stack_samples))[...,-args.categories:]
    global_slip = []
    with torch.no_grad(): 
        for idx_model,model in enumerate(models):
            model.eval()
            h = torch.zeros(1, args.hidden_dim).to(args.device)
            sub_slip = []
            for idx_row, row in enumerate(features):
                tensor_row = torch.tensor([copy(row)]).to(torch.float32).to(args.device)
                t0 = time.time()
                pred,h = model(tensor_row,h)
                t_total += time.time() - t0
                count += 1
                sub_slip.append(pred[0,1].detach().cpu().numpy())
                
            global_slip.append(sub_slip)
        global_slip = np.array(global_slip).mean(axis=0) * 100
        gt_slip = None
        gross_slip_index = None
        try:
            gt_slip = t[copy(labels_modified[...,1]).tolist().index(1)]
            reverse_labels = copy(labels_modified[...,1]).tolist()
            reverse_labels.reverse()
            gross_slip_index = t[len(labels_modified[...,1].tolist()) - (reverse_labels.index(1) + 1)]
        except ValueError:
            gt_slip = None
        
        
        fig = plt.figure(dpi=500,figsize=(12,8))
        
        ax1 = plt.subplot(211)
        ax1.set_facecolor('lavender')
        plt.axvline(gross_slip_index, color='darkviolet', alpha=0.5,lw=4,linestyle='--', label='Gross slip')
        plt.axhline(50, color='black', alpha = 1.0,lw=1.5,linestyle='--', label='Incipient slip threshold')
        plt.axvline(gt_slip, color='b', linestyle='--',lw=4, alpha =0.5,label='Incipient slip')
        plt.plot(t, global_slip, color='red', alpha = 0.68,lw=2,ls = '-')
        plt.xlabel('Time(s)', fontsize=label_size)
        plt.ylabel('Incipient slip probability (%)',fontsize=label_size)
        plt.ylim((0, 100))
        plt.xticks(fontsize=tick_font_size)
        plt.yticks(fontsize=tick_font_size)
        plt.title('Slip',fontsize=title_size)
    
    stop_path = args.stop_path #For instance: "./datasets/z=0.9-v=1-XY-stop.csv"
    df, labels = dataset_handler.reshape_sequence_raw(stop_path)
    labels_modified = dataset_handler.one_hot_label_2cats(labels)
    features = np.concatenate(
        [np.array(df[CONSTANTS.pillars[p][feature_start_num:feature_end_num]]) for p in pillar_ids],
        axis=1)

    features_len = features.shape[0] - (features.shape[0] % args.stack_samples)
    features = features[:features_len]
    labels_modified = labels_modified[:features_len]
    t = []
    for i in range(features_len):
        if i % args.stack_samples == 0:
            t.append(df["time"].iloc[i])
    features = features.reshape(int(features_len/args.stack_samples), int(n_features * args.stack_samples))
    labels_modified = labels_modified.reshape(int(features_len/args.stack_samples), int(args.categories * args.stack_samples))[...,-args.categories:]
    global_slip = []
    with torch.no_grad(): 
        for idx_model,model in enumerate(models):
            model.eval()
            h = torch.zeros(1, args.hidden_dim).to(args.device)
            sub_slip = []
            for idx_row, row in enumerate(features):
                tensor_row = torch.tensor([copy(row)]).to(torch.float32).to(args.device)
                t0 = time.time()
                pred,h = model(tensor_row,h)
                t_total += time.time() - t0
                count += 1
                
                sub_slip.append(pred[0,1].detach().cpu().numpy())
                
            global_slip.append(sub_slip)
    
        global_slip = np.array(global_slip).mean(axis=0) * 100
        gt_slip = None
        gross_slip_index = None
        try:
            gt_slip = t[copy(labels_modified[...,1]).tolist().index(1)]
            reverse_labels = copy(labels_modified[...,1]).tolist()
            reverse_labels.reverse()
            gross_slip_index = t[len(labels_modified[...,1].tolist()) - (reverse_labels.index(1) + 1)]
        except ValueError:
            gt_slip = None
        
        
        ax2 = plt.subplot(212)
        ax2.set_facecolor('lavender')
        plt.axvline(2.1, color='orange', alpha = 0.9,lw=4,linestyle='--', label='Movement stop')
        plt.plot(t, global_slip, color='red', alpha = 0.68,lw=2,ls = '-', label='Incipient slip probability')
        plt.axhline(50, color='black', alpha = 1.0,lw=1.5,linestyle='--')
        plt.xlabel('Time(s)', fontsize=label_size)
        plt.ylabel('Incipient slip probability (%)',fontsize=label_size)
        plt.ylim((0, 100))
        plt.xticks(fontsize=tick_font_size)
        plt.yticks(fontsize=tick_font_size)
        plt.title('Stop',fontsize=title_size)
    
    lines1, labels1 = fig.axes[0].get_legend_handles_labels()
    lines2, labels2 = fig.axes[1].get_legend_handles_labels()
    
    fig.legend(lines1+lines2, 
              labels1+labels2, 
               loc = 'center',  
               ncol=3, 
               bbox_to_anchor=(0.5, 0.97),
               fontsize=legend_font_size)
    
    plt.subplots_adjust(wspace=v_space, hspace=h_space)
    
    os.makedirs(args.save_path, exist_ok=True)
    plt.savefig(os.path.join(args.save_path, 'pred_slip_stop.jpg'), bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-gpu', default=1, type=int)
    parser.add_argument('--eval', default=1, type=int)
    parser.add_argument('--eval_all', default=1, type=int)
    parser.add_argument('--save-csv', default=0, type=int)
    parser.add_argument('--save-img', default=1, type=int)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--model-load-path', default='pre-trained-models/gripping',
                        help='模型目录（含 reload_info.npy 与 ckpt_*_model_*.pth）')
    parser.add_argument('--model-load-num', default='final', help='checkpoint 编号（epoch 或 final）')
    parser.add_argument('--slip-path', default=None, help='slip 数据 CSV，例如 datasets/fullset/case_1/z=0.9-v=1.0-XY.csv')
    parser.add_argument('--stop-path', default=None, help='stop 数据 CSV，例如 datasets/fullset/case_1/z=0.9-v=1.0-XY-stop.csv')
    args = parser.parse_args()
    if not args.slip_path or not args.stop_path:
        parser.error('--slip-path 与 --stop-path 必填（各给一条 CSV 路径，见 README 数据准备一节）')
    args = utils.reload_args(args)
    args.exp_name = 'eval'
    args = utils.args_handler(args)
    run(args)