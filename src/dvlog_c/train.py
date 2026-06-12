import numpy as np
from re import T
import argparse
import torch
import logging
from .dataset import MyDataLoader
from torch.utils.data import DataLoader
import math
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from math import cos
from tqdm import tqdm
import torch.nn as nn
import time
import os
from sklearn.metrics import confusion_matrix
from .model import Net
import pandas as pd
from sklearn import metrics

# ====================== 超参数配置 ======================
lr = 0.0005  # 提高学习率以加快收敛
epochSize = 500
warmupEpoch = 10  # 增加warmup轮数
testRows = 1
schedule = 'cosine'
classes = ['Normal', 'Depression']

# 损失权重配置（关键：平衡各项损失）
LOSS_WEIGHTS = {
    'classification': 1.0,      # 分类损失权重
    'recon': 0.000004,              # 重建损失权重
    'cross_modal_path': 0.3,   # 跨模态病理特征对比损失
    'same_modal_path': 0.01,    # 同模态病理特征对比损失
    'personalized_repulsive': 1.0,  # 个性化特征排斥损失
    'orthogonal': 1.0,         # 正交约束损失
    'path_sup': 1.0,        # 🆕 病理特征正向监督权重
    'pers_adv': 1.0         # 🆕 个性化特征对抗监督权重 (GRL)
}

ps = []
rs = []
f1s = []
totals = []

total_pre = []
total_label = []

tim = time.strftime('%m_%d__%H_%M', time.localtime())
filepath = 'self_AWF_Per_Lr_New_Enhanced_Disentangled_Contrastive_log_' + str(tim)
savePath1 = "self_AWF_Per_Lr_New_Enhanced_Disentangled_Contrastive_model_" + str(tim)

if not os.path.exists(filepath):
    os.makedirs(filepath)

logging.basicConfig(level=logging.NOTSET,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=filepath + '/' + 'AWF_enhanced_disentangled_training.log',
                    filemode='w')


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 0.5 * (cos(min((current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps),
                              1) * math.pi) + 1)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(VideoPath, AudioPath, X_train, X_test, labelPath, numkfold):
    mytop = 0
    topacc = 60
    top_p = 0
    top_r = 0
    top_f1 = 0
    top_pre = []
    top_label = []

    trainSet = MyDataLoader(VideoPath, AudioPath, X_train, labelPath, "train")
    trainLoader = DataLoader(trainSet, batch_size=15, shuffle=True)
    devSet = MyDataLoader(VideoPath, AudioPath, X_test, labelPath, "dev")
    devLoader = DataLoader(devSet, batch_size=4, shuffle=False)
    print("trainLoader finish", len(trainLoader), len(devLoader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(pathological_dim=15, personalized_dim=15).to(device)

    # 使用AdamW优化器（更好的权重衰减）
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999),
                                  eps=1e-8,
                                  weight_decay=0.01,  # 添加权重衰减
                                  amsgrad=False
                                  )

    train_steps = len(trainLoader) * epochSize
    warmup_steps = len(trainLoader) * warmupEpoch
    target_steps = len(trainLoader) * epochSize

    if schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=train_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=target_steps)

    logging.info('第 {} 折训练开始！'.format(numkfold))
    savePath = str(savePath1) + '/' + str(numkfold)
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    for epoch in range(1, epochSize + 1):
        p = float(epoch) / epochSize
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        model.grl_alpha = alpha

        loop = tqdm(enumerate(trainLoader), total=len(trainLoader))
        total_loss_epoch = 0
        loss_components_epoch = {key: 0 for key in LOSS_WEIGHTS.keys()}
        correct = 0
        total = 0

        model.train()
        for batch_idx, (videoData, audioData, label) in loop:
            videoData, audioData, label = videoData.to(device), audioData.to(device), label.to(device)
            
            # 前向传播
            output, losses = model(videoData, audioData, label, return_losses=True)
            
            # 计算总损失
            total_loss = (
                    losses['classification_loss'] * LOSS_WEIGHTS['classification'] +
                    losses['recon_loss'] * LOSS_WEIGHTS['recon'] +
                    losses['cross_modal_path_loss'] * LOSS_WEIGHTS['cross_modal_path'] +
                    losses['same_modal_path_loss'] * LOSS_WEIGHTS['same_modal_path'] +
                    losses['personalized_repulsive_loss'] * LOSS_WEIGHTS['personalized_repulsive'] +
                    losses['orthogonal_loss'] * LOSS_WEIGHTS['orthogonal'] +
                    losses['path_sup_loss'] * LOSS_WEIGHTS['path_sup'] +  # 🆕 正向监督
                    losses['pers_adv_loss'] * LOSS_WEIGHTS['pers_adv']  # 🆕 逆向对抗
            )
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            # 统计
            total_loss_epoch += total_loss.item()
            for key in LOSS_WEIGHTS.keys():
                if key == 'classification':
                    loss_components_epoch[key] += losses['classification_loss'].item()
                elif key == 'recon':
                    loss_components_epoch[key] += losses['recon_loss'].item()
                elif key == 'cross_modal_path':
                    loss_components_epoch[key] += losses['cross_modal_path_loss'].item()
                elif key == 'same_modal_path':
                    loss_components_epoch[key] += losses['same_modal_path_loss'].item()
                elif key == 'personalized_repulsive':
                    loss_components_epoch[key] += losses['personalized_repulsive_loss'].item()
                elif key == 'orthogonal':
                    loss_components_epoch[key] += losses['orthogonal_loss'].item()
                elif key == 'path_sup':  # 🆕 记录新的 Loss
                    loss_components_epoch[key] += losses['path_sup_loss'].item()
                elif key == 'pers_adv':  # 🆕 记录新的 Loss
                    loss_components_epoch[key] += losses['pers_adv_loss'].item()
            
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum()

            loop.set_description(f'Train Epoch [{epoch}/{epochSize}]')
            loop.set_postfix(loss=total_loss.item(), acc=f'{100.0 * correct / total:.2f}%')

        # 记录训练信息
        avg_loss = total_loss_epoch / len(trainLoader)
        train_acc = 100.0 * correct / total
        
        loss_info = ', '.join([f'{key}: {loss_components_epoch[key]/len(trainLoader):.4f}' 
                               for key in LOSS_WEIGHTS.keys()])
        logging.info(f'Epoch: {epoch}, Train Loss: {avg_loss:.4f}, Acc: {train_acc:.2f}%')
        logging.info(f'Loss Components: {loss_info}')

        # 验证阶段
        if epoch >= warmupEpoch and epoch % testRows == 0:
            model.eval()
            print("*******验证阶段********")
            loop = tqdm(enumerate(devLoader), total=len(devLoader))
            
            val_loss = 0
            correct = 0
            total = 0
            label_list = []
            pred_list = []

            with torch.no_grad():
                for batch_idx, (videoData, audioData, label) in loop:
                    videoData, audioData, label = videoData.to(device), audioData.to(device), label.to(device)
                    
                    devOutput, dev_losses = model(videoData, audioData, label, return_losses=True)
                    
                    # 计算验证损失
                    batch_loss = (
                            dev_losses['classification_loss'] * LOSS_WEIGHTS['classification'] +
                            dev_losses['recon_loss'] * LOSS_WEIGHTS['recon'] +
                            dev_losses['cross_modal_path_loss'] * LOSS_WEIGHTS['cross_modal_path'] +
                            dev_losses['same_modal_path_loss'] * LOSS_WEIGHTS['same_modal_path'] +
                            dev_losses['personalized_repulsive_loss'] * LOSS_WEIGHTS['personalized_repulsive'] +
                            dev_losses['orthogonal_loss'] * LOSS_WEIGHTS['orthogonal'] +
                            dev_losses['path_sup_loss'] * LOSS_WEIGHTS['path_sup'] +  # 🆕
                            dev_losses['pers_adv_loss'] * LOSS_WEIGHTS['pers_adv']  # 🆕
                    )

                    val_loss += batch_loss.item()

                    _, predicted = torch.max(devOutput.data, 1)
                    total += label.size(0)
                    correct += predicted.eq(label.data).cpu().sum()

                    label_list.extend(label.data.tolist())
                    pred_list.extend(predicted.tolist())

            acc = 100.0 * correct / total
            avg_val_loss = val_loss / len(devLoader)
            
            print(f"验证标签: {label_list}")
            print(f"预测结果: {pred_list}")
            
            label_array = np.array(label_list)
            pred_array = np.array(pred_list)

            # 计算混淆矩阵和指标
            cm = confusion_matrix(label_array, pred_array)
            print(f"混淆矩阵:\n{cm}")
            
            if cm.shape == (2, 2):
                tp = cm[1, 1]
                fp = cm[0, 1]
                fn = cm[1, 0]
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = 0
                recall = 0
                f1score = 0

            logging.info(f'Epoch: {epoch}, Val Loss: {avg_val_loss:.4f}, Acc: {acc:.2f}%')
            logging.info(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1score:.4f}')
            
            print(f'验证 - Epoch: {epoch}, Loss: {avg_val_loss:.4f}, Acc: {acc:.2f}%')
            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1score:.4f}')
            
            # 更新最佳结果
            if acc > mytop:
                mytop = acc
                top_p = precision
                top_r = recall
                top_f1 = f1score
                top_pre = pred_list
                top_label = label_list

            # 保存模型
            if acc > topacc:
                topacc = acc
                checkpoint = {
                    'net': model.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                    'epoch': epoch,
                    'scheduler': scheduler.state_dict()
                }
                model_name = f"enhanced_ep{epoch}_acc{acc:.2f}_p{precision:.4f}_r{recall:.4f}_f1{f1score:.4f}.pth"
                torch.save(checkpoint, os.path.join(savePath, model_name))
                logging.info(f'保存模型: {model_name}')

    totals.append(mytop)
    ps.append(top_p)
    rs.append(top_r)
    f1s.append(top_f1)
    logging.info(f'第 {numkfold} 折最佳准确率: {mytop:.2f}%')
    logging.info('')

    print("训练结束")

    return np.array(top_label), np.array(top_pre)

def parse_args():
    parser = argparse.ArgumentParser(description="Train the DVLOG_C model.")
    parser.add_argument("--video-dir", required=True, help="Directory containing video .npy feature files.")
    parser.add_argument("--audio-dir", required=True, help="Directory containing audio .npy feature files.")
    parser.add_argument("--label-dir", required=True, help="Directory containing *_Depression.csv label files.")
    parser.add_argument("--folds", type=int, default=10, help="Number of stratified K-fold splits.")
    parser.add_argument("--seed", type=int, default=2222, help="Random seed.")
    return parser.parse_args()


if __name__ == '__main__':
    import random
    from sklearn.model_selection import KFold, StratifiedKFold

    args = parse_args()
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tcn = args.video_dir
    mdnAudioPath = args.audio_dir
    labelPath = args.label_dir

    Y = []  # 存储视频文件对应的标签
    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    X = os.listdir(tcn)
    X.sort(key=lambda x: int(x.split(".")[0]))
    X = np.array(X)

    for i in X:
        file_csv = pd.read_csv(os.path.join(labelPath, (str(i.split('.npy')[0]) + "_Depression.csv")))
        bdi = int(file_csv.columns[0])
        Y.append(bdi)

    numkfold = 0

    for train_index, test_index in kf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        numkfold += 1
        logging.info(f'第 {numkfold} 折训练集: {X_train}')
        logging.info(f'第 {numkfold} 折测试集: {X_test}')
        total_label_0, total_pre_0 = train(tcn, mdnAudioPath, X_train, X_test, labelPath, numkfold)
        total_pre.append(total_pre_0)
        total_label.append(total_label_0)

    total_pre = np.concatenate(total_pre, axis=0)
    total_label = np.concatenate(total_label, axis=0)
    
    np.save(os.path.join(filepath, "total_pre.npy"), total_pre)
    np.save(os.path.join(filepath, "total_label.npy"), total_label)

    logging.info(f'所有折准确率: {totals}')
    logging.info(f'平均准确率: {sum(totals) / len(totals):.2f}%')
    logging.info(f'所有折精确率: {ps}')
    logging.info(f'平均精确率: {sum(ps) / len(ps):.4f}')
    logging.info(f'所有折召回率: {rs}')
    logging.info(f'平均召回率: {sum(rs) / len(rs):.4f}')
    logging.info(f'所有折F1分数: {f1s}')
    logging.info(f'平均F1分数: {sum(f1s) / len(f1s):.4f}')
    
    print("=" * 50)
    print(f"最终结果:")
    print(f"平均准确率: {sum(totals) / len(totals):.2f}%")
    print(f"平均精确率: {sum(ps) / len(ps):.4f}")
    print(f"平均召回率: {sum(rs) / len(rs):.4f}")
    print(f"平均F1分数: {sum(f1s) / len(f1s):.4f}")
    print("=" * 50)
