import sys
import numpy as np
from re import T
import argparse
import torch
import torch.nn.functional as F
import logging
import warnings

# 灞忚斀甯歌璀﹀憡锛堜笉褰卞搷璁粌锛?
warnings.filterwarnings('ignore', message='.*flash attention.*')
warnings.filterwarnings('ignore', message='.*device.*argument.*deprecated.*')
warnings.filterwarnings('ignore', category=UserWarning)
from .dataset import MyDataLoader
from torch.utils.data import DataLoader
import math
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from math import cos
from tqdm import tqdm
import torch.nn as nn
import time
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
import random

# 鉁?瀵煎叆鏀硅繘鐨勭嚎绱㈢骇瀵规瘮瀛︿範妯″瀷
from .model import CueLevelContrastiveBCDCIE

# ====================== 璁粌鍙傛暟閰嶇疆 ======================
lr = 0.00001
epochSize = 300
warmupEpoch = 0
testRows = 1
schedule = 'cosine'
classes = ['Normal', 'Depression']

# 鉁?瀵规瘮瀛︿範鐩稿叧鍙傛暟
use_contrastive_learning = True
contrastive_warmup_epochs = 15  # 棰勭儹闃舵锛氫笉鍚敤瀵规瘮瀛︿範鎹熷け
pathological_dim = 32
personalized_dim = 32

# 馃啎 鍙В閲婃€х浉鍏冲弬鏁?
log_interpretability_interval = 10  # 姣忛殧澶氬皯epoch璁板綍涓€娆″彲瑙ｉ噴鎬т俊鎭?

# 鉁?鏃╁仠鍙傛暟
early_stop_patience = 300  # 楠岃瘉闆嗗噯纭巼杩炵画澶氬皯杞棤鎻愬崌鍒欏仠姝?

# 鉁?鎹熷け鏉冮噸锛堜笌妯″瀷鍐呴儴涓€鑷达級
classification_weight = 1.0
reconstruction_weight = 0.00005
contrastive_weight = 0.5
orthogonal_weight = 0.1
auxiliary_weight = 0.2              # 杈呭姪鍒嗙被鎹熷け鏉冮噸
path_supervision_weight = 1.0       # 馃啎 鐥呯悊鐗瑰緛姝ｅ悜鐩戠潱鏉冮噸
pers_adversarial_weight = 0.5       # 馃啎 涓€у寲鐗瑰緛瀵规姉鐩戠潱鏉冮噸 (GRL)

# 璁板綍缁撴灉
ps = []
rs = []
f1s = []
totals = []
total_pre = []
total_label = []

# 鏃堕棿鎴冲拰璺緞
tim = time.strftime('%m_%d__%H_%M', time.localtime())
filepath = 'A_Cue_level_Contrastive_BCDCIE_log_' + str(tim)
savePath1 = "A_Cue_level_Contrastive_BCDCIE_model_" + str(tim)

if not os.path.exists(filepath):
    os.makedirs(filepath)

logging.basicConfig(level=logging.NOTSET,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=filepath + '/' + 'Cue_level_Contrastive_training.log',
                    filemode='w')


# ====================== 瀛︿範鐜囪皟搴﹀櫒 ======================
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


# 馃啎 绾跨储閲嶈鎬х洃鎺у嚱鏁帮紙鏀硅繘鐗堬級
def log_cue_importance(results, epoch, numkfold, phase='train'):
    """
    璁板綍绾跨储閲嶈鎬т俊鎭紙閫傞厤鏀硅繘鍚庣殑妯″瀷杈撳嚭锛?

    Args:
        results: 妯″瀷杈撳嚭缁撴灉
        epoch: 褰撳墠epoch
        numkfold: 褰撳墠fold
        phase: 'train' 鎴?'dev'
    """
    if 'cue_weights' not in results or 'cue_importance' not in results:
        return

    cue_weights_dict = results['cue_weights']  # {'gaze': [batch, 2], ...}
    cue_importance = results['cue_importance']  # [batch, 4]
    cue_names = ['gaze', 'pose', 'landmark', 'au']

    # 璁板綍鍒版棩蹇?
    logging.info(f'[{phase.upper()}] Fold {numkfold}, Epoch {epoch} - Cue Interpretability:')
    logging.info('-' * 60)

    # 1. 姣忎釜绾跨储鐨勭梾鐞?涓€у寲鏉冮噸
    logging.info('Individual Cue Weights (Pathological / Personalized):')
    for cue_name in cue_names:
        if cue_name in cue_weights_dict:
            weights = cue_weights_dict[cue_name]  # [batch, 2]
            avg_weights = weights.mean(dim=0).detach().cpu().numpy()
            logging.info(f'  {cue_name.capitalize():10s}: Path {avg_weights[0]:.4f}, Pers {avg_weights[1]:.4f}')

    # 2. 绾跨储閲嶈鎬э紙鏁翠綋璐＄尞锛?
    logging.info('\nCue Importance (Overall Contribution):')
    avg_importance = cue_importance.mean(dim=0).detach().cpu().numpy()
    for i, cue_name in enumerate(cue_names):
        logging.info(f'  {cue_name.capitalize():10s}: {avg_importance[i]:.4f} ({avg_importance[i]*100:.1f}%)')

    # 3. 绫诲埆绾х粺璁★紙濡傛灉鏈夛級
    if 'avg_weights_depression' in results:
        logging.info('\nDepression Samples - Average Weights:')
        for cue_name in cue_names:
            if cue_name in results['avg_weights_depression']:
                weights = results['avg_weights_depression'][cue_name].detach().cpu().numpy()
                logging.info(f'  {cue_name.capitalize():10s}: Path {weights[0]:.4f}, Pers {weights[1]:.4f}')

    if 'avg_weights_normal' in results:
        logging.info('\nNormal Samples - Average Weights:')
        for cue_name in cue_names:
            if cue_name in results['avg_weights_normal']:
                weights = results['avg_weights_normal'][cue_name].detach().cpu().numpy()
                logging.info(f'  {cue_name.capitalize():10s}: Path {weights[0]:.4f}, Pers {weights[1]:.4f}')

    # 4. 绾跨储閲嶈鎬х殑绫诲埆宸紓
    if 'avg_cue_importance_depression' in results and 'avg_cue_importance_normal' in results:
        dep_importance = results['avg_cue_importance_depression'].detach().cpu().numpy()
        norm_importance = results['avg_cue_importance_normal'].detach().cpu().numpy()
        logging.info('\nCue Importance by Label:')
        logging.info('  Cue        | Depression | Normal    | Diff')
        logging.info('  ' + '-' * 50)
        for i, cue_name in enumerate(cue_names):
            diff = dep_importance[i] - norm_importance[i]
            logging.info(f'  {cue_name.capitalize():10s} | {dep_importance[i]:.4f}     | {norm_importance[i]:.4f}    | {diff:+.4f}')

    logging.info('-' * 60)


# ====================== 闆嗘垚璁粌鍑芥暟 ======================
def train_cue_contrastive_model(VideoPath, AudioPath, X_train, X_test, labelPath, numkfold):
    """
    闆嗘垚绾跨储绾у姣斿涔犵殑璁粌鍑芥暟锛堟敼杩涚増锛?
    """
    mytop = 0
    topacc = 60
    top_p = 0
    top_r = 0
    top_f1 = 0
    top_pre = []
    top_label = []

    # 鏁版嵁鍔犺浇
    trainSet = MyDataLoader(VideoPath, AudioPath, X_train, labelPath, "train")
    trainLoader = DataLoader(trainSet, batch_size=15, shuffle=True)
    devSet = MyDataLoader(VideoPath, AudioPath, X_test, labelPath, "dev")
    devLoader = DataLoader(devSet, batch_size=4, shuffle=False)
    print("trainLoader finish", len(trainLoader), len(devLoader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 鉁?鍒涘缓鏀硅繘鐨勫姣斿涔犳ā鍨?
    model = CueLevelContrastiveBCDCIE(
        pathological_dim=pathological_dim,
        personalized_dim=personalized_dim
    ).to(device)
    logging.info('='*60)
    logging.info('Using IMPROVED Cue-Level Contrastive Learning Model')
    logging.info('='*60)
    logging.info('Key Features:')
    logging.info('  1. 鉁?No Cross Attention - Preserves Cue Independence')
    logging.info('  2. 鉁?Per-Cue Adaptive Weighting - Independent Weight Assignment')
    logging.info('  3. 鉁?Simple Fusion + Cue Importance Gating')
    logging.info('  4. 鉁?Enhanced Interpretability - Traceable Cue Contribution')
    logging.info('='*60)
    logging.info(f'Pathological Dim: {pathological_dim}')
    logging.info(f'Personalized Dim: {personalized_dim}')
    logging.info(f'Contrastive Warmup Epochs: {contrastive_warmup_epochs}')
    logging.info('='*60)

    # 鎹熷け鍑芥暟锛堢敤浜庨獙璇侀樁娈碉級
    lossFunc = nn.CrossEntropyLoss().to(device)

    # 浼樺寲鍣?
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 weight_decay=0,
                                 amsgrad=False)

    # 瀛︿範鐜囪皟搴?
    train_steps = len(trainLoader) * epochSize
    warmup_steps = len(trainLoader) * warmupEpoch
    target_steps = len(trainLoader) * epochSize

    if schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=train_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=target_steps)

    logging.info('The {}  fold training begins锛侊紒'.format(numkfold))
    savePath = str(savePath1) + '/' + str(numkfold)
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # 鏃╁仠璁℃暟鍣?
    no_improve_cnt = 0

    # ====================== 璁粌寰幆 ======================
    for epoch in range(1, epochSize):
        p = float(epoch) / epochSize
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        model.grl_alpha = alpha  # 浼犵粰妯″瀷

        loop = tqdm(enumerate(trainLoader), total=len(trainLoader))
        traloss_one = 0
        correct = 0
        total = 0
        lable1 = []
        pre1 = []

        # 鉁?鎹熷け缁勪欢璁板綍
        epoch_classification_loss = 0
        epoch_reconstruction_loss = 0
        epoch_contrastive_loss = 0
        epoch_orthogonal_loss = 0
        epoch_auxiliary_loss = 0
        epoch_path_sup_loss = 0  # 馃啎 鐥呯悊姝ｅ悜鐩戠潱
        epoch_pers_adv_loss = 0  # 馃啎 涓€у寲瀵规姉鐩戠潱

        # 鍚屾瑕嗙洊妯″瀷鍐呴儴鐨勬崯澶辨潈閲嶏紝淇濊瘉涓庤缁冭剼鏈竴鑷?
        model.contrastive_weight       = contrastive_weight
        model.reconstruction_weight    = reconstruction_weight
        model.classification_weight    = classification_weight
        model.orthogonal_weight        = orthogonal_weight
        model.auxiliary_weight         = auxiliary_weight
        model.path_supervision_weight = path_supervision_weight  # 馃啎
        model.pers_adversarial_weight = pers_adversarial_weight  # 馃啎

        model.train()
        for batch_idx, (videoData, audioData, label) in loop:
            videoData, audioData, label = videoData.to(device), audioData.to(device), label.to(device)

            # 鉁?鍓嶅悜浼犳挱
            results = model(videoData, audioData, label)
            output = results['classification_output']
            total_loss = results['total_loss']

            # 鉁?璁板綍鍚勭粍浠舵崯澶?
            losses = results['losses']
            epoch_classification_loss += losses['classification'].item() if torch.is_tensor(losses['classification']) else losses['classification']
            epoch_reconstruction_loss += losses['reconstruction'].item() if torch.is_tensor(losses['reconstruction']) else losses['reconstruction']
            epoch_orthogonal_loss += losses['orthogonal'].item() if torch.is_tensor(losses['orthogonal']) else losses['orthogonal']

            # 銆愭柊澧炪€戣褰曞鍋剁洃鐫ｆ崯澶?
            if 'path_supervision' in losses:
                epoch_path_sup_loss += losses['path_supervision'].item() if torch.is_tensor(
                    losses['path_supervision']) else losses['path_supervision']
            if 'pers_adversarial' in losses:
                epoch_pers_adv_loss += losses['pers_adversarial'].item() if torch.is_tensor(
                    losses['pers_adversarial']) else losses['pers_adversarial']
            # 杈呭姪鎹熷け
            if 'auxiliary' in losses:
                aux_loss_value = losses['auxiliary'].item() if torch.is_tensor(losses['auxiliary']) else losses['auxiliary']
                epoch_auxiliary_loss += aux_loss_value

            # 瀵规瘮瀛︿範鎹熷け
            contrastive_loss_value = losses['contrastive'].item() if torch.is_tensor(losses['contrastive']) else losses['contrastive']
            epoch_contrastive_loss += contrastive_loss_value

            # 鉁?鏍规嵁璁粌闃舵璋冩暣鎹熷け浣跨敤绛栫暐
            if epoch < contrastive_warmup_epochs:
                # 棰勭儹闃舵锛氫笉浣跨敤瀵规瘮瀛︿範鎹熷け
                aux_term = losses['auxiliary'] if 'auxiliary' in losses and torch.is_tensor(losses['auxiliary']) else 0.0
                path_term = losses.get('path_supervision', 0.0)
                pers_term = losses.get('pers_adversarial', 0.0)
                traLoss = (classification_weight * losses['classification'] +
                           reconstruction_weight * losses['reconstruction'] +
                           orthogonal_weight * losses['orthogonal'] +
                           aux_term)
            else:
                # 姝ｅ紡璁粌闃舵锛氫娇鐢ㄥ畬鏁存崯澶?
                traLoss = total_loss

            traloss_one += traLoss.item()
            optimizer.zero_grad()
            traLoss.backward()

            # 鉁?姊害瑁佸壀
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum()

            loop.set_description(f'Train Epoch [{epoch}/{epochSize}]')
            loop.set_postfix(loss=traloss_one / (batch_idx + 1))

        # 鉁?璁板綍璁粌鏃ュ織
        avg_train_loss = traloss_one / len(trainLoader)
        train_acc = 100.0 * correct / total

        if epoch < contrastive_warmup_epochs:
            logging.info('銆怶armup銆慐poch: {}, Batch: {}, Total Loss:{:.4f}, Acc:{:.2f}%, '
                         'Cls:{:.4f}, Recon:{:.4f}, Ortho:{:.4f}, PSup:{:.4f}, PAdv:{:.4f}, Alpha:{:.2f}'.format(
                epoch, batch_idx + 1, avg_train_loss, train_acc,
                       epoch_classification_loss / len(trainLoader),
                       epoch_reconstruction_loss / len(trainLoader),
                       epoch_orthogonal_loss / len(trainLoader),
                       epoch_path_sup_loss / len(trainLoader),
                       epoch_pers_adv_loss / len(trainLoader),
                alpha))
        else:
            logging.info('銆怲raining銆慐poch: {}, Batch: {}, Total Loss:{:.4f}, Acc:{:.2f}%, '
                         'Cls:{:.4f}, Recon:{:.4f}, Contr:{:.4f}, Ortho:{:.4f}, PSup:{:.4f}, PAdv:{:.4f}, Alpha:{:.2f}'.format(
                epoch, batch_idx + 1, avg_train_loss, train_acc,
                       epoch_classification_loss / len(trainLoader),
                       epoch_reconstruction_loss / len(trainLoader),
                       epoch_contrastive_loss / len(trainLoader),
                       epoch_orthogonal_loss / len(trainLoader),
                       epoch_path_sup_loss / len(trainLoader),
                       epoch_pers_adv_loss / len(trainLoader),
                alpha))

        # 馃啎 瀹氭湡璁板綍绾跨储閲嶈鎬?
        if epoch % log_interpretability_interval == 0:
            log_cue_importance(results, epoch, numkfold, phase='train')

        # ====================== 楠岃瘉闃舵 ======================
        if epoch - warmupEpoch >= 0 and epoch % testRows == 0:
            train_num = 0
            correct = 0
            total = 0
            label2 = []
            pre2 = []

            model.eval()
            print("*******dev********")
            loop = tqdm(enumerate(devLoader), total=len(devLoader))

            # 馃啎 鐢ㄤ簬绱Н楠岃瘉缁撴灉
            all_results = {
                'cue_weights': {'gaze': [], 'pose': [], 'landmark': [], 'au': []},
                'cue_importance': []
            }

            with torch.no_grad():
                loss_one = 0
                dev_classification_loss = 0
                dev_reconstruction_loss = 0
                dev_path_sup_loss = 0  # 馃啎
                dev_pers_adv_loss = 0  # 馃啎

                for batch_idx, (videoData, audioData, label) in loop:
                    videoData, audioData, label = videoData.to(device), audioData.to(device), label.to(device)

                    # 楠岃瘉鏃朵紶鍏abels锛堢敤浜庤绠楃被鍒骇缁熻锛?
                    results = model(videoData, audioData, label)
                    devOutput = results['classification_output']
                    dev_losses = results['losses']

                    # 馃啎 鏀堕泦绾跨储閲嶈鎬т俊鎭?
                    if 'cue_weights' in results:
                        for cue_name in ['gaze', 'pose', 'landmark', 'au']:
                            if cue_name in results['cue_weights']:
                                all_results['cue_weights'][cue_name].append(
                                    results['cue_weights'][cue_name].detach()
                                )
                    if 'cue_importance' in results:
                        all_results['cue_importance'].append(results['cue_importance'].detach())

                    # 鉁?鎵嬪姩璁＄畻楠岃瘉鎹熷け
                    cls_loss = lossFunc(devOutput, label.long())
                    loss = cls_loss
                    dev_classification_loss += cls_loss.item()

                    recon_loss = dev_losses['reconstruction']
                    loss += reconstruction_weight * (recon_loss.item() if torch.is_tensor(recon_loss) else recon_loss)
                    dev_reconstruction_loss += (recon_loss.item() if torch.is_tensor(recon_loss) else recon_loss)

                    # 馃啎 楠岃瘉鏃剁殑瀵瑰伓鐩戠潱鎹熷け
                    if 'path_supervision' in dev_losses:
                        p_sup = dev_losses['path_supervision']
                        p_sup_val = p_sup.item() if torch.is_tensor(p_sup) else p_sup
                        loss += path_supervision_weight * p_sup_val
                        dev_path_sup_loss += p_sup_val

                    if 'pers_adversarial' in dev_losses:
                        p_adv = dev_losses['pers_adversarial']
                        p_adv_val = p_adv.item() if torch.is_tensor(p_adv) else p_adv
                        loss += pers_adversarial_weight * p_adv_val
                        dev_pers_adv_loss += p_adv_val

                    loss_one += loss.item()
                    train_num += label.size(0)

                    _, predicted = torch.max(devOutput.data, 1)
                    total += label.size(0)
                    correct += predicted.eq(label.data).cpu().sum()

                    label2.append(label.data)
                    pre2.append(predicted)
                    lable1 += label.data.tolist()
                    pre1 += predicted.tolist()

            # 鉁?璁＄畻楠岃瘉鎸囨爣
            acc = 100.0 * correct / total
            lable1 = np.array(lable1)
            pre1 = np.array(pre1)

            cm = confusion_matrix(lable1, pre1)
            print("Confusion Matrix:")
            print(cm)

            if cm.shape == (2, 2):
                tp = cm[1, 1]
                fp = cm[0, 1]
                fn = cm[1, 0]
                tn = cm[0, 0]

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                logging.info('Confusion Matrix: TP={}, FP={}, FN={}, TN={}'.format(tp, fp, fn, tn))
            else:
                precision = 0
                recall = 0
                f1score = 0
                logging.warning('Confusion matrix shape is not (2,2), got {}'.format(cm.shape))

            logging.info('Precision: {:.4f}'.format(precision))
            logging.info('Recall: {:.4f}'.format(recall))
            logging.info('F1-Score: {:.4f}'.format(f1score))

            avg_dev_loss = loss_one / len(devLoader)

            logging.info('銆怐ev銆慐poch: {}, Total Loss:{:.4f}, Acc:{:.2f}%, '
                        'Cls:{:.4f}, Recon:{:.4f}, P:{:.4f}, R:{:.4f}, F1:{:.4f}'.format(
                            epoch, avg_dev_loss, acc,
                            dev_classification_loss / len(devLoader),
                            dev_reconstruction_loss / len(devLoader),
                            dev_path_sup_loss / len(devLoader),  # 馃啎
                            dev_pers_adv_loss / len(devLoader),  # 馃啎
                            precision, recall, f1score))

            loop.set_description(f'__Dev Epoch [{epoch}/{epochSize}]')
            loop.set_postfix(loss=avg_dev_loss, acc=acc)
            print('Dev epoch:{}, Loss:{:.4f}, Acc:{:.2f}%, P:{:.4f}, R:{:.4f}, F1:{:.4f}'.format(
                epoch, avg_dev_loss, acc, precision, recall, f1score))

            # 馃啎 鍦ㄩ獙璇佺粨鏉熷悗璁板綍鍙В閲婃€?
            if len(all_results['cue_importance']) > 0 and epoch % log_interpretability_interval == 0:
                # 鍚堝苟鎵€鏈塨atch鐨勭粨鏋?
                aggregated_results = {
                    'cue_weights': {},
                    'cue_importance': torch.cat(all_results['cue_importance'], dim=0)
                }

                for cue_name in ['gaze', 'pose', 'landmark', 'au']:
                    if len(all_results['cue_weights'][cue_name]) > 0:
                        aggregated_results['cue_weights'][cue_name] = torch.cat(
                            all_results['cue_weights'][cue_name], dim=0
                        )

                # 娣诲姞鏈€鍚庝竴涓猙atch鐨勭被鍒骇缁熻锛堜綔涓哄弬鑰冿級
                if 'avg_weights_depression' in results:
                    aggregated_results['avg_weights_depression'] = results['avg_weights_depression']
                if 'avg_weights_normal' in results:
                    aggregated_results['avg_weights_normal'] = results['avg_weights_normal']
                if 'avg_cue_importance_depression' in results:
                    aggregated_results['avg_cue_importance_depression'] = results['avg_cue_importance_depression']
                if 'avg_cue_importance_normal' in results:
                    aggregated_results['avg_cue_importance_normal'] = results['avg_cue_importance_normal']

                log_cue_importance(aggregated_results, epoch, numkfold, phase='dev')

            # 鉁?鏇存柊鏈€浣崇粨鏋?+ 鏃╁仠璁℃暟
            if acc > mytop:
                mytop = max(acc, mytop)
                top_p = precision
                top_r = recall
                top_f1 = f1score
                top_pre = pre2
                top_label = label2
                no_improve_cnt = 0
                logging.info('馃幆 New best accuracy: {:.2f}%'.format(mytop))
            else:
                no_improve_cnt += 1
                logging.info('鈴?No improvement ({}/{}). Best so far: {:.2f}%'.format(
                    no_improve_cnt, early_stop_patience, mytop))
                if no_improve_cnt >= early_stop_patience:
                    logging.info('馃洃 Early stopping at epoch {}, best acc: {:.2f}%'.format(
                        epoch, mytop))
                    print('Early stopping at epoch {}, best acc: {:.2f}%'.format(epoch, mytop))
                    break

            # 鉁?淇濆瓨妯″瀷
            if acc > topacc:
                topacc = max(acc, topacc)
                checkpoint = {
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'scheduler': scheduler.state_dict(),
                    'accuracy': acc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1score,
                    # 馃啎 娣诲姞妯″瀷閰嶇疆
                    'config': {
                        'pathological_dim': pathological_dim,
                        'personalized_dim': personalized_dim,
                        'model_version': 'improved_no_cross_attention'
                    }
                }
                model_name = f"cue_contrastive_ep{epoch}_acc{acc:.2f}_p{precision:.4f}_r{recall:.4f}_f1{f1score:.4f}.pth"
                torch.save(checkpoint, os.path.join(savePath, model_name))
                logging.info('馃捑 Model saved: {}'.format(model_name))

    # 鉁?澶勭悊鏈€缁堢粨鏋?
    if len(top_pre) > 0 and len(top_label) > 0:
        top_pre = torch.cat(top_pre, axis=0).cpu()
        top_label = torch.cat(top_label, axis=0).cpu()
    else:
        logging.warning('No predictions saved during training!')
        top_pre = torch.tensor([])
        top_label = torch.tensor([])

    totals.append(mytop)
    ps.append(top_p)
    rs.append(top_r)
    f1s.append(top_f1)

    logging.info('='*50)
    logging.info('Fold {} Summary:'.format(numkfold))
    logging.info('Best Accuracy: {:.2f}%'.format(mytop))
    logging.info('Precision: {:.4f}'.format(top_p))
    logging.info('Recall: {:.4f}'.format(top_r))
    logging.info('F1-Score: {:.4f}'.format(top_f1))
    logging.info('='*50)

    print("Train fold {} end".format(numkfold))
    return top_label, top_pre


# ====================== 涓昏缁冩祦绋?======================
def parse_args():
    parser = argparse.ArgumentParser(description="Train the LMVD_C cue-level model.")
    parser.add_argument("--video-dir", required=True, help="Directory containing video .npy feature files.")
    parser.add_argument("--audio-dir", required=True, help="Directory containing audio .npy feature files.")
    parser.add_argument("--label-dir", required=True, help="Directory containing *_Depression.csv label files.")
    parser.add_argument("--folds", type=int, default=10, help="Number of stratified K-fold splits.")
    parser.add_argument("--seed", type=int, default=2222, help="Random seed.")
    return parser.parse_args()


if __name__ == '__main__':
    # 璁剧疆闅忔満绉嶅瓙
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 鏁版嵁璺緞
    tcn = args.video_dir
    mdnAudioPath = args.audio_dir
    labelPath = args.label_dir

    # 鍑嗗鏁版嵁
    Y = []
    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    X = os.listdir(tcn)
    X.sort(key=lambda x: int(x.split(".")[0]))
    X = np.array(X)
    print("Total samples:", len(X))
    print("Sample files:", X[:5], "...")

    for i in X:
        file_csv = pd.read_csv(os.path.join(labelPath, (str(i.split('.npy')[0]) + "_Depression.csv")))
        bdi = int(file_csv.columns[0])
        Y.append(bdi)

    Y = np.array(Y)
    print("Label distribution:", np.bincount(Y))

    numkfold = 0

    # 鉁?K鎶樹氦鍙夐獙璇佽缁?
    logging.info('='*60)
    logging.info('Starting 10-Fold Cross Validation Training')
    logging.info('Model: IMPROVED Cue-Level Contrastive Learning BCDCIE')
    logging.info('='*60)
    logging.info('Key Improvements:')
    logging.info('  1. No Cross Attention - Preserves Cue Independence')
    logging.info('  2. Per-Cue Adaptive Weighting')
    logging.info('  3. Simple Fusion + Cue Importance Gating')
    logging.info('  4. Enhanced Interpretability')
    logging.info('='*60)
    logging.info('Total Samples: {}'.format(len(X)))
    logging.info('='*60)

    for train_index, test_index in kf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        numkfold += 1

        logging.info('')
        logging.info('='*50)
        logging.info('Fold {} / 10'.format(numkfold))
        logging.info('Train samples: {}, Test samples: {}'.format(len(X_train), len(X_test)))
        logging.info('Train label dist: {}'.format(np.bincount(Y_train)))
        logging.info('Test label dist: {}'.format(np.bincount(Y_test)))
        logging.info('='*50)

        total_label_0, total_pre_0 = train_cue_contrastive_model(
            tcn, mdnAudioPath, X_train, X_test, labelPath, numkfold
        )

        if len(total_pre_0) > 0:
            total_pre.append(total_pre_0)
            total_label.append(total_label_0)

    # 鉁?淇濆瓨鏈€缁堢粨鏋?
    if len(total_pre) > 0:
        total_pre = torch.cat(total_pre, axis=0).cpu().numpy()
        total_label = torch.cat(total_label, axis=0).cpu().numpy()
        np.save(filepath + "/total_pre.npy", total_pre)
        np.save(filepath + "/total_label.npy", total_label)

    # 鉁?璁板綍鏈€缁堢粺璁＄粨鏋?
    logging.info('')
    logging.info('='*60)
    logging.info('FINAL RESULTS - 10-Fold Cross Validation')
    logging.info('='*60)
    logging.info('Accuracy per fold: {}'.format(['{:.2f}%'.format(x) for x in totals]))
    logging.info('Average Accuracy: {:.2f}%'.format(sum(totals) / len(totals)))
    logging.info('Std Accuracy: {:.2f}%'.format(np.std(totals)))
    logging.info('')
    logging.info('Precision per fold: {}'.format(['{:.4f}'.format(x) for x in ps]))
    logging.info('Average Precision: {:.4f}'.format(sum(ps) / len(ps)))
    logging.info('Std Precision: {:.4f}'.format(np.std(ps)))
    logging.info('')
    logging.info('Recall per fold: {}'.format(['{:.4f}'.format(x) for x in rs]))
    logging.info('Average Recall: {:.4f}'.format(sum(rs) / len(rs)))
    logging.info('Std Recall: {:.4f}'.format(np.std(rs)))
    logging.info('')
    logging.info('F1-Score per fold: {}'.format(['{:.4f}'.format(x) for x in f1s]))
    logging.info('Average F1-Score: {:.4f}'.format(sum(f1s) / len(f1s)))
    logging.info('Std F1-Score: {:.4f}'.format(np.std(f1s)))
    logging.info('='*60)

    print("\n" + "="*60)
    print("=== 鏈€缁堢粨鏋滄眹鎬?===")
    print("="*60)
    print(f"骞冲潎鍑嗙‘鐜? {sum(totals) / len(totals):.2f}% (卤{np.std(totals):.2f}%)")
    print(f"骞冲潎绮剧‘鐜? {sum(ps) / len(ps):.4f} (卤{np.std(ps):.4f})")
    print(f"骞冲潎鍙洖鐜? {sum(rs) / len(rs):.4f} (卤{np.std(rs):.4f})")
    print(f"骞冲潎F1鍒嗘暟: {sum(f1s) / len(f1s):.4f} (卤{np.std(f1s):.4f})")
    print("="*60)

    print("\n鉁?Training completed successfully!")
    print(f"馃搧 Log file: {filepath}/Cue_level_Contrastive_training.log")
    print(f"馃捑 Models saved in: {savePath1}/")
