# Testing functions.
# author: ynie
# date: April, 2020
from net_utils.utils import LossRecorder
from time import time
import torch
from net_utils.ap_helper import APCalculator
import numpy as np

ScanNet2Cat = {'0':'table','1':'chair','2':'bookshelf','3':'sofa','4':'trash bin','5':'cabinet','6':'display','7':'bathtub' }

def test_func(cfg, tester, test_loader):
    '''
    test function.
    :param cfg: configuration file
    :param tester: specific tester for networks
    :param test_loader: dataloader for testing
    :return:
    '''
    device = torch.device('cuda') if cfg.config['device']['use_gpu'] else torch.device('cpu')
    mode = cfg.config['mode']
    batch_size = cfg.config[mode]['batch_size']
    loss_recorder = LossRecorder(batch_size)
    AP_IOU_THRESHOLDS = cfg.config[mode]['ap_iou_thresholds']
    evaluate_mesh_mAP = True if cfg.config[mode]['phase'] == 'completion' and cfg.config['generation'][
        'generate_mesh'] and cfg.config[mode]['evaluate_mesh_mAP'] else False
    ap_calculator_list = [APCalculator(iou_thresh, cfg.dataset_config.class2type, evaluate_mesh_mAP) for iou_thresh in
                          AP_IOU_THRESHOLDS]
    cfg.log_string('-'*100)
    acc_loss_per_cat = torch.zeros((8,2)).to(device)
    for iter, data in enumerate(test_loader):
        loss, est_data,loss_per_cat = tester.test_step(data)
        dist, labels = loss_per_cat
        if len(labels.size())==0:
            labels = labels.unsqueeze(0)
        for batch_id in range(dist.shape[0]):
            acc_loss_per_cat[labels[batch_id].item()][0] += dist[batch_id].item()
            acc_loss_per_cat[labels[batch_id].item()][1] += 1
        eval_dict = est_data[4]
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(eval_dict['batch_pred_map_cls'], eval_dict['batch_gt_map_cls'])
        # visualize intermediate results.
        if cfg.config['generation']['dump_results']:
            tester.visualize_step(mode, iter, data, est_data, eval_dict)

        loss_recorder.update_loss(loss)

        if ((iter + 1) % cfg.config['log']['print_step']) == 0:
            cfg.log_string('Process: Phase: %s. Epoch %d: %d/%d. Current loss: %s.' % (
            mode, 0, iter + 1, len(test_loader), str({key: np.mean(item) for key, item in loss.items()})))

    acc_loss_per_cat[:,0] /= acc_loss_per_cat[:,1]   
    
    return loss_recorder.loss_recorder, ap_calculator_list,acc_loss_per_cat[:,0]

def test(cfg, tester, test_loader):
    '''
    train epochs for network
    :param cfg: configuration file
    :param tester: specific tester for networks
    :param test_loader: dataloader for testing
    :return:
    '''
    cfg.log_string('-' * 100)
    # set mode
    mode = cfg.config['mode']
    tester.net.train(mode == 'train')
    start = time()
    test_loss_recoder, ap_calculator_list,dist_per_cat = test_func(cfg, tester, test_loader)
    cfg.log_string('Test time elapsed: (%f).' % (time()-start))
    for key, test_loss in test_loss_recoder.items():
        cfg.log_string('Test loss (%s): %f' % (key, test_loss.avg))

    # Evaluate average precision
    AP_IOU_THRESHOLDS = cfg.config[mode]['ap_iou_thresholds']
    for i, ap_calculator in enumerate(ap_calculator_list):
        cfg.log_string(('-'*10 + 'iou_thresh: %f' + '-'*10) % (AP_IOU_THRESHOLDS[i]))
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            cfg.log_string('eval %s: %f' % (key, metrics_dict[key]))
            
    cfg.log_string('---average chamfer distances per category---') 
    for i in range(8):
        cfg.log_string(f'{ScanNet2Cat[str(i)]:15} : {dist_per_cat[i]:10f}')
