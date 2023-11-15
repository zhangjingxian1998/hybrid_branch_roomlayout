from models.heads import HRMerge_fpn_jiangwei
from models.hr_cfg import model_cfg
from models.hrnet import HighResolutionNet
import pytorch_lightning as pl
from models.loss import Loss
from datasets import SUNRGBD, Structured3D, NYU303
import torch
from models.utils import (test_evaluate,evaluate, get_optimizer,gt_check, post_process, _sigmoid)
from models.reconstruction import (ConvertLayout,Reconstruction)
import numpy as np
import yaml
from easydict import EasyDict
from models.reinforce import reinforce_attention
class Detector(pl.LightningModule):
    def __init__(self):
        super().__init__()
        with open('cfg.yaml', 'r') as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
            cfg = EasyDict(config)
        self.cfg = cfg
        self.result = []
        extra = model_cfg['backbone']['extra']
        pretrained = model_cfg['pretrained']
        self.backbone = HighResolutionNet(extra)
        self.merge = HRMerge_fpn_jiangwei()

        self.reiforce = reinforce_attention()

        self.init_weights(pretrained=pretrained)

    def forward(self, x):
        x = self.backbone(x)
        x = self.merge(x)

        x = self.reiforce(x)
        #out = {
        #     'plane_center': plane_center,
        #     'plane_offset': plane_xy,
        #     'plane_wh': plane_wh,
        #     'plane_params_pixelwise': plane_params_pixelwise,
        #     'plane_params_instance': plane_params_instance,
        #     'line_region': line_region,
        #     'line_params': line_params,
        #     'feature': x
        # }
        return x
    def training_step(self, inputs, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        # return None
        for key, value in inputs.items():
            if key=='img_name':
                    continue
            inputs[key] = value.to('cuda')
        # forward
        x = self.forward(inputs['img'])
        loss, loss_stats = self.criterion(x, **inputs)
        self.log('plane_hm_loss', loss_stats['plane_hm_loss'])
        self.log('plane_wh_loss', loss_stats['plane_wh_loss'])
        self.log('plane_offset_loss', loss_stats['plane_offset_loss'])
        self.log('plane_param_loss', loss_stats['plane_param_loss'])
        self.log('plane_param_i_loss', loss_stats['plane_param_i_loss'])
        self.log('plane_pixelwise_depth_loss', loss_stats['plane_pixelwise_depth_loss'])
        self.log('plane_instance_depth_loss', loss_stats['plane_instance_depth_loss'])
        self.log('line_hm_loss', loss_stats['line_hm_loss'])
        self.log('line_offset_loss', loss_stats['line_offset_loss'])
        self.log('line_alpha_loss', loss_stats['line_alpha_loss'])
        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, inputs, batch_index):
        global best_acc
        global best_info
        dts_planes = []
        dts_lines = []
        gts_planes = []
        gts_lines = []
        for key, value in inputs.items():
            if key=='img_name':
                    continue
            inputs[key] = value.to('cuda')
        # forward
        x = self.forward(inputs['img'])

        x['plane_center'] = _sigmoid(x['plane_center'])
        x['line_region'] = _sigmoid(x['line_region'])
        x['line_params'][:, 1:2] = _sigmoid(x['line_params'][:, 1:2])

        # loss = self.criterion(x, **inputs)
        dt_planes, dt_lines, dt_params3d, _ = post_process(x)
        gt_planes, gt_lines, gt_params3d = gt_check(inputs)
        dts_planes.extend(dt_planes)  # each img topk dt planes
        gts_planes.extend(gt_planes)
        dts_lines.extend([dt[dt[:, 3] == 1] for dt in dt_lines])  # each img has variable number of dt lines
        gts_lines.extend([gt[gt[:, 3] == 1] for gt in gt_lines])

        mAR_p, mAP_p, mAR_l, mAP_l = evaluate(dts_planes, dts_lines, gts_planes, gts_lines)

        best_acc = mAP_p + mAP_l
        # # Logging to TensorBoard by default
        self.log('mAR_p', mAR_p,batch_size=1)
        self.log('mAP_p', mAP_p,batch_size=1)
        self.log('mAR_l', mAR_l,batch_size=1)
        self.log('mAP_l', mAP_l,batch_size=1)
        self.log('best_acc', best_acc,batch_size=1)

    def test_step(self, inputs, batch_index):
        self.dirs = inputs['dirs']
        self.dir = inputs['dir'][0]
        if batch_index % 1000 == 0:
            print(batch_index)

        if self.cfg.data == 'Structured3D':
            # forward
            for key, value in inputs.items():
                if key=='img_name' or key == 'dirs' or key == 'dir':
                        continue
                inputs[key] = value.to('cuda')
            
            x = self.forward(inputs['img'])
            # if batch_index>6270:
            #     torch.save(set_xishu,'weight_save.pt')
            x['plane_center'] = _sigmoid(x['plane_center'])
            x['line_region'] = _sigmoid(x['line_region'])
            x['line_params'][:, 1:2] = _sigmoid(x['line_params'][:, 1:2])
            # loss, loss_stats = self.criterion(x, **inputs)

            # post process on output feature map size and extract plane and line detection results
            dt_planes, dt_lines, dt_params3d_instance, dt_params3d_pixelwise = post_process(x, Mnms=1)
            # dt_planes (x y x y score cls) 应该是第一个x,y是平面的左上角位置，第二个x,y是平面右下角位置，然后是置信度和分类
            # dt_lines (m,b,score,1)
            for i in range(1):
                # generate layout with a post-process according to detection results
                (_ups, _downs, _attribution, _params_layout), (ups, downs, attribution, params_layout), (pfloor, pceiling) = Reconstruction(
                    dt_planes[i],
                    dt_params3d_instance[i],
                    dt_lines[i],
                    K=inputs['intri'][i].cpu().numpy(),
                    size=(720, 1280),
                    threshold=(0.3, 0.3, 0.3, 0.3))

                # convert no opt results to segmentation and depth map and evaluate results
                _seg, _depth, _, _polys = ConvertLayout(
                    inputs['img'][i], _ups, _downs, _attribution,
                    K=inputs['intri'][i].cpu().numpy(), pwalls=_params_layout,
                    pfloor=pfloor, pceiling=pceiling,
                    ixy1map=inputs['ixy1map'][i].cpu().numpy(),
                    valid=inputs['iseg'][i].cpu().numpy(),
                    oxy1map=inputs['oxy1map'][i].cpu().numpy(), pixelwise=None)

                _res = test_evaluate(inputs['iseg'][i].cpu().numpy(),
                                inputs['idepth'][i].cpu().numpy(), _seg, _depth)

                # convert opt results to segmentation and depth map and evaluate results
                seg, depth, img, polys = ConvertLayout(
                    inputs['img'][i], ups, downs, attribution,
                    K=inputs['intri'][i].cpu().numpy(), pwalls=params_layout,
                    pfloor=pfloor, pceiling=pceiling,
                    ixy1map=inputs['ixy1map'][i].cpu().numpy(),
                    valid=inputs['iseg'][i].cpu().numpy(),
                    oxy1map=inputs['oxy1map'][i].cpu().numpy(), pixelwise=None)

                res = test_evaluate(inputs['iseg'][i].cpu().numpy(),
                            inputs['idepth'][i].cpu().numpy(), seg, depth)

                self.log("iou:",res[0])
                self.log("pe:",res[1])
                self.log("edge:",res[2])
                self.log("rmse:",res[3])
                self.log("us_rmse:",res[4])
                self.log("pe_hung:",res[5])

                self.log("_iou:",_res[0])
                self.log("_pe:",_res[1])
                self.log("_edge:",_res[2])
                self.log("_rmse:",_res[3])
                self.log("_us_rmse:",_res[4])
                self.log("_pe_hung:",_res[5])

        elif self.cfg.data == 'NYU303':
            self.dirs = inputs['dirs']
            for key, value in inputs.items():
                if key=='img_name':
                    continue
                inputs[key] = value.to('cuda')
            # forward

            x = self.forward(inputs['img'])
            x['plane_center'] = _sigmoid(x['plane_center'])
            x['line_region'] = _sigmoid(x['line_region'])
            x['line_params'][:, 1:2] = _sigmoid(x['line_params'][:, 1:2])

            # post process on output feature map size
            dt_planes, dt_lines, dt_params3d_instance, dt_params3d_pixelwise = post_process(x, Mnms=1)
            # convert sunrgbd crop image to fullres img
            dt_planes[:, :, :4] = dt_planes[:, :, :4] + \
                np.array([41, 45, 41, 45]) / 4.
            dt_lines[:, :, 1] = dt_lines[:, :, 1] + \
                41/4. - dt_lines[:, :, 0] * 45/4.

            # reconstruction
            for i in range(1):
                (_ups, _downs, _attribution, _params_layout), (ups, downs, attribution, params_layout), (pfloor, pceiling) = Reconstruction(
                    dt_planes[i], 
                    dt_params3d_instance[i], 
                    dt_lines[i],
                    K=inputs['full_intri'][i].cpu().numpy(), 
                    size=(480, 640), 
                    threshold=(0.12, 0.1, 0.1, 0.2), 
                    downsample=4)

                # no opt
                _seg, _depth, img, _ = ConvertLayout(inputs['fullimg'][i], _ups, _downs, _attribution,
                                                    K=inputs['full_intri'][i].cpu().numpy(), pwalls=_params_layout,
                                                    pfloor=pfloor, pceiling=pceiling,
                                                    ixy1map=inputs['ixy1map'][i].cpu().numpy(), valid=inputs['iseg'][i].cpu().numpy())
                _res = test_evaluate(inputs['iseg'][i].cpu().numpy(), inputs['idepth'][i].cpu().numpy(), _seg, _depth,evaluate_3D=False)
                
                # opt
                seg, depth, _, _ = ConvertLayout(inputs['fullimg'][i], ups, downs, attribution,
                                                K=inputs['full_intri'][i].cpu().numpy(), pwalls=params_layout,
                                                pfloor=pfloor, pceiling=pceiling,
                                                ixy1map=inputs['ixy1map'][i].cpu().numpy(), valid=inputs['iseg'][i].cpu().numpy())
                res = test_evaluate(inputs['iseg'][i].cpu().numpy(), inputs['idepth'][i].cpu().numpy(), seg, depth,evaluate_3D=False)

                self.log("iou:",res[0])
                self.log("pe:",res[1])
                self.log("edge:",res[2])
                self.log("rmse:",res[3])
                self.log("us_rmse:",res[4])
                self.log("_pe_hung:",res[5])

                self.log("iou_wo:",_res[0])
                self.log("pe_wo:",_res[1])
                self.log("edge_wo:",_res[2])
                self.log("rmse_wo:",_res[3])
                self.log("us_rmse_wo:",_res[4])
                self.log("_pe_hung_wo:",_res[5])
        elif self.cfg.data == 'Hedau':
            for key, value in inputs.items():
                if key=='img_name':
                    continue
                inputs[key] = value.to('cuda')
            x = self.forward(inputs['img'])
            x['plane_center'] = _sigmoid(x['plane_center'])
            x['line_region'] = _sigmoid(x['line_region'])
            x['line_params'][:, 1:2] = _sigmoid(x['line_params'][:, 1:2])
            dt_planes, dt_lines, dt_params3d_instance, dt_params3d_pixelwise = post_process(x, Mnms=1)
            for i in range(1):
                (_ups, _downs, _attribution, _params_layout), (ups, downs, attribution, params_layout), (pfloor, pceiling) = Reconstruction(
                    dt_planes[i], 
                    dt_params3d_instance[i], 
                    dt_lines[i],
                    K=inputs['full_intri'][i].cpu().numpy(), 
                    size=(480, 640), 
                    threshold=(0.3, 0.1, 0.1, 0.3), 
                    downsample=4)

                # no opt
                _seg, _depth, img, _ = ConvertLayout(inputs['img'][i], _ups, _downs, _attribution,
                                                    K=inputs['full_intri'][i].cpu().numpy(), pwalls=_params_layout,
                                                    pfloor=pfloor, pceiling=pceiling,
                                                    ixy1map=inputs['ixy1map'][i].cpu().numpy(), valid=inputs['iseg'][i].cpu().numpy())
                _res = test_evaluate(inputs['iseg'][i].cpu().numpy(), inputs['idepth'][i].cpu().numpy(), _seg, _depth,evaluate_3D=False)
                
                # opt
                seg, depth, _, _ = ConvertLayout(inputs['img'][i], ups, downs, attribution,
                                                K=inputs['full_intri'][i].cpu().numpy(), pwalls=params_layout,
                                                pfloor=pfloor, pceiling=pceiling,
                                                ixy1map=inputs['ixy1map'][i].cpu().numpy(), valid=inputs['iseg'][i].cpu().numpy())
                res = test_evaluate(inputs['iseg'][i].cpu().numpy(), inputs['idepth'][i].cpu().numpy(), seg, depth,evaluate_3D=False)
                
                ###############################################################
                self.log("iou:",res[0])
                self.log("pe:",res[1])
                self.log("edge:",res[2])
                self.log("rmse:",res[3])
                self.log("us_rmse:",res[4])
                self.log("_pe_hung:",res[5])

                self.log("iou_wo:",_res[0])
                self.log("pe_wo:",_res[1])
                self.log("edge_wo:",_res[2])
                self.log("rmse_wo:",_res[3])
                self.log("us_rmse_wo:",_res[4])
                self.log("_pe_hung_wo:",_res[5])
        elif self.cfg.data == 'NYU303_ALL':
            pass
        elif self.cfg.data == 'CUSTOM':
            pass
        elif self.cfg.data == 'SUNRGBD':
            global best_acc
            global best_info
            dts_planes = []
            dts_lines = []
            gts_planes = []
            gts_lines = []
            for key, value in inputs.items():
                if key=='img_name':
                        continue
                inputs[key] = value.to('cuda')
            # forward
            x = self.forward(inputs['img'])


            # time_now = time.localtime(time.time())
            # if time_now[3]==12:
            #     time.sleep(1)


            x['plane_center'] = _sigmoid(x['plane_center'])
            x['line_region'] = _sigmoid(x['line_region'])
            x['line_params'][:, 1:2] = _sigmoid(x['line_params'][:, 1:2])

            # loss = self.criterion(x, **inputs)
            dt_planes, dt_lines, dt_params3d, _ = post_process(x)
            gt_planes, gt_lines, gt_params3d = gt_check(inputs)
            dts_planes.extend(dt_planes)  # each img topk dt planes
            gts_planes.extend(gt_planes)
            dts_lines.extend([dt[dt[:, 3] == 1] for dt in dt_lines])  # each img has variable number of dt lines
            gts_lines.extend([gt[gt[:, 3] == 1] for gt in gt_lines])
            # dt_planes = np.array(dt_planes)
            # gts_planes = np.array(gts_planes)
            # dts_lines = np.array(dts_lines)
            # gts_lines = np.array(gts_lines)
            mAR_p, mAP_p, mAR_l, mAP_l = evaluate(dts_planes, dts_lines, gts_planes, gts_lines)

            best_acc = mAP_p + mAP_l
            # # Logging to TensorBoard by default
            self.log('mAR_p', mAR_p,batch_size=1)
            self.log('mAP_p', mAP_p,batch_size=1)
            self.log('mAR_l', mAR_l,batch_size=1)
            self.log('mAP_l', mAP_l,batch_size=1)
            self.log('best_acc', best_acc,batch_size=1)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained)
        self.merge.init_weights()

    def criterion(self,pred,**target):
        criterion = Loss(self.cfg.Weights)
        # criterion = nn.CrossEntropyLoss()
        loss, loss_stats = criterion(pred,**target)
        return loss, loss_stats

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.cfg.Solver)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.cfg.lr_step)
        return ([optimizer],[scheduler])

    def train_dataloader(self):
        if self.cfg.data == 'Structured3D':
            dataset = Structured3D(self.cfg.Dataset.Structured3D, 'training')
            self.data_length = dataset.__len__()
            return torch.utils.data.DataLoader(dataset,batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers,drop_last=True,pin_memory=True)
        elif self.cfg.data == 'SUNRGBD':
            dataset = SUNRGBD(self.cfg.Dataset.SUNRGBD, 'train',split='all')
            self.data_length = dataset.__len__()
            return torch.utils.data.DataLoader(dataset,batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers,drop_last=True,pin_memory=True)
        elif self.cfg.data == 'NYU303':
            dataset = SUNRGBD(self.cfg.Dataset.SUNRGBD, 'train',split='nyu')
            self.data_length = dataset.__len__()
            return torch.utils.data.DataLoader(dataset,batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers,drop_last=True,pin_memory=True)
    
    def val_dataloader(self):
        if self.cfg.data == 'Structured3D':
            dataset = Structured3D(self.cfg.Dataset.Structured3D, 'validation')
            return torch.utils.data.DataLoader(dataset,batch_size=1, shuffle=False, num_workers=self.cfg.num_workers,pin_memory=True)
        elif self.cfg.data == 'SUNRGBD':
            dataset = SUNRGBD(self.cfg.Dataset.SUNRGBD, 'test')
            return torch.utils.data.DataLoader(dataset,batch_size=1, shuffle=False, num_workers=self.cfg.num_workers,pin_memory=True)
        elif self.cfg.data == 'NYU303':
            dataset = SUNRGBD(self.cfg.Dataset.SUNRGBD, 'test')
            return torch.utils.data.DataLoader(dataset,batch_size=1, shuffle=False, num_workers=self.cfg.num_workers,pin_memory=True)
    
    def test_dataloader(self):
        # if self.cfg.exam:
        #     assert self.cfg.data == 'NYU303', 'provide one example of nyu303 to test'
    #  dataset
        if self.cfg.data == 'Structured3D':
            dataset = Structured3D(self.cfg.Dataset.Structured3D, 'test')
        elif self.cfg.data == 'NYU303':
            dataset = NYU303(self.cfg.Dataset.NYU303, 'test')
        elif self.cfg.data == 'SUNRGBD':
            dataset = SUNRGBD(self.cfg.Dataset.SUNRGBD, 'test')
        else:
            raise NotImplementedError
        return torch.utils.data.DataLoader(dataset, num_workers=self.cfg.num_workers)
    
    def add_extra_args(self,args):
        self.cfg.update(vars(args))
