import torch.nn as nn
import pprint
import torch
from nets.utils.layer_tools import C3,Conv,Classify 
from nets.utils.struct_cfg import yolov5_cfg

class yolov5_cls_st(nn.Module):
    def __init__(self,cfg:dict):
        super(yolov5_cls_st, self).__init__()
        self.cfg = cfg
        self.feature_ = self._build_feature_map()
        self.cls = self._build_cls('1')
    def get_name(self):   #!^
         return "yolov5_cls_st"
    def _build_feature_map(self):
        feature_map = nn.ModuleList()
        # feature_map = []
        feature_map_cfg = self.cfg['feature_map']
        feature_map.append(Conv(*feature_map_cfg[0][1]))    
        feature_map.append(Conv(*feature_map_cfg[1][1])) 
        feature_map.extend([C3(*feature_map_cfg[2][1])]*feature_map_cfg[2][0][0]) 
        feature_map.append(Conv(*feature_map_cfg[3][1])) 
        feature_map.extend([C3(*feature_map_cfg[4][1])]*feature_map_cfg[4][0][0]) 
        feature_map.append(Conv(*feature_map_cfg[5][1])) 
        feature_map.extend([C3(*feature_map_cfg[6][1])]*feature_map_cfg[6][0][0]) 
        feature_map.append(Conv(*feature_map_cfg[7][1])) 
        feature_map.extend([C3(*feature_map_cfg[8][1])]*feature_map_cfg[8][0][0]) 
        return nn.Sequential(*feature_map)

    def _build_cls(self,indx):
        classify = nn.Sequential()
        classify_cfg = self.cfg['classify']
        classify.add_module(f'cls_{indx}',Classify(*classify_cfg))
        return classify
    
    def forward(self,x):
        x = self.feature_(x)
        out1 = self.cls(x)
        return out1


#~ s meaning st-struct
def yolov5n_cls_st(num_cls1):
    cfg = yolov5_cfg(num_classes1=num_cls1).yolov5_n()
    return yolov5_cls_st(cfg)

def yolov5s_cls_st(num_cls1):
    cfg = yolov5_cfg(num_classes1=num_cls1).yolov5_s()
    return yolov5_cls_st(cfg)

def yolov5l_cls_st(num_cls1):
    cfg = yolov5_cfg(num_classes1=num_cls1).yolov5_l()
    return yolov5_cls_st(cfg)
     
def yolov5m_cls_st(num_cls1):
    cfg = yolov5_cfg(num_classes1=num_cls1).yolov5_m()
    return yolov5_cls_st(cfg)




