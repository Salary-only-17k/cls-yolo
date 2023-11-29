import torch
import sys
import numpy as np
import pathlib
import os
sys.path.append("..")

from nets.yolov5_62cls import yolov5n_cls_st,yolov5s_cls_st,yolov5l_cls_st,yolov5m_cls_st

class pt2onnx:
    def __init__(self,ptDir,inputSz=[1, 3,224,224]) -> None:
        self.ptDir = ptDir
    
        self.inputSz =inputSz
    def core(self,pth):
        file_name = os.path.basename(pth).replace(".pt",".onnx")
        model = torch.load(pth,map_location='cpu')['model']
        model.eval()
        input_data = np.random.randn(*self.inputSz).astype(np.float32)
        dummy_input =  torch.from_numpy(input_data)
        output = model(dummy_input)
        print(f"output:  ",output)
        # torch.onnx.export(model, dummy_input, os.path.join(self.onnxDir, file_name), verbose=True,opset_version=12)
    def loop(self):
        ptPth_lst = [str(p) for p in list(pathlib.Path(self.ptDir).glob("*.pt"))]
        for ptPth in ptPth_lst:
            self.core(ptPth)


if __name__ == "__main__":
    demo = pt2onnx(r"D:\code\cls-yolo\runs\train\sh_2023_11_13-165341")    
    demo.loop()