import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'platevision/backend'))
import openvino as ov
from openvino import opset13 as ops
import numpy as np
import cv2
from app.services.detector import OpenVINOPlateDetector

def grid(height,width):
    points=[];scales=[]
    for stride in (8,16,32):
        y,x=np.mgrid[:height//stride,:width//stride]
        points.append(np.stack([x.ravel()+.5,y.ravel()+.5]))
        scales.append(np.full(x.size,stride))
    return np.concatenate(points,axis=1)[None].astype(np.float32),np.concatenate(scales)[None].astype(np.float32)

def resized_model(core,path,height,width):
    model=core.read_model(str(path));old_grid,old_stride=grid(640,640);new_grid,new_stride=grid(height,width)
    changed={'grid':0,'stride':0,'reshape':0}
    for node in model.get_ops():
        if node.get_type_name()!='Constant':continue
        values=np.array(node.data)
        if values.shape==old_grid.shape and np.allclose(values,old_grid):
            replacement=new_grid;changed['grid']+=1
        elif values.shape==old_stride.shape and np.allclose(values,old_stride):
            replacement=new_stride;changed['stride']+=1
        elif values.shape in {(4,),(3,)} and values[-1]==8400 and values[0]==1:
            replacement=values.copy();replacement[-1]=new_grid.shape[-1];changed['reshape']+=1
        else:continue
        node.output(0).replace(ops.constant(replacement).output(0))
    if changed!={'grid':2,'stride':1,'reshape':2}:raise ValueError(changed)
    model.get_parameters()[0].set_partial_shape(ov.PartialShape([1,3,height,width]))
    model.validate_nodes_and_infer_types()
    return model

if __name__=='__main__':
    cv2.setNumThreads(2);core=ov.Core();image=cv2.imread('tests/three-vehicle-composite.jpg')
    for h,w in [(640,640),(736,1280),(864,1536),(1088,1920)]:
        model=resized_model(core,'platevision/models/plate-detector.onnx',h,w)
        p=ov.preprocess.PrePostProcessor(model)
        p.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout('NHWC')).set_color_format(ov.preprocess.ColorFormat.BGR)
        p.input().model().set_layout(ov.Layout('NCHW'));p.input().preprocess().convert_color(ov.preprocess.ColorFormat.RGB).convert_element_type(ov.Type.f32).scale(255.)
        d=object.__new__(OpenVINOPlateDetector);d.compiled=core.compile_model(p.build(),'GPU',{'PERFORMANCE_HINT':'LATENCY'});d.request=d.compiled.create_infer_request();d.height=h;d.width=w;d.classes=1;d.allowed=None
        for _ in range(3):d.detect(image)
        times=[]
        for _ in range(15):
            t=time.perf_counter();boxes=d.detect(image);times.append((time.perf_counter()-t)*1000)
        print(h,w,np.percentile(times,[50,95]),boxes,flush=True)
