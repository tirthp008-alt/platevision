"""Rebuild the fixed YOLOv8 export's spatial grid, without changing weights.

Only the verified single-class 640 px export is accepted. Other architectures
retain the ordinary tiled detector rather than guessing their output layout.
"""
import numpy as np


def anchor_grid(height,width):
    if height%32 or width%32:
        raise ValueError('Detector dimensions must be multiples of 32.')
    points=[];scales=[]
    for stride in (8,16,32):
        y,x=np.mgrid[:height//stride,:width//stride]
        points.append(np.stack([x.ravel()+.5,y.ravel()+.5]))
        scales.append(np.full(x.size,stride))
    return np.concatenate(points,axis=1)[None].astype(np.float32),np.concatenate(scales)[None].astype(np.float32)


def resize_plate_model(model,height,width):
    import openvino as ov
    from openvino import opset13 as ops
    if list(model.input(0).shape)!=[1,3,640,640] or list(model.output(0).shape)!=[1,5,8400]:
        raise ValueError('High-resolution adapter requires the verified YOLOv8 plate export.')
    old_grid,old_stride=anchor_grid(640,640)
    new_grid,new_stride=anchor_grid(height,width)
    replacements=[];counts={'grid':0,'stride':0,'reshape':0}
    for node in model.get_ops():
        if node.get_type_name()!='Constant':continue
        value=np.array(node.data)
        if value.shape==old_grid.shape and np.array_equal(value,old_grid):
            replacement=new_grid;counts['grid']+=1
        elif value.shape==old_stride.shape and np.array_equal(value,old_stride):
            replacement=new_stride;counts['stride']+=1
        elif value.shape in {(4,),(3,)} and value[-1]==8400 and value[0]==1:
            replacement=value.copy();replacement[-1]=new_grid.shape[-1];counts['reshape']+=1
        else:continue
        replacements.append((node,replacement))
    if counts!={'grid':2,'stride':1,'reshape':2}:
        raise ValueError(f'Unsupported detection head constants: {counts}')
    for node,value in replacements:
        node.output(0).replace(ops.constant(value).output(0))
    model.get_parameters()[0].set_partial_shape(ov.PartialShape([1,3,height,width]))
    model.validate_nodes_and_infer_types()
    return model
