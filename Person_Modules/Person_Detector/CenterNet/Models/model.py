from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
# from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
#from .networks.seg_dla_dcn import get_pose_net as get_dla_segdcn
from .networks.pose_hardnet import get_pose_net as get_hardnet

_model_factory = {
  # 'dla': get_dla_dcn,
#  'dla_seg': get_dla_segdcn,
  'hardnet': get_hardnet,
}

def create_model(task, arch, heads, head_conv):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  if task == 'mpseg' and arch == 'dla':
    get_model = _model_factory['dla_seg']
  else:
    get_model = _model_factory[arch]
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  return model

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}.'.format(
          k, model_state_dict[k].shape, state_dict[k].shape))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k))
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k))
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model