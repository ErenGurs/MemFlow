from __future__ import print_function, division
import argparse
from loguru import logger as loguru_logger
import random
from core.Networks import build_network
import sys
sys.path.append('core')
from PIL import Image
import PIL
#import PIL.Image
import torchvision
import os
import numpy as np
import torch
from utils import flow_viz
from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate
from core.utils import backwarp
from inference import inference_core_skflow as inference_core
from process_data import FlowUtils

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def inference(cfg):
    model = build_network(cfg).cuda()
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        ckpt = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
        if 'module' in list(ckpt_model.keys())[0]:
            for key in ckpt_model.keys():
                ckpt_model[key.replace('module.', '', 1)] = ckpt_model.pop(key)
            model.load_state_dict(ckpt_model, strict=True)
        else:
            model.load_state_dict(ckpt_model, strict=True)

    model.eval()

    print(f"preparing image...")
    print(f"Input image sequence dir = {cfg.seq_dir}")
    image_list = sorted(os.listdir(cfg.seq_dir))
    
    # Eren: Make it similar to PWCNet scripts
    if ( args.firstfile != None and args.secondfile != None):
        image_list = [args.firstfile, args.secondfile]

    imgs = [frame_utils.read_gen(os.path.join(cfg.seq_dir, path)) for path in image_list]
    imgs = [np.array(img).astype(np.uint8) for img in imgs]
    # grayscale images
    if len(imgs[0].shape) == 2:
        imgs = [np.tile(img[..., None], (1, 1, 3)) for img in imgs]
    else:
        imgs = [img[..., :3] for img in imgs]
    imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]

    images = torch.stack(imgs)

    processor = inference_core.InferenceCore(model, config=cfg)
    # 1, T, C, H, W
    images = images.cuda().unsqueeze(0)

    padder = InputPadder(images.shape)
    images = padder.pad(images)

    images = 2 * (images / 255.0) - 1.0
    flow_prev = None
    results = []
    print(f"start inference...")
    for ti in range(images.shape[1] - 1):
        print (f"Processing {ti} of {images.shape[1]} frames", end="\r")
        flow_low, flow_pre = processor.step(images[:, ti:ti + 2], end=(ti == images.shape[1] - 2),
                                            add_pe=('rope' in cfg and cfg.rope), flow_init=flow_prev)
        flow_pre = padder.unpad(flow_pre[0]).cpu()
        results.append(flow_pre)
        if 'warm_start' in cfg and cfg.warm_start:
            flow_prev = forward_interpolate(flow_low[0])[None].cuda()

    if not os.path.exists(cfg.vis_dir):
        os.makedirs(cfg.vis_dir)

    # Use FlowUtils package
    fu = FlowUtils()

    print(f"save results...")
    N = len(results)
    for idx in range(N):
        print (f"Saving {idx} of {N} frames", end="\r")
        flow_img = flow_viz.flow_to_image(results[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save('{}/flow_{:04}_to_{:04}.png'.format(cfg.vis_dir, idx + 1, idx + 2))

        #bkwrp = backwarp.ModuleBackwarp()
        # Backwarp second image + Denormalize
        backwarped_tensor = (backwarp.function_backwarp(images[:,idx+1], results[idx].cuda().unsqueeze(dim=0)) + 1.0) / 2
        torchvision.utils.save_image(backwarped_tensor, cfg.vis_dir + '/' + f"backwarp_{idx+1}.png")
        
        #backwarped_frame = Image.fromarray(
        #np.array(torch.squeeze(backwarped_tensor.cpu())).transpose(1, 2, 0).astype(np.uint8))
        #backwarped_frame.save(cfg.vis_dir + '/' + 'backwarp.png')


        fu.write_flo(np.array(torch.squeeze(results[idx].cpu())).transpose(1, 2, 0),
                  '{}/flow_{:04}_to_{:04}.flo'.format(cfg.vis_dir, idx + 1, idx + 2))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='MemFlowNet', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--firstfile', type=str, help='first image in seq_dir (optional)')
    parser.add_argument('--secondfile', type=str, help='second image in seq_dir (optional)')

    parser.add_argument('--seq_dir', help="folder for input images. If there are subfolders, it will loop through them", default='default')
    parser.add_argument('--vis_dir', help="output folder which will follow the structure of input seq_dir", default='default')

    args = parser.parse_args()

    if args.stage == 'things':
        from configs.things_memflownet import get_cfg
    elif args.stage == 'sintel':
        from configs.sintel_memflownet import get_cfg
    elif args.stage == 'sintel_t':  # Transformer based MemFlow trained on Sintel
        from configs.sintel_memflownet_t import get_cfg
    elif args.stage == 'spring_only':
        from configs.spring_memflownet import get_cfg
    elif args.stage == 'kitti':
        from configs.kitti_memflownet import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))

    # initialize random seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)


    orig_seq_dir = cfg.seq_dir
    orig_vis_dir = cfg.vis_dir
    dirList = sorted(os.scandir(args.seq_dir), key=lambda e: e.name)
    #for entry in os.scandir(args.seq_dir):
    for entry in dirList:
        if entry.name.startswith('.'):
            continue
        # If seq_dir includes other directories (like processed BMS data), iterate each directory
        if entry.is_dir():
            print ("Directory: ", entry.name)
            cfg.seq_dir = orig_seq_dir + '/' + entry.name
            cfg.vis_dir = orig_vis_dir + '/' + entry.name

            inference(cfg)
        else:
            print (entry.name)
            inference(cfg)
            break

    #inference(cfg)
