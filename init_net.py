"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
from collections import OrderedDict
import argparse
import time
import json
import os

import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import torch
import cv2

import craft_model.craft_utils as craft_utils
import craft_model.imgproc as  imgproc
import craft_model.file_utils as file_utils
from craft_model.craft import CRAFT


args: dict = None
with open("config.json", "r") as file:
    args = json.load(file)
assert args is not None


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args["canvas_size"], interpolation=cv2.INTER_LINEAR, mag_ratio=args["mag_ratio"])
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args["show_time"]:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def process_image(net, image, refine_net):
    bboxes, polys, score_text = test_net(net, image, args["text_threshold"], args["link_threshold"], args["low_text"], args["cuda"], args["poly"], refine_net)
    return bboxes, polys, score_text

def craft_factory():
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args["trained_model"] + ')')
    if args["cuda"]:
        net.load_state_dict(copyStateDict(torch.load(args["trained_model"])))
    else:
        net.load_state_dict(copyStateDict(torch.load(args["trained_model"], map_location='cpu')))

    if args["cuda"]:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args["refine"]:
        from craft_model.refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args["refiner_model"] + ')')
        if args["cuda"]:
            refine_net.load_state_dict(copyStateDict(torch.load(args["refiner_model"])))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args["refiner_model"], map_location='cpu')))

        refine_net.eval()
        args["poly"] = True
    
    return net, refine_net

if __name__=="__main__":
    net, refine_net = craft_factory()
    image = imgproc.loadImage("./testing/1.png")
    bboxes, polys, score_text = process_image(net, image, refine_net)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for box in bboxes:
        for point_prev, point_next in zip(box[0:-1], box[1:]):
            point_prev_int = list(map(round, point_prev))
            point_next_int = list(map(round, point_next))
            cv2.line(image_bgr, point_prev_int, point_next_int, (0, 0, 255))
    
    cv2.imshow("img", image_bgr)
    cv2.waitKey(0)