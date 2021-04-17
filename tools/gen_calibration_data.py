#!/usr/bin/env python
# -*- coding=utf-8 -*-
import argparse
import numpy as np
import os
import glob
import struct
import random
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--inDir", required=True, help="Input directory")
parser.add_argument("--outDir", required=True, help="Output directory")
parser.add_argument("--batch", type=int, default=16, help="INPUT_N")
parser.add_argument("--channel", type=int, default=3, help="INPUT_C")
parser.add_argument("--height", type=int, default=224, help="INPUT_H")
parser.add_argument("--width", type=int, default=224, help="INPUT_W")
parser.add_argument("--num_calibration_images", default=0, help="NUM_CALIBRATION_IMAGES")
parser.add_argument("--mean", default=(0, 0, 0), help="NUM_CALIBRATION_IMAGES")
parser.add_argument("--scale", default=(1, 1, 1), help="NUM_CALIBRATION_IMAGES")

args = parser.parse_args()

CALIBRATION_DATASET = args.inDir + "/*.jpg"

# images to test
print("Location of dataset = " + CALIBRATION_DATASET)
images = glob.glob(CALIBRATION_DATASET)
random.shuffle(images)
if args.num_calibration_images == 0:
    args.num_calibration_images = len(images)
images = images[:args.num_calibration_images]
num_batches = args.num_calibration_images // args.batch + (args.num_calibration_images % args.batch > 0)

print("Total number of images = " + str(len(images)))
print("NUM_PER_BATCH = " + str(args.batch))
print("NUM_BATCHES = " + str(num_batches))

# output
outDir = args.outDir + "/batches_{}x{}x{}x{}".format(args.batch, args.channel, args.height, args.width)

if os.path.exists(outDir):
    os.system("rm " + outDir + "/*")

# prepare output
if not os.path.exists(outDir):
    os.makedirs(outDir)

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
count = 0
for i in range(num_batches):
    batch_file = outDir + "/batch" + str(i)
    batch = np.zeros(shape=(args.batch, args.channel, args.height, args.width), dtype=np.float32)
    for j in range(args.batch):
        im = cv2.imread(images[count])
        im = cv2.resize(im, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
        in_ = np.array(im, dtype=np.float32)
        in_ -= np.array(args.mean)
        in_ *= np.array(args.scale)
        in_ = in_.transpose((2, 0, 1))

        # assert batch.shape[1] >= 3 and batch.shape[1] % 3 == 0
        # num = batch.shape[1] // 3
        # for idx in range(num):
        #     batch[j, 3*idx:3*idx+3, :, :] = in_
        batch[j] = in_
        count += 1

    # save
    batch.tofile(batch_file)

    # Prepend batch shape information
    ba = bytearray(struct.pack("4i", batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]))

    with open(batch_file, "rb+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(ba)
        f.write(content)

    print(batch_file)
