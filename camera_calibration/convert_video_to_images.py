import cv2
import os
import argparse

parser = argparse.ArgumentParser(description="Change video to images")
parser.add_argument("--input", type=str, help="The input video name")
parser.add_argument("--output", type=str, default=None, help="The input video name")

args = parser.parse_args()
input_name = args.input
if args.output is None:
    output_name = input_name.split(".")[0]
else:
    output_name = args.output

if not os.path.exists(output_name):
    os.makedirs(output_name)
    
vidcap = cv2.VideoCapture(input_name)
if not vidcap.isOpened():
    print("Error when opening vidoe stream or file")

def write_frames(vidcap):
    count = 0
    while vidcap.isOpened():
        # vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames,image = vidcap.read()
        if hasFrames:

            savepath = os.path.join(output_name, "image_%05d.jpg" % count)
            cv2.imwrite(savepath, image)     # save frame as JPG file
        count += 1

write_frames(vidcap)