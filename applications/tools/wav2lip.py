#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import argparse
import pandas as pd
import os
import time
import cv2
import numpy as np
import paddle
from ppgan.apps.wav2lip_predictor import Wav2LipPredictor

parser = argparse.ArgumentParser(
    description=
    'Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path',
                    type=str,
                    help='Name of saved checkpoint to load weights from',
                    default=None)
parser.add_argument(
    '--audio',
    type=str,
    help='Filepath of video/audio file to use as raw audio source')
parser.add_argument('--face',
                    type=str,
                    help='Filepath of video/image that contains faces to use')
parser.add_argument('--outfile',
                    type=str,
                    help='Video path to save result. See default for an e.g.',
                    default='results/result_voice.mp4')

parser.add_argument(
    '--static',
    type=bool,
    help='If True, then use only first video frame for inference',
    default=False)
parser.add_argument(
    '--fps',
    type=float,
    help='Can be specified only if input is a static image (default: 25)',
    default=25.,
    required=False)

parser.add_argument(
    '--pads',
    nargs='+',
    type=int,
    default=[0, 10, 0, 0],
    help=
    'Padding (top, bottom, left, right). Please adjust to include chin at least'
)

parser.add_argument('--face_det_batch_size',
                    type=int,
                    help='Batch size for face detection',
                    default=32)
parser.add_argument('--wav2lip_batch_size',
                    type=int,
                    help='Batch size for Wav2Lip model(s)',
                    default=32)

parser.add_argument(
    '--resize_factor',
    default=1,
    type=int,
    help=
    'Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p'
)

parser.add_argument(
    '--crop',
    nargs='+',
    type=int,
    default=[0, -1, 0, -1],
    help=
    'Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
    'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width'
)

parser.add_argument(
    '--box',
    nargs='+',
    type=int,
    default=[-1, -1, -1, -1],
    help=
    'Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).'
)

parser.add_argument(
    '--rotate',
    default=False,
    action='store_true',
    help=
    'Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
    'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument(
    '--nosmooth',
    default=False,
    action='store_true',
    help='Prevent smoothing face detections over a short temporal window')
parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
parser.add_argument(
    "--face_detector",
    dest="face_detector",
    type=str,
    default='sfd',
    help="face detector to be used, can choose s3fd or blazeface")
parser.add_argument("--face_enhancement",
                    dest="face_enhancement",
                    action="store_true",
                    help="use face enhance for face")

parser.add_argument("--csv", type=str)
parser.add_argument("--csvroot", type=str)
parser.add_argument("--outroot", type=str)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=-1)
parser.add_argument('--sort_by_duration', action='store_true', help='T')
parser.add_argument('--video_root', type=str)
parser.add_argument('--audio_root', type=str)
parser.add_argument('--save_duration_path', type=str)

# Specific dubbing args
# parser.add_argument('--outfile', type=str) # this is already defined
parser.add_argument("--outfileFaceOnly", type=str)
parser.add_argument('--copyfile', type=str)
parser.add_argument("--copyfileFaceOnly", type=str)
parser.add_argument('--mergeFace', type=str)
parser.add_argument("--mergeFullFrame", type=str)

parser.set_defaults(face_enhancement=False)

def get_frames(file):
    video_stream = cv2.VideoCapture(file)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    frames = []

    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)

    return frames

if __name__ == "__main__":
    args = parser.parse_args()
    if args.cpu:
        paddle.set_device('cpu')

    predictor = Wav2LipPredictor(checkpoint_path=args.checkpoint_path,
                                 static=args.static,
                                 fps=args.fps,
                                 pads=args.pads,
                                 face_det_batch_size=args.face_det_batch_size,
                                 wav2lip_batch_size=args.wav2lip_batch_size,
                                 resize_factor=args.resize_factor,
                                 crop=args.crop,
                                 box=args.box,
                                 rotate=args.rotate,
                                 nosmooth=args.nosmooth,
                                 face_detector=args.face_detector,
                                 face_enhancement=args.face_enhancement)

    if args.csv:
        print("Using CSV")
        videos_file = pd.read_csv(args.csv)
        if args.sort_by_duration:
            videos_file = videos_file.sort_values('Duration', ascending=False)
        start = args.start
        end = args.end if args.end != -1 else len(videos_file) 

        if 'Path' in videos_file.columns:
            videoPaths = list(videos_file['Path'])
            if 'AudioPath' in videos_file.columns:
                audioPaths = list(videos_file['AudioPath'])
            else:
                audioPaths = list(videos_file['Path'])
        elif 'Audio Source' in videos_file.columns and 'Video Source' in videos_file.columns:
            videoPaths = list(videos_file['Video Source'])
            audioPaths = list(videos_file['Audio Source'])
        else:
            videoPaths = audioPaths = None
            raise Exception()
            
        fullFrameRoot = os.path.join(args.outroot, "fullFrame")
        faceFrameRoot = os.path.join(args.outroot, "faceFrame")
        os.makedirs(fullFrameRoot, exist_ok=True)
        os.makedirs(faceFrameRoot, exist_ok=True)

        fullFrameOrigRoot = os.path.join(args.outroot, "fullFrame-orig")
        faceFrameOrigRoot = os.path.join(args.outroot, "faceFrame-orig")
        os.makedirs(fullFrameOrigRoot, exist_ok=True)
        os.makedirs(faceFrameOrigRoot, exist_ok=True)

        fullFrameMergedRoot = os.path.join(args.outroot, "fullFrame-merged")
        faceFrameMergedRoot = os.path.join(args.outroot, "faceFrame-merged")
        os.makedirs(fullFrameMergedRoot, exist_ok=True)
        os.makedirs(faceFrameMergedRoot, exist_ok=True)

        outPathsFullFrame = [os.path.join(fullFrameRoot, "-".join(path.split("/")[-2:])) for path in videoPaths]
        outPathsFaceFrame = [os.path.join(faceFrameRoot, "-".join(path.split("/")[-2:])) for path in videoPaths]

        outPathsOrigFullFrame = [os.path.join(fullFrameOrigRoot, "-".join(path.split("/")[-2:])) for path in videoPaths]
        outPathsOrigFaceFrame = [os.path.join(faceFrameOrigRoot, "-".join(path.split("/")[-2:])) for path in videoPaths]

        outPathsMergedFace = [os.path.join(faceFrameMergedRoot, "-".join(path.split("/")[-2:])) for path in videoPaths]
        outPathsMergedFullFrame = [os.path.join(fullFrameMergedRoot, "-".join(path.split("/")[-2:])) for path in videoPaths]

        if args.csvroot:
            videoPaths = [os.path.join(args.csvroot, x) for x in videoPaths]
            audioPaths = [os.path.join(args.csvroot, x) for x in audioPaths]

        recon_losses_1 = []
        recon_losses_2 = []
        duration = []

        for i in range(start, end):
            print("\n\nPROCESSING VIDEO", i)
            videoPath, audioPath = videoPaths[i], audioPaths[i]
            videoPath = videoPath.replace("/mnt/newdisk/home/arjun.ashok/data/lrs3-HD", "/mnt/disks/sdc/lrs3-HD")
            print(videoPath)
            if args.video_root:
                videoPath = os.path.join(args.video_root, videoPaths[i])
            if args.audio_root:
                audioPath = os.path.join(args.audio_root, audioPaths[i])

            start_time = time.time()
            predictor.run(videoPath, audioPaths[i], outPathsFullFrame[i], outPathsOrigFullFrame[i], \
                        outPathsFaceFrame[i], outPathsOrigFaceFrame[i], outPathsMergedFace[i], outPathsMergedFullFrame[i])
            end_time = time.time()
            duration.append(end_time - start_time)

            originalFrames = np.array(get_frames(videoPath))
            predictedFrames = np.array(get_frames(outPathsFullFrame[i]))

        if args.save_duration_path:
            df = pd.DataFrame({"VideoPath":videoPaths[start:end], \
                            "Duration": videos_file['Duration'][start:end], \
                            "Inference time": duration})
            df.to_csv(args.save_duration_path, index=False)

    else:
        predictor.run(args.face, args.audio, args.outfile)