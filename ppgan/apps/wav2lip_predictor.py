from os import listdir, path, makedirs
import platform
import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import datetime
from pytz import timezone
import paddle
from paddle.utils.download import get_weights_path_from_url
from ppgan.faceutils import face_detection
from ppgan.utils import audio
from ppgan.models.generators.wav2lip import Wav2Lip
from .base_predictor import BasePredictor

WAV2LIP_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/wav2lip_hq.pdparams'
mel_step_size = 16


class Wav2LipPredictor(BasePredictor):
    def __init__(self,  checkpoint_path = None,
                 static = False,
                 fps = 25,
                 pads = [0, 10, 0, 0],
                 face_det_batch_size = 16,
                 wav2lip_batch_size = 128,
                 resize_factor = 1,
                 crop = [0, -1, 0, -1],
                 box = [-1, -1, -1, -1],
                 rotate = False,
                 nosmooth = False,
                 face_detector = 'sfd',
                 face_enhancement = False):
        self.img_size = 96
        self.checkpoint_path = checkpoint_path
        self.static = static
        self.fps = fps
        self.pads = pads
        self.face_det_batch_size = face_det_batch_size
        self.wav2lip_batch_size = wav2lip_batch_size
        self.resize_factor = resize_factor
        self.crop = crop
        self.box = box
        self.rotate = rotate
        self.nosmooth = nosmooth
        self.face_detector = face_detector
        self.face_enhancement = face_enhancement
        if face_enhancement:
            from ppgan.faceutils.face_enhancement import FaceEnhancement
            self.faceenhancer = FaceEnhancement()
        makedirs('./temp', exist_ok=True)

    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i:i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self, images):
        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False,
            face_detector=self.face_detector)

        batch_size = self.face_det_batch_size

        # print("In face_detect. Images[0]:", images[0].shape)

        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    predictions.extend(
                        detector.get_detections_for_batch(
                            np.array(images[i:i + batch_size])))
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError(
                        'Image too big to run face detection on GPU. Please use the --resize_factor argument'
                    )
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(
                    batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = self.pads
        noFaceFrames = 0
        for rect, image in zip(predictions, images):
            if rect is None:
                y1, y2 = 0, image.shape[0]
                x1, x2 = 0, image.shape[1]
                noFaceFrames += 1
                # cv2.imwrite(
                #     'temp/faulty_frame.jpg',
                #     image)  # check this frame where the face was not detected.
                # raise ValueError(
                #     'Face not detected! Ensure the video contains a face in all the frames.'
                # )
            else:
                y1 = max(0, rect[1] - pady1)
                y2 = min(image.shape[0], rect[3] + pady2)
                x1 = max(0, rect[0] - padx1)
                x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])

        print("# frames with no face:", noFaceFrames)
        boxes = np.array(results)
        if not self.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)]
                   for image, (x1, y1, x2, y2) in zip(images, boxes)]

        del detector
        return results

    def datagen(self, frames, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.box[0] == -1:
            if not self.static:
                face_det_results = self.face_detect(
                    frames)  # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect([frames[0]])
        else:
            print(
                'Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.box
            face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)]
                                for f in frames]

        print("Number of mels:", len(mels))
        for i, m in enumerate(mels):
            idx = 0 if self.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.img_size, self.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(
                    mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size // 2:] = 0

                img_batch = np.concatenate(
                    (img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(
                    mel_batch,
                    [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                # print("Image batch shape: {}. Mel batch shape: {}".\
                #     format(img_batch.shape, mel_batch.shape))

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(
                mel_batch,
                [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def run(self, face, audio_seq, outfile, copyfile=None,  outfileFaceOnly=None, copyfileFaceOnly=None, mergeFace=None, mergeFullFrame=None):
        date_time = datetime.datetime.now(timezone("Asia/Kolkata")).strftime("-%d-%m-%Y-%H:%M:%S")
        temp_dir = path.join('temp', date_time, outfile)
        makedirs(temp_dir, exist_ok=True)

        if os.path.isfile(face) and path.basename(
                face).split('.')[1] in ['jpg', 'png', 'jpeg']:
            self.static = True
        
        if not os.path.isfile(face):
            raise ValueError(
                '--face argument must be a valid path to video/image file')

        elif path.basename(
                face).split('.')[1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(face)]
            fps = self.fps

        else:
            video_stream = cv2.VideoCapture(face)
            fps = video_stream.get(cv2.CAP_PROP_FPS)

            print('Reading video frames. FPS is', fps)

            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if self.resize_factor > 1:
                    frame = cv2.resize(
                        frame, (frame.shape[1] // self.resize_factor,
                                frame.shape[0] // self.resize_factor))

                if self.rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = self.crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]

                full_frames.append(frame)

                # print("Frame shape:", frame.shape)

        print("Number of frames available for inference: " +
              str(len(full_frames)))

        if not audio_seq.endswith('.wav'):
            print('Extracting raw audio...')
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(
                audio_seq, path.join(temp_dir, 'temp.wav'))

            subprocess.call(command, shell=True)
            audio_seq =  path.join(temp_dir, 'temp.wav')

        wav = audio.load_wav(audio_seq, 16000)
        mel = audio.melspectrogram(wav)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again'
            )

        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx:start_idx + mel_step_size])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        full_frames = full_frames[:len(mel_chunks)]

        batch_size = self.wav2lip_batch_size
        gen = self.datagen(full_frames.copy(), mel_chunks)

        model = Wav2Lip()
        if self.checkpoint_path is None:
            model_weights_path = get_weights_path_from_url(WAV2LIP_WEIGHT_URL)
            weights = paddle.load(model_weights_path)
        else:
            weights = paddle.load(self.checkpoint_path)
        model.load_dict(weights)
        model.eval()
        print("Model loaded")
        # import pdb; pdb.set_trace()
        
        out = outOrig = out_faceFrame = outOrig_faceFrame = merged_faceFrames = merged_fullFrames = None

        for i, (img_batch, mel_batch, frames, coords) in enumerate(
                tqdm(gen,
                     total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
            if i == 0:
                frame_h, frame_w = full_frames[0].shape[:-1]

                # Full Frame types
                out = cv2.VideoWriter(path.join(temp_dir, 'result_fullFrame.avi'),
                                      cv2.VideoWriter_fourcc(*'DIVX'), fps,
                                      (frame_w, frame_h))
                if copyfile: outOrig = cv2.VideoWriter(path.join(temp_dir, 'orig_fullFrame.avi'),
                                                        cv2.VideoWriter_fourcc(*'DIVX'), fps,
                                                        (frame_w, frame_h))
                
                # Face Only Types            
                if outfileFaceOnly: out_faceFrame = cv2.VideoWriter(path.join(temp_dir, 'result_faceFrame.avi'),
                                      cv2.VideoWriter_fourcc(*'DIVX'), fps,
                                      (self.img_size, self.img_size))
                if copyfileFaceOnly: outOrig_faceFrame = cv2.VideoWriter(path.join(temp_dir, 'orig_faceFrame.avi'),
                                      cv2.VideoWriter_fourcc(*'DIVX'), fps,
                                      (self.img_size, self.img_size))

    			# Merged Types
                if mergeFace: merged_faceFrames = cv2.VideoWriter(path.join(temp_dir, 'merged_faceFrames.avi'),
                                                            cv2.VideoWriter_fourcc(*'DIVX'), fps,
                                                            (self.img_size*2, self.img_size))
                if mergeFullFrame:	merged_fullFrames = cv2.VideoWriter(path.join(temp_dir, 'merged_fullFrames.avi'),
                                                            cv2.VideoWriter_fourcc(*'DIVX'), fps,
                                                            (frame_w*2, frame_h))
            img_batch = paddle.to_tensor(np.transpose(
                img_batch, (0, 3, 1, 2))).astype('float32')
            mel_batch = paddle.to_tensor(np.transpose(
                mel_batch, (0, 3, 1, 2))).astype('float32')

            # print("Image batch shape: {}. Mel batch shape: {}".\
            #     format(img_batch.shape, mel_batch.shape))

            with paddle.no_grad():
                pred = model(mel_batch, img_batch)

            # print("Predictions shape: {}".\
            #     format(pred.shape))

            pred = pred.numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                replacedFrame = f.copy()
                if self.face_enhancement:
                    p = self.faceenhancer.enhance_from_image(p)
                pred_resized = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                replacedFrame[y1:y2, x1:x2] = pred_resized
                out.write(replacedFrame)
                if copyfile: outOrig.write(f)
                if outfileFaceOnly: out_faceFrame.write(p.astype(np.uint8))
                if copyfileFaceOnly: 
                    orig_face_resized = cv2.resize(f[y1:y2, x1:x2].astype(np.uint8), (self.img_size, self.img_size))
                    outOrig_faceFrame.write(orig_face_resized)
                if mergeFace:
                    orig_face_resized = cv2.resize(f[y1:y2, x1:x2].astype(np.uint8), (self.img_size, self.img_size))
                    orig_pred_faceFrames = np.hstack((orig_face_resized.astype(np.uint8).copy(), p.astype(np.uint8).copy()))
                    merged_faceFrames.write(orig_pred_faceFrames)
                if mergeFullFrame:
                    orig_pred_fullFrames = np.hstack((f, replacedFrame))
                    merged_fullFrames.write(orig_pred_fullFrames)
                    
        out.release()
        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(
            audio_seq, path.join(temp_dir, 'result_fullFrame.avi'), outfile)
        subprocess.call(command, shell=platform.system() != 'Windows')

        if outfileFaceOnly: 
            out_faceFrame.release()
            command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {} -loglevel error'.format(
                    audio_seq, path.join(temp_dir, 'result_faceFrame.avi'), outfileFaceOnly)
            subprocess.call(command, shell=platform.system() != 'Windows')

        if copyfile: 
            outOrig.release()
            command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {} -loglevel error'.format(
                    audio_seq, path.join(temp_dir, 'orig_fullFrame.avi'), copyfile)
            subprocess.call(command, shell=platform.system() != 'Windows')

        if copyfileFaceOnly: 
            outOrig_faceFrame.release()
            command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {} -loglevel error'.format(
                    audio_seq, path.join(temp_dir, 'orig_faceFrame.avi'), copyfileFaceOnly)
            subprocess.call(command, shell=platform.system() != 'Windows')

        if mergeFace:
            merged_faceFrames.release()
            command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {} -loglevel error'.format(
                    audio_seq, path.join(temp_dir, 'merged_faceFrames.avi'), mergeFace)
            subprocess.call(command, shell=platform.system() != 'Windows')

        if mergeFullFrame:
            merged_fullFrames.release()
            command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {} -loglevel error'.format(
                    audio_seq, path.join(temp_dir, 'merged_fullFrames.avi'), mergeFullFrame)
            subprocess.call(command, shell=platform.system() != 'Windows')