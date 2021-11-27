"""Camera Trap

An application that records video when motion is detected.

Contact
-------
Author: Andre Telfer
Email: andretelfer@cmail.carleton.ca
"""

import cv2
import logging
import multiprocessing
import sys
import ffmpeg
import time
import numpy as np

from datetime import datetime
from ruamel.yaml import YAML
from flask import Flask
from multiprocessing import Process, Queue, Pipe
from pathlib import Path
from collections import deque

app = Flask(__name__)

# Load the config
config = YAML().load(Path('config.yaml'))
videos = Path(config['video_folder']) 
videos = videos.absolute() / datetime.now().strftime("%y%m%d_%H%M%S")
videos.mkdir(parents=True)

# Shared objects
capture_queue = Queue()
motion_queue = Queue()
video_queue = Queue()
motion_reader, motion_writer = Pipe(duplex=False)

# Configure the logger
logger = multiprocessing.get_logger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')

fh = logging.FileHandler(videos/'test.log')
fh.setFormatter(formatter)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
logger.addHandler(sh)

# Start capturing frames
def capture_task(config, capture_queue, **kwargs):
    logger = multiprocessing.get_logger()
    cap = cv2.VideoCapture(config['device'])
    cap.set(cv2.CAP_PROP_FPS, config['fps'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])

    logger.info(f'FPS: {cap.get(cv2.CAP_PROP_FPS)}')
    print(cap.get(cv2.CAP_PROP_FPS))
    while True:
        start = time.time_ns()
        ret, frame = cap.read()
        if ret:
            capture_queue.put(frame)
            logger.debug(f"capture queue size: {capture_queue.qsize()}")           

        logger.debug(f'frame capture loop: {(time.time_ns() - start)/1e6}ms')

# Start reading frames
def split_task(config, capture_queue, video_queue, **kwargs):
    logger = multiprocessing.get_logger()
    while True:
        frame = capture_queue.get()

        # Small frame for motion
        small_frame = cv2.resize(
            frame, (100, 100), interpolation=cv2.INTER_LINEAR)
        motion_queue.put(small_frame)
        logger.debug(f"motion queue size: {motion_queue.qsize()}")

        # Larger frame for video
        video_queue.put(frame)
        logger.debug(f"video out queue size: {video_queue.qsize()}")

def motion_task(config, motion_queue, motion_writer, **kwargs):
    logger = multiprocessing.get_logger()
    fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    while True:
        frame = motion_queue.get()
        mask = fgbg.apply(frame)
        motion = np.sum(mask) > config['threshold']
        motion_writer.send(motion)
        logger.debug(f"motion: {np.sum(mask)}")
        
def create_writer(config):
    out_filename = str(videos / datetime.now().strftime("%y%m%d_%H%M%S")) + '.mp4'
    height = config['height']
    width = config['width']
    return (ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}')
        .output(out_filename, pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True))

def save_task(config, video_queue, motion_reader, **kwargs):
    buffer = config['fps']*3
    history = deque(maxlen=buffer)
    motion = False
    record_for = 0
    writer = None

    while True:
        frame = video_queue.get().astype(np.uint8).tobytes()
        if motion_reader.poll():
            motion = motion_reader.recv()

        if motion:
            record_for = buffer

        if record_for > 0:
            record_for -= 1
            if writer is None:
                logging.info("Start recording")
                writer = create_writer(config)
                for hframe in history:
                    writer.stdin.write(hframe)

            writer.stdin.write(frame)
        elif writer is not None:
            logging.info("Stop recording")
            # Close the video stream
            writer.stdin.close()
            writer = None

        history.append(frame) # This will overwrite old values

# Config and connections
shared_kwargs = {
    'config': config,
    'capture_queue': capture_queue,
    'motion_queue': motion_queue,
    'video_queue': video_queue,
    'motion_reader': motion_reader,
    'motion_writer': motion_writer,
}

split_process = Process(
    name="split", target=split_task, kwargs=shared_kwargs)
split_process.start()

capture_process = Process(
    name="capture", target=capture_task, kwargs=shared_kwargs)
capture_process.start()

motion_process = Process(
    name="motion", target=motion_task, kwargs=shared_kwargs)
motion_process.start()

save_process = Process(
    name="save", target=save_task, kwargs=shared_kwargs)
save_process.start()
