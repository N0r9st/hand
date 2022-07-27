import os
import subprocess
from pathlib import Path

import cv2
import gym
import numpy as np


class VideoRecorder(gym.Wrapper):
    def __init__(self, env, video_folder, save_every, ffmpeg=False):
        super().__init__(env)
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        self.images = []
        self.reset_count = 0
        self.ffmpeg = ffmpeg
        self.video_folder = video_folder
        self.save_every = save_every

    def step(self, action: np.ndarray):
        obs, reward, done, info = self.env.step(action)
        self.images.append(obs[:, :, :3])
        return obs, reward, done, info
    
    def reset(self):
        if self.reset_count%self.save_every == 0 and self.images:
            self.save_video(self.images)
        obs = self.env.reset()
        self.images = [obs[:, :, :3]]
        self.reset_count += 1
        return obs

    def save_video(self, images):
        filename = str(self.reset_count) + ".mp4"
        if self.ffmpeg:
            _filename = '_' + filename
        else:
            _filename = filename
        filename = os.path.join(self.video_folder, filename)
        _filename = os.path.join(self.video_folder, _filename)
        out = cv2.VideoWriter(_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, images[0].shape[:2])
        for frame in images:
            out.write(frame)
        out.release()
        if self.ffmpeg:
            subprocess.run(["ffmpeg", "-y", "-i", _filename, filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(_filename)

def video_from_images(filename, images, ffmpeg=True):
    if ffmpeg:
        first, second = os.path.splitext(filename)
        _filename = first + '_' + second
    else:
        _filename = filename
    out = cv2.VideoWriter(_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, images[0].shape[:2])
    for frame in images:
        out.write(frame)
    out.release()
    if ffmpeg:
        subprocess.run(["ffmpeg", "-y", "-i", _filename, filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(_filename)
