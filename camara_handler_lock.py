import os
import cv2
import numpy as np
import threading
from pyorbbecsdk import *
from _utils import frame_to_bgr_image

class CameraHandler:
    def __init__(self):
        self.pipeline = Pipeline()
        config = Config()
        self.initialize_streams(config)
        try:
            # Add alignment between color and depth streams
            self.pipeline.start(config)
        except OBError as e:
            print(f"Pipeline start failed: {str(e)}")
            raise
        self.latest_color = None
        self.latest_depth = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.capture_loop)
        self.thread.start()

    def initialize_streams(self, config):
        try:
            # Color configuration (unchanged)
            color_profile = self.find_target_profile(
                self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR),
                target_width=640, 
                target_height=480,
                target_format=OBFormat.RGB,
                min_fps=30
            )
            if color_profile:
                config.enable_stream(color_profile)
                self.color_scale = (640, 480)
            else:
                print("Using default color profile")
                config.enable_stream(OBStreamColor(), 
                                    OBFormat.RGB, 
                                    OBResolution(640, 480), 
                                    30)

            # Fixed depth configuration
            depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if depth_profile_list:
                depth_profile = self.find_target_profile(
                    depth_profile_list,
                    target_width=640,
                    target_height=480,
                    target_format=OBFormat.Y16,  # Changed from Y12 to Y16
                    min_fps=30
                )
                if depth_profile:
                    print(f"Using custom depth profile: {depth_profile}")
                    config.enable_stream(depth_profile)
                else:
                    print("Using default depth profile")
                    default_depth = depth_profile_list.get_default_video_stream_profile()
                    config.enable_stream(default_depth)
                    print(f"Default depth format: {default_depth.get_format().name}")
                
                # Set depth scale parameters
                self.depth_scale = (640, 480)
            else:
                raise RuntimeError("No depth profiles available")

        except Exception as e:
            print(f"Stream initialization failed: {str(e)}")
            raise

    def find_target_profile(self, profile_list, target_width, target_height, target_format=None, min_fps=0):
        """
        在流配置列表中查找匹配的目标配置
        :param profile_list: StreamProfileList 对象
        :param target_width: 目标宽度
        :param target_height: 目标高度
        :param target_format: 目标格式（OBFormat类型），None表示不限制
        :param min_fps: 最低要求的帧率（实际帧率>=此值）
        :return: VideoStreamProfile 或 None
        """
        if not profile_list:
            return None

        best_profile = None
        max_fps = -1

        for index in range(profile_list.get_count()):
            try:
                profile = profile_list.get_stream_profile_by_index(index)
                video_profile = profile.as_video_stream_profile()
                
                # 检查基础条件
                match_resolution = (video_profile.get_width() == target_width 
                                    and video_profile.get_height() == target_height)
                match_format = (target_format is None 
                                    or video_profile.get_format() == target_format)
                match_fps = (video_profile.get_fps() >= min_fps)
                
                if match_resolution and match_format and match_fps:
                    # 选择最高帧率的配置
                    if video_profile.get_fps() > max_fps:
                        best_profile = video_profile
                        max_fps = video_profile.get_fps()
                        
            except OBError:
                continue  # 忽略非视频流配置

        return best_profile

    def capture_loop(self):
        while self.running:
            try:
                # Increased timeout for frame synchronization
                frames = self.pipeline.wait_for_frames(25)  
                if frames:
                    # Get synchronized color and depth frames
                    aligned_frames = frames
                    
                    with self.lock:
                        # Process color frame
                        color_frame = aligned_frames.get_color_frame()
                        if color_frame:
                            self.latest_color = frame_to_bgr_image(color_frame)
                        
                        # Process depth frame with validation
                        depth_frame = aligned_frames.get_depth_frame()
                        if depth_frame:
                            self.process_depth_frame(depth_frame)
                            # print(f"Depth frame captured: {depth_frame.get_timestamp()}")
                        # else:
                        #     print("Invalid depth frame received")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Capture error: {str(e)}")

    def process_depth_frame(self, frame):
        data = np.frombuffer(frame.get_data(), dtype=np.uint16)
        data = data.reshape((frame.get_height(), frame.get_width()))
        data = (data * frame.get_depth_scale()).astype(np.uint16)
        self.latest_depth = data

    def get_frames(self):
        # Old get_frames
        # with self.lock:
        #     if self.latest_color is None or self.latest_depth is None:
        #         return None, None
        #     else:
        #         return self.latest_color.copy(), self.latest_depth.copy()

        #New get_frames
        # self.latest_color = None
        # self.latest_depth = None
        while True:
            with self.lock:
                # Check if both frames are available
                if self.latest_color is not None and self.latest_depth is not None:
                    # Make copies and reset if desired (optional)
                    color_copy = self.latest_color.copy()
                    depth_copy = self.latest_depth.copy()
                    # Uncomment below to reset after retrieval
                    # self.latest_color = None
                    # self.latest_depth = None
                    return color_copy, depth_copy
            # Release the lock while waiting to allow updates
            time.sleep(0.01)

    def stop(self):
        self.running = False
        self.thread.join()
        self.pipeline.stop()
