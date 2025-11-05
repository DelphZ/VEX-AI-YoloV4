import serial
import json
import time
import cv2
import numpy as np
from camara_handler import CameraHandler
from vex_detect import YOLODetector

class VexCommunicator:
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyACM1', 115200, timeout=1)
        self.camera = CameraHandler()
        self.detector = YOLODetector(
            weights='/home/delph/vex_detect_yolov4/runs/train/yolov4-vex-detect-160/weights/best.pt',
            cfg='/home/delph/vex_detect_yolov4/cfg/yolov4-tiny.cfg',
            names='/home/delph/vex_detect_yolov4/_vex.names'
        )
        self.running = True
        self.start_detecting = False
        self.need_debug_print = False
    def process_request(self, category, confidence):
        if self.need_debug_print:
            print("start getting frames")
        color_frame, depth_frame = self.camera.get_frames()
        if color_frame is None or depth_frame is None:
            return None
        if self.need_debug_print:
            print("Start yolo detect")
        detections = self.detector.detect(color_frame)
        self.start_detecting = True
        if self.need_debug_print:
            print("start filtering out")
        results = []
        
        for det in detections:
            if det['class'] == category and det['confidence'] >= confidence:
                bbox = det['bbox']
                distance = self.calculate_depth(depth_frame, bbox)
                if distance:
                    results.append({
                        'x': (bbox[0] + bbox[2]) / 2,
                        'y': (bbox[1] + bbox[3]) / 2,
                        'distance': distance,
                        'category': det['class'],
                        'conf': det['confidence'],
                    })
                else:
                    results.append({
                        'x': (bbox[0] + bbox[2]) / 2,
                        'y': (bbox[1] + bbox[3]) / 2,
                        'distance': 100,
                        'category': det['class'],
                        'conf': det['confidence'],
                    })
        
        return results

    def calculate_depth(self, depth_data, bbox):
        # Convert bbox coordinates to depth scale
        x1 = int(bbox[0] * (self.camera.depth_scale[0] / self.camera.color_scale[0]))
        y1 = int(bbox[1] * (self.camera.depth_scale[1] / self.camera.color_scale[1]))
        x2 = int(bbox[2] * (self.camera.depth_scale[0] / self.camera.color_scale[0]))
        y2 = int(bbox[3] * (self.camera.depth_scale[1] / self.camera.color_scale[1]))
        
        region = depth_data[y1:y2, x1:x2].flatten()
        valid = region[region > 0]
        if len(valid) == 0:
            return None
        
        sorted_depths = np.sort(valid)
        n = len(sorted_depths)
        trimmed = sorted_depths[int(n*0.1):int(n*0.9)]
        return float(np.mean(trimmed))

    def run(self):
        hand_shake = False
        while self.running:
            if self.start_detecting and not hand_shake:
                self.ser.write(b"jetson start detecting\n")
                if self.ser.in_waiting > 0:
                    hand_shake = True
            elif not hand_shake:
                _ = self.process_request(0, 0.5)
                time.sleep(0.1)
                continue
            category = 0
            confidence = 0.5
            if self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode().strip()
                    self.ser.reset_input_buffer()
                    if line == "handshake":
                        print("jetson start detecting\n")
                        self.ser.write(b"jetson start detecting\n")
                        time.sleep(0.01)
                        continue
                    #print(line)
                    #category = int(line)
                    #confidence = 0.55
                    if ',' in line:
                        category_str, confidence_str = line.split(',')
                        category = int(category_str)
                        confidence = float(confidence_str)
                        if 0 <= category <= 4:
                            results = self.process_request(category, confidence)
                            if results is not None:
                                self.ser.write(json.dumps(results).encode() + b'\n')
                                print("send the detected obj")
                            else:
                                self.ser.write(b"no obj detected\n")
                                print("No obj detected")
                        #if self.need_debug_print:
                        print(f"Category: {category}, Confidence: {confidence}")
                    else:
                        print(f"Invalid data: {line}")
                    #print(f"category is {category}, conf is {confidence}")
                    if confidence < 0.05:
                        confidence = 0.05
                    #category = 0
                    #if 0 <= category <= 2:
                    #    results = self.process_request(category, confidence)
                    #    if results is not None:
                    #        self.ser.write(json.dumps(results).encode() + b'\n')
                    #    else:
                    #        print("None frame is get.")
                except Exception as e:
                    print(f"Error: {str(e)}")
            time.sleep(0.01)

if __name__ == '__main__':
    communicator = VexCommunicator()
    try:
        communicator.run()
    finally:
        communicator.camera.stop()
