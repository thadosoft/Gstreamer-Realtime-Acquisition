from abc import ABC, abstractmethod
import numpy as np
import cv2 as cv
from typing import Optional 

class FrameProcessor(ABC):
    @abstractmethod
    def process(self, frame: np.ndarray, metadata: dict) -> np.ndarray:
        """Process frame and return modified frame"""
        pass

class DisplayProcessor(FrameProcessor):
    def __init__(self, window_name: str = "IP Camera Frame", display_size: tuple = (640, 480)):
        self.window_name = window_name
        self.display_size = display_size
    
    def process(self, frame: np.ndarray, metadata: dict) -> np.ndarray:
        frame_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        return cv.resize(frame_bgr, self.display_size)
    
    def show(self, frame: np.ndarray) -> bool:
        """Returns True if should quit"""
        cv.imshow(self.window_name, frame)
        return cv.waitKey(1) & 0xFF == ord('q')
    
    def cleanup(self):
        cv.destroyAllWindows()

class YOLOProcessor(FrameProcessor):
    def __init__(self, detector, confidence: float = 0.5):
        self.detector = detector
        self.confidence = confidence

    def process(self, frame: np.ndarray, metadata: dict) -> np.ndarray:
        results = self.detector.predict(mode='normal', img=frame, conf=self.confidence)[0]

        if results and hasattr(results, 'boxes') and results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f"{cls}: {conf:.2f}"
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, label, (x1, y1 - 10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame