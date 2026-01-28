import multiprocessing
import threading
import time

from config.configuration import CameraConfig, PipelineConfig, RecordingConfig
from gst_cameraip import VideoFrameCapture
from processor.processor import YOLOProcessor
from YOLO.ULObjectDetection import ULObjectDetection


def start_recording(capture, filename):
    time.sleep(2)
    capture.start_recording(filename)


def start_capturing(capture):
    capture.start()


if __name__ == "__main__":
    camera_config = CameraConfig(
        host="192.168.1.108",
        port=554,
        username="admin",
        password="aits@1605",
        path="/cam/realmonitor?channel=1&subtype=0",
    )
    pipeline_config = PipelineConfig(latency=0, drop_on_latency=True)
    recording_config = RecordingConfig()

    detector = ULObjectDetection(ckpt_path="YOLO", model_name="model")
    detector.export_openvino(fresh=False)
    processors = []

    capture = VideoFrameCapture(
        camera_config=camera_config,
        do_record=True,
        recording_config=recording_config,
        processors=processors,
    )
    capture.start()
