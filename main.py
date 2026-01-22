from gst_cameraip import VideoFrameCapture
from config.configuration import CameraConfig, PipelineConfig
from processor.processor import YOLOProcessor
from YOLO.ULObjectDetection import ULObjectDetection

if __name__ == "__main__":
    camera_config = CameraConfig(
        host="192.168.1.108",
        port=554,
        username="admin",
        password="aits@1605",
        path="/cam/realmonitor?channel=1&subtype=0"
    )
    
    pipeline_config = PipelineConfig(latency=0, drop_on_latency=True)
    
    # Optional: Add YOLO processor
    detector = ULObjectDetection(ckpt_path="YOLO", model_name="model")
    processors = [YOLOProcessor(detector, confidence=0.5)]
    # processors = []
    
    capture = VideoFrameCapture(camera_config, pipeline_config, processors)
    capture.start()