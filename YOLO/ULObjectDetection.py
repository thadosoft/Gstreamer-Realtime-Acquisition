import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO


class ULObjectDetection:
    def __init__(self, ckpt_path, model_name):
        self.ckpt_path = ckpt_path
        self.model_name = model_name
        self.model = YOLO(f"{ckpt_path}/{model_name}.pt")
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tensorrt_model = None
        self.openvino_model = None

        # self.slicer_normal = sv.InferenceSlicer(
        #     callback=self.slicer_callback_normal,
        #     slice_wh=(512, 512),
        #     overlap_ratio_wh=(0.4, 0.4),
        #     overlap_filter_strategy=sv.OverlapFilter.NONE
        # )

    def export_tensorrt(self, fresh):
        if fresh is True:
            self.model.export(format="engine")
        self.tensorrt_model = YOLO(
            f"{self.ckpt_path}\\{self.model_name}.engine", task="detect"
        )
        # self.slicer_tensorrt = sv.InferenceSlicer(
        #     callback=self.slicer_callback_tensorrt,
        #     slice_wh=(512, 512),
        #     overlap_ratio_wh=(0.4, 0.4),
        #     overlap_filter_strategy=sv.OverlapFilter.NONE
        # )
        #

    def export_openvino(self, fresh):
        if fresh is True:
            self.model.export(format="openvino")
        self.openvino_model = YOLO(
            f"{self.ckpt_path}/{self.model_name}_openvino_model/",
            task="detect",
        )

    def predict(self, mode, img, conf=0.5):
        if mode == "normal":
            # print('NORMAL')
            return self.model(img, conf=conf)
        elif mode == "tensorrt":
            print("TENSORRT")
            return self.tensorrt_model.predict(img, conf=conf, iou=0.2, verbose=True)
        elif mode == "openvino":
            print("OPENVINO")
            return self.openvino_model.predict(img, conf=conf, iou=0.2, verbose=True)

    def slicer_callback_normal(self, slice: np.ndarray) -> sv.Detections:
        result = self.model(slice, conf=0.4)[0]
        detections = sv.Detections.from_ultralytics(result)
        return detections

    def slicer_callback_tensorrt(self, slice: np.ndarray) -> sv.Detections:
        result = self.tensorrt_model.predict(slice, conf=0.4)[0]
        detections = sv.Detections.from_inference(result)
        return detections

    def slicer_predict(self, mode, img):
        if mode == "normal":
            return self.slicer_normal(img)
        elif mode == "tensorrt":
            return self.slicer_tensorrt(img)
