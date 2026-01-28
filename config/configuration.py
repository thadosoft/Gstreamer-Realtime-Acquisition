from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraConfig:
    host: str
    port: int = 554
    username: Optional[str] = None
    password: Optional[str] = None
    path: str = ""

    @property
    def rtsp_url(self) -> str:
        return f"rtsp://{self.host}:{self.port}{self.path}"


@dataclass
class RecordingConfig:
    cameraName: str = "dahua"
    path: str = "/home/deka/Videos/GST/"
    fps: int = 25


@dataclass
class PipelineConfig:
    latency: int = 0
    buffer_mode: int = 3
    drop_on_latency: bool = True
    do_retransmission: bool = False
    max_buffers: int = 1
    sync: bool = False
