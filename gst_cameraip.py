import logging
import sys
import time
from typing import List, Optional

import gi
import numpy as np

gi.require_version("GLib", "2.0")
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

from config.configuration import CameraConfig, PipelineConfig, RecordingConfig
from monitor.monitor import ResourceMetrics, ResourceMonitor
from processor.processor import DisplayProcessor, FrameProcessor

logging.basicConfig(
    level=logging.INFO, format="[%(name)s] [%(levelname)8s] - %(message)s"
)
logger = logging.getLogger(__name__)


class GStreamerPipelineBuilder:
    """Builds GStreamer pipeline elements"""

    @staticmethod
    def create_decoder():
        decode = Gst.ElementFactory.make("nvh264dec", "decode")
        if not decode:
            logger.warning("GPU decoder not available, falling back to CPU")
            decode = Gst.ElementFactory.make("avdec_h264", "decode")
        else:
            logger.info("Using NVIDIA GPU decoder for H.264")
        return decode

    @staticmethod
    def create_elements(
        recording_location: str,
        do_recording: bool,
        long_record: bool,
        keep_encoding: bool,
    ) -> dict:
        elements = {
            "source": Gst.ElementFactory.make("rtspsrc", "source"),
            "depay": Gst.ElementFactory.make("rtph264depay", "depay"),
            "parse": Gst.ElementFactory.make("h264parse", "parse"),
            "decode": GStreamerPipelineBuilder.create_decoder(),
            "queue_display": Gst.ElementFactory.make("queue", "queue_display"),
            "convert": Gst.ElementFactory.make("videoconvert", "convert"),
            "sink": Gst.ElementFactory.make("appsink", "sink"),
        }

        if do_recording:
            if long_record and not keep_encoding:
                elements["tee"] = Gst.ElementFactory.make("tee", "tee")
                elements["queue_record"] = Gst.ElementFactory.make(
                    "queue", "queue_record"
                )

                # Videoscale + capsfilter (giữ để downscale nếu cần giảm RAM/file size)
                elements["videoscale"] = Gst.ElementFactory.make(
                    "videoscale", "videoscale"
                )
                elements["capsfilter"] = Gst.ElementFactory.make(
                    "capsfilter", "capsfilter"
                )

                # Encoder: x265enc (HEVC)
                elements["encoder"] = Gst.ElementFactory.make("x265enc", "encoder")

                # Parse HEVC
                elements["h265parse_record"] = Gst.ElementFactory.make(
                    "h265parse", "h265parse_record"
                )

                elements["splitmuxsink"] = Gst.ElementFactory.make(
                    "splitmuxsink", "splitmuxsink"
                )

                # ===== TỐI ƯU X265ENC CHO FILE NHỎ + CHẤT LƯỢNG TỐT =====
                # 1. CRF mode (Constant Rate Factor) – tốt nhất cho chất lượng ổn định
                #    CRF 32–34: nén mạnh, file nhỏ, chất lượng khá (tăng lên 36–38 nếu muốn nhỏ hơn)
                elements["encoder"].set_property(
                    "option-string", "crf=38"
                )  # Hoặc "crf=34" để nhỏ hơn nữa

                # 2. Speed preset: slow = nén tốt hơn, vẫn real-time trên CPU mạnh
                elements["encoder"].set_property(
                    "speed-preset", "fast"
                )  # Hoặc "medium" nếu CPU load cao
                # Các lựa chọn: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow

                # 4. Keyframe interval lớn để nén tốt hơn
                elements["encoder"].set_property("key-int-max", 120)  # ~4–5 giây @25fps

                # 5. Tinh chỉnh thêm qua option-string (nếu cần)
                # Ví dụ: elements["encoder"].set_property("option-string", "crf=32:psy-rd=2.0:aq-mode=3:no-sao=1")
                #   - psy-rd=2.0: cải thiện perceptual quality
                #   - aq-mode=3: adaptive quantization tốt
                #   - no-sao=1: tắt SAO để giảm artifact ở một số cảnh (thử nếu thấy blocky)

                # Nếu muốn dùng bitrate thay vì CRF (ít khuyến khích hơn)
                # elements["encoder"].set_property("bitrate", 2000)  # kbps
                # elements["encoder"].set_property("option-string", "vbv-maxrate=2500:vbv-bufsize=5000")

                # Nếu downscale (rất khuyến khích cho 2560x1440 để giảm RAM + file size)
                # caps = Gst.Caps.from_string("video/x-raw,width=1920,height=1080")  # 1080p
                # hoặc "video/x-raw,width=1280,height=720" cho nhỏ hơn
                # elements["capsfilter"].set_property("caps", caps)

                # SplitMuxSink: dùng .mkv cho HEVC (tương thích tốt hơn .mp4 ở một số player)
                recording_pattern = recording_location + "video_%05d.mkv"
                elements["splitmuxsink"].set_property("location", recording_pattern)
                elements["splitmuxsink"].set_property(
                    "max-size-time", 600 * 1000000000
                )  # 10 phút
                elements["splitmuxsink"].set_property("send-keyframe-requests", True)
                elements["splitmuxsink"].set_property(
                    "muxer-factory", "matroskamux"
                )  # .mkv
                # Nếu muốn .mp4: "muxer-factory", "mp4mux" (nhưng cần test seek/file size)

                # Queue: thêm leaky để tránh backlog RAM
                elements["queue_record"].set_property("leaky", 2)  # drop old nếu đầy
                elements["queue_record"].set_property("max-size-buffers", 5)
                elements["queue_record"].set_property(
                    "max-size-time", 2000000000
                )  # ~2s
                elements["queue_record"].set_property("max-size-bytes", 0)
            elif keep_encoding and long_record:
                elements["tee"] = Gst.ElementFactory.make("tee", "tee")
                elements["queue_record"] = Gst.ElementFactory.make(
                    "queue", "queue_record"
                )

                # Parse H264 để đảm bảo stream hợp lệ
                elements["h264parse_record"] = Gst.ElementFactory.make(
                    "h264parse", "h264parse_record"
                )

                # SplitFileSink để tự động chia file H264 raw
                elements["splitfilesink"] = Gst.ElementFactory.make(
                    "multifilesink", "splitfilesink"
                )

                # Cấu hình multifilesink
                recording_pattern = recording_location + "video_%05d.h264"
                elements["splitfilesink"].set_property("location", recording_pattern)

                # Chia file theo thời gian hoặc kích thước
                # Option 1: Chia theo thời gian (số frame * thời gian mỗi frame)
                # Ví dụ: 30fps * 600s = 18000 frames = 10 phút
                elements["splitfilesink"].set_property(
                    "max-files", 0
                )  # Không giới hạn số file
                elements["splitfilesink"].set_property(
                    "next-file", 4
                )  # 4 = max-size mode
                elements["splitfilesink"].set_property(
                    "max-file-size", 100 * 1024 * 1024
                )  # 100MB mỗi file

                # Hoặc chia theo số frame (nếu biết framerate)
                # elements["splitfilesink"].set_property("next-file", 3)  # 3 = max-duration mode
                # elements["splitfilesink"].set_property("max-file-duration", 600 * 1000000000)  # 10 phút

                # Queue settings
                elements["queue_record"].set_property("max-size-buffers", 0)
                elements["queue_record"].set_property("max-size-time", 0)
                elements["queue_record"].set_property("max-size-bytes", 0)
            else:
                elements["queue_record"] = Gst.ElementFactory.make(
                    "queue", "queue_record"
                )
                elements["filesink"] = Gst.ElementFactory.make("filesink", "recorder")
                elements["mp4mux"] = Gst.ElementFactory.make("mp4mux", "muxer")
                elements["filesink"] = Gst.ElementFactory.make("filesink", "recorder")
                elements["tee"] = Gst.ElementFactory.make("tee", "tee")

                recording_path = recording_location + str(time.time()) + ".mp4"
                elements["filesink"].set_property("location", recording_path)

                elements["queue_record"].set_property(
                    "max-size-buffers", 0
                )  # Nhiều buffer
                elements["queue_record"].set_property("max-size-time", 0)  # 3 giây
                elements["queue_record"].set_property("max-size-bytes", 0)  # 10MB

        if elements["queue_display"]:
            elements["queue_display"].set_property("max-size-buffers", 1)
            elements["queue_display"].set_property("leaky", 2)

        if not all(elements.values()):
            raise RuntimeError("Not all GStreamer elements could be created")

        return elements


class PerformanceTracker:
    """Tracks FPS and latency metrics"""

    def __init__(self):
        self.last_time = time.perf_counter()
        self.frame_count = 0
        self.fps = 0.0
        self.latency_ms = 0.0
        self.pipeline_start_time: Optional[float] = None

    def update(self) -> float:
        """Update metrics and return interval"""
        current_time = time.perf_counter()
        interval = current_time - self.last_time
        self.last_time = current_time
        self.fps = 1.0 / interval if interval > 0 else 0.0
        self.frame_count += 1
        return interval

    def calculate_latency(self, buffer, pipeline):
        """Calculate latency from buffer timestamps"""
        pts = buffer.pts
        if pts == Gst.CLOCK_TIME_NONE or self.pipeline_start_time is None:
            return

        clock = pipeline.get_clock()
        if clock:
            base_time = pipeline.get_base_time()
            running_time = clock.get_time() - base_time
            if running_time > pts:
                self.latency_ms = (running_time - pts) / Gst.MSECOND


class VideoFrameCapture:
    """Main video capture class using GStreamer"""

    def __init__(
        self,
        camera_config: CameraConfig,
        do_record: bool,
        recording_config: RecordingConfig,
        pipeline_config: Optional[PipelineConfig] = None,
        processors: Optional[List[FrameProcessor]] = None,
    ):
        self.camera_config = camera_config
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.recording_config = recording_config or RecordingConfig()
        self.do_record = do_record
        self.processors = processors or []

        self.pipeline = None
        self.elements = {}
        self.loop = None
        self.is_recording = False

        self.performance = PerformanceTracker()
        self.resource_monitor = ResourceMonitor()
        self.display_processor = DisplayProcessor()

        Gst.init(sys.argv[1:])
        self._create_pipeline()

    def _create_pipeline(self):
        """Create and configure the GStreamer pipeline"""
        self.elements = GStreamerPipelineBuilder.create_elements(
            self.recording_config.path,
            self.do_record,
            self.recording_config.longRecording,
            self.recording_config.keepEnconding,
        )
        self._configure_source()
        self._configure_sink()
        self._build_pipeline()
        self._link_elements()

        logger.info(f"Pipeline created for: {self.camera_config.rtsp_url}")

    def _configure_source(self):
        """Configure RTSP source element"""
        source = self.elements["source"]
        source.set_property("location", self.camera_config.rtsp_url)
        source.set_property("user-id", self.camera_config.username)
        source.set_property("user-pw", self.camera_config.password)
        source.set_property("latency", self.pipeline_config.latency)
        source.set_property("buffer-mode", self.pipeline_config.buffer_mode)
        source.set_property("drop-on-latency", self.pipeline_config.drop_on_latency)
        source.set_property("do-retransmission", self.pipeline_config.do_retransmission)
        source.connect("pad-added", self._on_pad_added)

    def _configure_sink(self):
        """Configure appsink element"""
        sink = self.elements["sink"]
        sink.set_property("emit-signals", True)
        sink.set_property("sync", self.pipeline_config.sync)
        sink.set_property("max-buffers", self.pipeline_config.max_buffers)
        sink.set_property("drop", True)
        sink.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB"))
        sink.connect("new-sample", self._on_new_sample)

    def _build_pipeline(self):
        """Build pipeline and add elements"""
        self.pipeline = Gst.Pipeline.new("video-pipeline")
        for element in self.elements.values():
            self.pipeline.add(element)

    def _link_elements(self):
        """Link pipeline elements"""
        record_links = [
            ("depay", "parse"),
            ("parse", "tee"),
            # Display branch
            ("tee", "decode"),
            ("decode", "queue_display"),
            ("queue_display", "convert"),
            ("convert", "sink"),
            ("tee", "queue_record"),
            ("queue_record", "mp4mux"),
            ("mp4mux", "filesink"),
        ]

        long_record_links = [
            ("depay", "parse"),
            ("parse", "decode"),
            ("decode", "tee"),
            # Display branch (giữ nguyên)
            ("tee", "queue_display"),
            ("queue_display", "convert"),
            ("convert", "sink"),
            # Record branch
            ("tee", "queue_record"),
            ("queue_record", "videoscale"),
            ("videoscale", "capsfilter"),
            ("capsfilter", "encoder"),
            ("encoder", "h265parse_record"),
            ("h265parse_record", "splitmuxsink"),
        ]

        keep_encoding_links = [
            ("depay", "parse"),
            ("parse", "tee"),  # Tee H264 encoded stream TRƯỚC decode
            # Display branch
            ("tee", "decode"),
            ("decode", "queue_display"),
            ("queue_display", "convert"),
            ("convert", "sink"),
            # Record branch - LƯU H264 RAW
            ("tee", "queue_record"),
            ("queue_record", "h264parse_record"),
            ("h264parse_record", "splitfilesink"),
        ]

        main_links = [
            ("depay", "parse"),
            ("parse", "decode"),
            ("decode", "queue_display"),
            ("queue_display", "convert"),
            ("convert", "sink"),
        ]

        if self.do_record and not self.recording_config.longRecording:
            main_links = record_links
        elif (
            self.do_record
            and self.recording_config.longRecording
            and not self.recording_config.keepEnconding
        ):
            main_links = long_record_links
        elif (
            self.do_record
            and self.recording_config.keepEnconding
            and self.recording_config.longRecording
        ):
            main_links = keep_encoding_links

        for src, dst in main_links:
            if not self.elements[src].link(self.elements[dst]):
                raise RuntimeError(f"Failed to link {src} to {dst}")

    def _on_pad_added(self, src, new_pad):
        """Handle dynamic pad creation from rtspsrc"""
        logger.info(f"Received new pad '{new_pad.get_name()}' from '{src.get_name()}'")

        caps = new_pad.get_current_caps() or new_pad.query_caps(None)
        structure = caps.get_structure(0)

        if not structure.get_name().startswith("application/x-rtp"):
            return

        if structure.get_string("media") != "video":
            return

        sink_pad = self.elements["depay"].get_static_pad("sink")
        if sink_pad.is_linked():
            logger.info("Video pad already linked")
            return

        ret = new_pad.link(sink_pad)
        if ret == Gst.PadLinkReturn.OK:
            logger.info("Successfully linked rtspsrc to depayloader")
        else:
            logger.error(f"Failed to link: {ret}")

    def _on_new_sample(self, appsink):
        """Callback when new frame is available"""
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK

        interval = self.performance.update()
        buffer = sample.get_buffer()
        self.performance.calculate_latency(buffer, self.pipeline)

        frame = self._extract_frame(sample, buffer)
        if frame is not None:
            metrics = self.resource_monitor.measure()
            self._process_frame(frame, interval, metrics)

        return Gst.FlowReturn.OK

    def _extract_frame(self, sample, buffer) -> Optional[np.ndarray]:
        """Extract numpy frame from GStreamer buffer"""
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return None

        frame = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data,
        ).copy()

        buffer.unmap(map_info)
        return frame

    def _process_frame(
        self, frame: np.ndarray, interval: float, metrics: ResourceMetrics
    ):
        """Process frame through all processors"""
        logger.info(
            f"Frame #{self.performance.frame_count} | "
            f"Size: {frame.shape[1]}x{frame.shape[0]} | "
            f"FPS: {self.performance.fps:.2f} | "
            f"Latency: {self.performance.latency_ms:.2f}ms | "
            f"CPU: {metrics.cpu_usage:.2f}% | "
            f"RAM: {metrics.ram_usage_mb:.2f}MB"
        )

        metadata = {
            "frame_count": self.performance.frame_count,
            "fps": self.performance.fps,
        }

        for processor in self.processors:
            frame = processor.process(frame, metadata)

        display_frame = self.display_processor.process(frame, metadata)
        GLib.idle_add(self._show_frame, display_frame)

    def _show_frame(self, frame: np.ndarray) -> bool:
        if self.display_processor.show(frame):
            self.stop()
        return False

    def _handle_message(self, bus, message):
        """Handle bus messages"""
        msg_type = message.type

        handlers = {
            Gst.MessageType.EOS: lambda: (logger.info("End-Of-Stream"), self.stop()),
            Gst.MessageType.ERROR: lambda: self._handle_error(message),
            Gst.MessageType.LATENCY: lambda: self.pipeline.recalculate_latency(),
        }

        if msg_type in handlers:
            handlers[msg_type]()
        elif msg_type == Gst.MessageType.STATE_CHANGED and message.src == self.pipeline:
            old, new, _ = message.parse_state_changed()
            logger.info(f"Pipeline: {old.value_nick} → {new.value_nick}")

    def _handle_error(self, message):
        err, debug = message.parse_error()
        logger.error(f"Error from {message.src.get_name()}: {err.message}")
        logger.error(f"Debug: {debug or 'none'}")
        self.stop()

    def start(self):
        """Start the pipeline"""
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start pipeline")

        self.performance.pipeline_start_time = time.perf_counter()

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._handle_message)

        self.loop = GLib.MainLoop()

        try:
            logger.info("Starting pipeline...")
            self.loop.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()

    def stop(self):
        """Stop everything gracefully"""
        if self.pipeline:
            # Gửi EOS vào toàn bộ pipeline
            self.pipeline.send_event(Gst.Event.new_eos())

            # Đợi EOS được xử lý (hoặc timeout 5s)
            bus = self.pipeline.get_bus()
            logger.log(logging.INFO, "Waiting for EOS...")
            bus.timed_pop_filtered(
                5 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR
            )
            logger.log(logging.INFO, "EOS received")
        if self.loop:
            self.loop.quit()

    def cleanup(self):
        self.display_processor.cleanup()
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        logger.info("Pipeline stopped and cleaned up")
