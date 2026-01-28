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
    def create_elements(recording_location: str, do_recording: bool) -> dict:
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
            elements["queue_record"] = Gst.ElementFactory.make("queue", "queue_record")
            elements["filesink"] = Gst.ElementFactory.make("filesink", "recorder")
            elements["mp4mux"] = Gst.ElementFactory.make("mp4mux", "muxer")
            elements["filesink"] = Gst.ElementFactory.make("filesink", "recorder")
            elements["tee"] = Gst.ElementFactory.make("tee", "tee")

            recording_path = recording_location + str(time.time()) + ".mp4"
            elements["filesink"].set_property("location", recording_path)

            elements["queue_record"].set_property("max-size-buffers", 0)  # Nhiều buffer
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
            self.recording_config.path, self.do_record
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

        main_links = [
            ("depay", "parse"),
            ("parse", "decode"),
            ("decode", "queue_display"),
            ("queue_display", "convert"),
            ("convert", "sink"),
        ]

        if self.do_record:
            main_links = record_links

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
