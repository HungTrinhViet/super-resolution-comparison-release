"""UI-independent demo inputs, sequential background inference and export."""
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
import logging
import math
from pathlib import Path
import platform
from queue import Empty, Queue
import threading
from time import perf_counter

import numpy as np
from PIL import Image

from ..inference import ALGORITHMS, InferenceRegistry
from ..io import save_rgb, validate_rgb
from ..metrics import evaluate

LOGGER = logging.getLogger(__name__)
DEMO_TIME_LABEL = 'Thời gian demo trong GUI, không phải kết quả benchmark đã nghiệm thu.'
METRIC_LABEL = 'Theo giao thức M1; không phải kết quả benchmark'
NO_VALID_HR = 'Không có HR tham chiếu hợp lệ. Không tính PSNR/SSIM.'
LARGE_IMAGE_WARNING = 'Ảnh LR lớn hơn 256 pixel ở ít nhất một chiều; cần xác nhận thời gian chạy có thể dài.'


def hr_caveat() -> str:
    """Size alone cannot establish the validity of an HR reference."""
    return ('PSNR/SSIM chỉ có ý nghĩa khi HR là ảnh gốc đúng của LR và được căn chỉnh khớp. '
            'Chương trình chỉ kiểm được kích thước.')


def _array_hash(rgb: np.ndarray) -> str:
    return hashlib.sha256(rgb.tobytes(order='C')).hexdigest()


@dataclass(frozen=True)
class LoadedImage:
    rgb: np.ndarray
    path: Path
    file_sha256: str
    original_mode: str
    conversion_notice: str | None
    requires_confirmation: bool
    size_warning: str | None


def load_input(path: Path) -> LoadedImage:
    """Convert explicitly at the demo boundary; never resize or orient input."""
    path = Path(path)
    if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp'}:
        raise ValueError('Chỉ hỗ trợ PNG/JPG/BMP.')
    raw = path.read_bytes()
    with Image.open(BytesIO(raw)) as image:
        if image.format not in {'PNG', 'JPEG', 'BMP'}:
            raise ValueError('Chỉ hỗ trợ PNG/JPG/BMP.')
        if image.getexif().get(274, 1) != 1:
            raise ValueError('Ảnh có hướng EXIF khác chuẩn. Hãy xoay ảnh trước khi mở.')
        mode = image.mode
        notice = f'Đã chuyển ảnh từ {mode} sang RGB 8-bit bằng Pillow.' if mode != 'RGB' else None
        rgb = np.array(image.convert('RGB') if mode != 'RGB' else image, dtype=np.uint8)
    validate_rgb(rgb)
    large = max(rgb.shape[:2]) > 256
    return LoadedImage(rgb, path.resolve(), hashlib.sha256(raw).hexdigest(),
                       mode, notice, large, LARGE_IMAGE_WARNING if large else None)


@dataclass(frozen=True)
class GuiResult:
    algorithm: str
    rgb: np.ndarray
    demo_seconds: float
    metrics: dict | None
    message: str
    checkpoint_sha256: str | None
    source_notice: str | None
    input_rgb_sha256: str
    hr_rgb_sha256: str | None
    time_label: str = DEMO_TIME_LABEL
    metric_label: str = METRIC_LABEL

    @property
    def elapsed_seconds(self) -> float:
        return self.demo_seconds

    @property
    def metric_message(self) -> str:
        return self.message


def process_result(lr: np.ndarray, prediction, elapsed_seconds: float,
                   hr: np.ndarray | None = None) -> GuiResult:
    """Keep pipeline pixels unchanged and call the accepted M1 evaluator."""
    validate_rgb(lr)
    validate_rgb(prediction.rgb)
    expected = (lr.shape[0] * 4, lr.shape[1] * 4, 3)
    if prediction.rgb.shape != expected:
        raise ValueError('Kết quả phải có kích thước gấp 4 lần ảnh LR.')
    metrics, message = None, NO_VALID_HR
    if hr is not None and hr.shape == expected:
        try:
            metrics = evaluate(hr, prediction.rgb)
            message = METRIC_LABEL
        except ValueError:
            LOGGER.exception('M1 reference cannot be evaluated')
            message = 'Không tính được PSNR/SSIM. Kiểm tra HR; vùng sau bỏ viền cần ít nhất 11x11.'
    if hr is not None:
        message += '\n' + hr_caveat()
    return GuiResult(prediction.algorithm, prediction.rgb, elapsed_seconds,
                     metrics, message, prediction.checkpoint_sha256,
                     prediction.source_notice, _array_hash(lr),
                     _array_hash(hr) if hr is not None else None)


def predict_for_gui(registry, algorithm: str, lr: np.ndarray,
                    hr: np.ndarray | None = None) -> GuiResult:
    """The same pure array entry point is used by workers and parity checks."""
    started = perf_counter()
    prediction = registry.predict(algorithm, lr)
    elapsed = perf_counter() - started
    return process_result(lr, prediction, elapsed, hr)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return 'Infinity' if value > 0 else '-Infinity' if value < 0 else 'NaN'
    return value


def save_result(result: GuiResult, input_image: LoadedImage,
                output_dir: Path, *, hr_image: LoadedImage | None = None) -> tuple[Path, Path]:
    """Export into a newly reserved directory; never replace any existing path."""
    if _array_hash(input_image.rgb) != result.input_rgb_sha256:
        raise ValueError('Ảnh đầu vào không khớp với ảnh đã xử lý.')
    if hr_image is not None and _array_hash(hr_image.rgb) != result.hr_rgb_sha256:
        raise ValueError('Ảnh HR không khớp với ảnh tham chiếu đã dùng.')
    metadata = {
        'algorithm': result.algorithm, 'scale': 4,
        'checkpoint_sha256': result.checkpoint_sha256,
        'source_notice': result.source_notice,
        'input_path': str(input_image.path),
        'input_file_sha256': input_image.file_sha256,
        'input_rgb_sha256': result.input_rgb_sha256,
        'input_shape': list(input_image.rgb.shape),
        'input_conversion_notice': input_image.conversion_notice,
        'hr_path': str(hr_image.path) if hr_image else None,
        'hr_file_sha256': hr_image.file_sha256 if hr_image else None,
        'hr_rgb_sha256': result.hr_rgb_sha256,
        'output_rgb_sha256': _array_hash(result.rgb),
        'output_shape': list(result.rgb.shape),
        'metrics': result.metrics, 'metric_message': result.message,
        'metric_label': result.metric_label, 'hr_caveat': hr_caveat(),
        'demo_seconds': result.demo_seconds, 'time_label': result.time_label,
        'versions': {'python': platform.python_version(),
                     **{name: version(name) for name in
                        ('numpy', 'Pillow', 'torch', 'scikit-image')}},
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    png, sidecar = output_dir / 'SR.png', output_dir / 'metadata.json'
    try:
        save_rgb(png, result.rgb)
        metadata['output_file_sha256'] = hashlib.sha256(png.read_bytes()).hexdigest()
        with sidecar.open('x', encoding='utf-8') as stream:
            json.dump(_json_safe(metadata), stream, ensure_ascii=False,
                      indent=2, allow_nan=False)
    except Exception:
        LOGGER.exception('Demo export failed in %s', output_dir)
        raise
    return png, sidecar


@dataclass(frozen=True)
class ControllerEvent:
    kind: str
    algorithm: str | None = None
    message: str = ''
    result: GuiResult | None = None
    stopped: bool = False
    detail: str = ''
    completed: int = 0
    errors: int = 0
    total: int = 0


class DemoController:
    """One background run at a time; all event delivery stays on the owner thread."""
    def __init__(self, registry=None, *, poll_ms: int = 25):
        self.registry = registry if registry is not None else InferenceRegistry()
        self.events = Queue()
        self.busy = False
        self.poll_ms = poll_ms
        self._owner = threading.main_thread().ident
        self._stop = threading.Event()
        self._attachment = 0
        self._poll_root = None
        self._poll_id = None

    def _require_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('Thao tác này phải chạy trên luồng chính.')

    def start(self, lr: np.ndarray, algorithms, hr: np.ndarray | None = None,
              *, confirmed_large: bool = False) -> None:
        self._require_owner()
        if self.busy:
            raise RuntimeError('Đang xử lý ảnh. Hãy đợi lượt hiện tại kết thúc.')
        algorithms = tuple(algorithms)
        if not 1 <= len(algorithms) <= 5 or len(set(algorithms)) != len(algorithms):
            raise ValueError('Hãy chọn từ 1 đến 5 thuật toán khác nhau.')
        if any(name not in ALGORITHMS for name in algorithms):
            raise ValueError('Thuật toán đã chọn không hợp lệ.')
        validate_rgb(lr)
        if hr is not None:
            validate_rgb(hr)
        if max(lr.shape[:2]) > 256 and not confirmed_large:
            raise ValueError('Ảnh LR lớn. Cần xác nhận trước khi chạy.')
        # Own snapshots so later caller edits cannot change inference provenance.
        lr, hr = lr.copy(), hr.copy() if hr is not None else None
        self._stop.clear()
        self.busy = True
        try:
            threading.Thread(target=self._run, args=(lr, algorithms, hr),
                             name='sr-demo-inference', daemon=True).start()
        except Exception:
            self.busy = False
            LOGGER.exception('Cannot start background inference')
            raise

    def stop(self) -> None:
        self._require_owner()
        self._stop.set()

    def _run(self, lr, algorithms, hr):
        algorithm = None
        completed = errors = 0
        try:
            for algorithm in algorithms:
                if self._stop.is_set():
                    break
                self.events.put(ControllerEvent('status', algorithm, f'Đang chạy {algorithm}'))
                try:
                    result = predict_for_gui(self.registry, algorithm, lr, hr)
                    self.events.put(ControllerEvent('result', algorithm, result=result))
                except Exception as error:
                    errors += 1
                    self._report_error(algorithm, error)
                    self._release_model(algorithm)
                finally:
                    completed += 1
        finally:
            try:
                self._release_model(algorithm)
            finally:
                stopped = self._stop.is_set()
                message = ('Đã dừng. ' if stopped else '')
                message += f'Đã xử lý {completed}/{len(algorithms)}. Thành công {completed - errors}, lỗi {errors}.'
                self.events.put(ControllerEvent('done', message=message, stopped=stopped,
                                                completed=completed, errors=errors,
                                                total=len(algorithms)))

    def _release_model(self, algorithm):
        try:
            # Release only on this worker, after predict has returned or raised.
            self.registry.release()
        except Exception as error:
            self._report_error(algorithm, error)

    def _report_error(self, algorithm, error):
        LOGGER.exception('Demo inference failed for %s', algorithm)
        if isinstance(error, MemoryError):
            message = 'Không đủ RAM. Hãy chọn ảnh nhỏ hơn hoặc ít thuật toán hơn.'
        elif isinstance(error, FileNotFoundError):
            message = 'Thiếu checkpoint. Xem scripts/download_models.py.'
        else:
            message = 'Không thể xử lý ảnh. Mở chi tiết để kiểm tra.'
        self.events.put(ControllerEvent('error', algorithm, message,
                                       detail=f'{type(error).__name__}: {error}'))

    def drain_events(self) -> list[ControllerEvent]:
        self._require_owner()
        pending = []
        while True:
            try:
                event = self.events.get_nowait()
            except Empty:
                return pending
            if event.kind == 'done':
                self.busy = False
            pending.append(event)

    def attach(self, root, callback) -> None:
        self._require_owner()
        self.detach()
        self._poll_root = root
        self._attachment += 1
        attachment = self._attachment
        def poll():
            self._poll_id = None
            if attachment != self._attachment:
                return
            for event in self.drain_events():
                try:
                    callback(event)
                except Exception:
                    LOGGER.exception('Demo event callback failed for %s', event.kind)
            if attachment == self._attachment:
                self._poll_id = root.after(self.poll_ms, poll)
        self._poll_id = root.after(self.poll_ms, poll)

    def detach(self) -> None:
        self._require_owner()
        self._attachment += 1
        root, identifier = self._poll_root, self._poll_id
        self._poll_root = self._poll_id = None
        if root is not None and identifier is not None:
            root.after_cancel(identifier)
