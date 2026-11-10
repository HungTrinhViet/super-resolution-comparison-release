"""Real Tk view checks with synthetic pixels and a controlled CPU backend."""
import importlib.util
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest


class Backend:
    def __init__(self, failing=(), blocked=False):
        self.failing = failing
        self.gate = threading.Event()
        self.started = threading.Event()
        self.calls = []
        if not blocked:
            self.gate.set()

    def predict(self, algorithm, lr):
        self.started.set()
        assert self.gate.wait(4)
        self.calls.append((algorithm, threading.get_ident()))
        if algorithm in self.failing:
            raise FileNotFoundError('synthetic checkpoint failure')
        return SimpleNamespace(algorithm=algorithm,
                               rgb=np.repeat(np.repeat(lr, 4, 0), 4, 1),
                               checkpoint_sha256=None, source_notice='synthetic source')

    def release(self):
        pass


@pytest.fixture
def view():
    assert importlib.util.find_spec('sr_compare.gui.app') is not None, 'Real GUI view is missing'
    import tkinter as tk
    from sr_compare.gui.app import DemoApp
    root = tk.Tk()
    backend = Backend()
    app = DemoApp(root, registry=backend)
    root.update()
    yield app, root, backend
    backend.gate.set()
    deadline = time.perf_counter() + 4
    while app.controller.busy and time.perf_counter() < deadline:
        root.update()
        time.sleep(.005)
    app.controller.detach()
    root.destroy()


def image_file(tmp_path, name='lr.png', size=(12, 10), mode='RGB'):
    path = tmp_path / name
    pixels = np.arange(size[0] * size[1] * 3, dtype=np.uint8).reshape(size[1], size[0], 3)
    image = Image.fromarray(pixels).convert(mode)
    image.save(path)
    return path


def finish(app, root):
    deadline = time.perf_counter() + 4
    while app.controller.busy and time.perf_counter() < deadline:
        root.update()
        time.sleep(.005)
    root.update()
    assert not app.controller.busy


def test_empty_view_has_seven_aligned_columns_and_locked_actions(view):
    app, root, _ = view
    assert root.minsize() == (1100, 700)
    assert len(app.canvases) == 2
    assert app.run_button.instate(['disabled'])
    assert app.stop_button.instate(['disabled'])
    assert len(app.tree['columns']) == 7
    assert app.tree.winfo_ismapped()
    assert not app.tree.get_children()
    for index, column in enumerate(app.tree['columns']):
        assert str(app.tree.column(column, 'anchor')) == ('w' if index < 2 else 'e')
    for canvas in app.canvases:
        assert any(canvas.itemcget(item, 'text') == 'Chọn ảnh LR để bắt đầu'
                   for item in canvas.find_all() if canvas.type(item) == 'text')


def test_load_conversion_wrong_hr_and_no_hr_metrics(view, tmp_path):
    app, root, _ = view
    app.load_lr(image_file(tmp_path, mode='L'))
    assert app.run_button.instate(['!disabled'])
    assert 'Đã chuyển sang RGB từ L' in app.lr_info.get()
    app.load_hr(image_file(tmp_path, name='wrong.png', size=(13, 12)))
    assert 'HR phải đúng 4 × LR (cần 48 × 40); không tính PSNR/SSIM' in app.hr_info.get()
    app.set_algorithms(['Bicubic'])
    app.run_selected()
    finish(app, root)
    assert app.results['Bicubic'].metrics is None
    assert tuple(app.tree.item('Bicubic', 'values')[2:6]) == ('-',) * 4


def test_worker_failure_continues_and_done_counts_successes(view, tmp_path):
    app, root, backend = view
    backend.failing = ('SRCNN',)
    app.load_lr(image_file(tmp_path))
    app.set_algorithms(['Bicubic', 'SRCNN', 'FSRCNN'])
    app.run_selected()
    finish(app, root)
    assert set(app.results) == {'Bicubic', 'FSRCNN'}
    assert set(app.errors) == {'SRCNN'}
    assert app.status.get() == 'Hoàn tất 2/3, 1 lỗi'
    assert app.tree.item('SRCNN', 'values')[1].startswith('Lỗi: ')
    assert app.tree.item('SRCNN', 'tags') == ('error',)
    assert all(owner != threading.get_ident() for _, owner in backend.calls)


def test_running_locks_inputs_stop_preserves_current_result(view, tmp_path):
    app, root, backend = view
    backend.gate.clear()
    app.load_lr(image_file(tmp_path))
    app.set_algorithms(['Bicubic', 'SRCNN'])
    app.run_selected()
    assert backend.started.wait(2)
    root.update()
    assert app.run_button.instate(['disabled'])
    assert app.stop_button.instate(['!disabled'])
    assert app.load_lr_button.instate(['disabled'])
    assert all(widget.instate(['disabled']) for widget in app.algorithm_checks.values())
    app.stop_button.invoke()
    backend.gate.set()
    finish(app, root)
    assert set(app.results) == {'Bicubic'}
    assert app.tree.item('SRCNN', 'values')[1] == 'Đã dừng'
    assert app.stop_button.instate(['disabled'])


def test_metric_caption_only_for_valid_hr_without_moving_table(view, tmp_path):
    from sr_compare.gui.controller import METRIC_LABEL

    app, root, _ = view
    assert app.metric_caption.cget('text') == ''
    app.load_lr(image_file(tmp_path))
    root.update()
    table_y = app.tree.winfo_rooty()
    assert app.metric_caption.cget('text') == ''

    app.load_hr(image_file(tmp_path, name='valid_hr.png', size=(48, 40)))
    root.update()
    assert app.metric_caption.cget('text') == METRIC_LABEL
    assert app.metric_caption.winfo_viewable()
    assert app.metric_caption.winfo_rooty() + app.metric_caption.winfo_height() <= table_y
    assert app.tree.winfo_rooty() == table_y

    app.load_hr(image_file(tmp_path, name='invalid_hr.png', size=(47, 40)))
    root.update()
    assert app.metric_caption.cget('text') == ''
    assert app.tree.winfo_rooty() == table_y


def test_valid_hr_metrics_warning_and_save_uses_original_pixels(view, tmp_path):
    app, root, _ = view
    path = image_file(tmp_path)
    app.load_lr(path)
    hr = np.repeat(np.repeat(app.lr_image.rgb, 4, 0), 4, 1)
    hr_path = tmp_path / 'hr.png'
    Image.fromarray(hr).save(hr_path)
    app.load_hr(hr_path)
    assert 'đúng 4 × LR' in app.hr_info.get()
    assert app.hr_warning.winfo_manager()
    app.set_algorithms(['Bicubic'])
    app.run_selected()
    finish(app, root)
    app.set_zoom(8)
    app.left_choice.set('LR')
    app.right_choice.set('Bicubic')
    root.update()
    png, sidecar = app.save_algorithm('Bicubic', tmp_path / 'saved')
    assert np.array_equal(np.array(Image.open(png)), hr)
    metadata = json.loads(sidecar.read_text(encoding='utf-8'))
    assert metadata['source_notice'] == 'synthetic source'
    assert metadata['time_label'] == 'Thời gian demo trong GUI, không phải kết quả benchmark đã nghiệm thu.'
    assert metadata['metrics']['y']['ssim'] == 1


def test_large_input_requires_confirmation_without_backend_call(view, tmp_path, monkeypatch):
    app, root, backend = view
    app.load_lr(image_file(tmp_path, size=(257, 4)))
    app.set_algorithms(['Bicubic'])
    from sr_compare.gui import app as module
    monkeypatch.setattr(module.messagebox, 'askyesno', lambda *a, **k: False)
    app.run_selected()
    root.update()
    assert not backend.calls
    assert not app.controller.busy
    assert app.large_warning.winfo_manager()


def test_shared_viewport_zoom_pan_and_bounded_lr_render(view, tmp_path):
    app, root, _ = view
    app.load_lr(image_file(tmp_path, size=(80, 70)))
    app.left_choice.set('LR')
    app.right_choice.set('LR')
    app.set_zoom(8)
    root.update()
    canvas = app.canvases[0]
    canvas.event_generate('<ButtonPress-1>', x=150, y=150)
    canvas.event_generate('<B1-Motion>', x=90, y=110)
    root.update()
    assert app.viewport.zoom == 8
    assert app.viewport.offset != (0, 0)
    assert app.zoom_buttons[8]['style'] == 'SegOn.TButton'
    for photo, target in zip(app._photos, app.canvases):
        assert photo.width() <= target.winfo_width()
        assert photo.height() <= target.winfo_height()
    app.canvases[1].event_generate('<MouseWheel>', delta=-120, x=80, y=80)
    root.update()
    assert app.viewport.zoom == 4


def test_source_dialog_uses_exact_repository_notices(view, monkeypatch):
    app, _, _ = view
    from pathlib import Path
    from sr_compare import esrgan
    from sr_compare.gui import app as module
    dialogs = []
    monkeypatch.setattr(module.messagebox, 'showinfo', lambda title, body, **kw: dialogs.append((title, body)))
    app.show_source('FSRCNN')
    app.show_source('ESRGAN')
    fsrcnn_notice = next(line for line in Path('docs/M4_FSRCNN_RESULTS.md').read_text(encoding='utf-8').splitlines()
                         if line.startswith('Trọng số lấy từ bản đăng lại'))
    assert dialogs == [('Thông tin nguồn', fsrcnn_notice),
                       ('Thông tin nguồn', esrgan.SOURCE_NOTICE)]
