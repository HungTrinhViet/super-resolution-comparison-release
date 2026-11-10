"""Background orchestration without any view or styled widgets."""
from collections import deque
import gc
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from sr_compare.gui.controller import ControllerEvent, DemoController


class FakeRoot:
    def __init__(self):
        self.callbacks = deque()
        self.owner = threading.get_ident()

    def after(self, milliseconds, callback):
        assert threading.get_ident() == self.owner
        self.callbacks.append(callback)

    def tick(self):
        self.callbacks.popleft()()


class Backend:
    def __init__(self, error=None):
        self.started = threading.Event()
        self.finish = threading.Event()
        self.error = error
        self.calls = []
        self.active = False
        self.released = False

    def predict(self, algorithm, lr):
        self.active = True
        self.calls.append((algorithm, threading.get_ident()))
        self.started.set()
        assert self.finish.wait(3), 'test must unblock inference'
        try:
            if self.error is not None:
                raise self.error
            return SimpleNamespace(rgb=np.repeat(np.repeat(lr, 4, axis=0), 4, axis=1),
                                   algorithm=algorithm, checkpoint_sha256=None,
                                   source_notice=None)
        finally:
            self.active = False

    def release(self):
        assert not self.active
        self.released = True


def pump(root, controller):
    deadline = time.perf_counter() + 3
    while controller.busy and time.perf_counter() < deadline:
        root.tick()
        time.sleep(.002)
    assert not controller.busy


def test_sequential_queue_main_thread_callbacks_and_busy_guard():
    backend = Backend()
    controller = DemoController(backend)
    root = FakeRoot()
    events = []
    owner = threading.get_ident()
    controller.attach(root, lambda event: events.append((event, threading.get_ident())))
    controller.start(np.zeros((6, 7, 3), dtype=np.uint8), ['Bicubic', 'SRCNN'])
    assert backend.started.wait(2)
    with pytest.raises(RuntimeError, match='Đang xử lý'):
        controller.start(np.zeros((6, 7, 3), dtype=np.uint8), ['Bicubic'])
    root.tick()
    assert controller.busy
    assert events[0][0].kind == 'status'
    assert not backend.released
    backend.finish.set()
    pump(root, controller)
    assert [name for name, _ in backend.calls] == ['Bicubic', 'SRCNN']
    assert all(thread != owner for _, thread in backend.calls)
    assert all(thread == owner for _, thread in events)
    assert [e.kind for e, _ in events] == ['status', 'result', 'status', 'result', 'done']
    assert all(e.result.elapsed_seconds >= 0 for e, _ in events if e.kind == 'result')
    assert backend.released
    assert (events[-1][0].completed, events[-1][0].errors, events[-1][0].total) == (2, 0, 2)


def test_stop_waits_for_current_prediction_and_releases_after_it():
    backend = Backend()
    controller = DemoController(backend)
    root, events = FakeRoot(), []
    controller.attach(root, events.append)
    controller.start(np.zeros((6, 7, 3), dtype=np.uint8), ['Bicubic', 'SRCNN'])
    assert backend.started.wait(2)
    controller.stop()
    root.tick()
    assert controller.busy and not backend.released
    backend.finish.set()
    pump(root, controller)
    assert len(backend.calls) == 1
    assert [e.kind for e in events] == ['status', 'result', 'done']
    assert events[-1].stopped
    assert (events[-1].completed, events[-1].errors, events[-1].total) == (1, 0, 2)
    assert backend.released


@pytest.mark.parametrize('error', [MemoryError('synthetic RAM exhaustion'),
                                  ValueError('synthetic hash mismatch'),
                                  FileNotFoundError('synthetic missing checkpoint')])
def test_errors_are_logged_delivered_and_allow_recovery(error, caplog):
    backend = Backend(error)
    backend.finish.set()
    controller = DemoController(backend)
    root, events = FakeRoot(), []
    controller.attach(root, events.append)
    lr = np.zeros((6, 7, 3), dtype=np.uint8)
    controller.start(lr, ['Bicubic', 'SRCNN'])
    pump(root, controller)
    errors = [e for e in events if e.kind == 'error']
    assert len(errors) == 2
    assert str(error) not in errors[0].message
    assert type(error).__name__ not in errors[0].message
    assert errors[0].detail == f'{type(error).__name__}: {error}'
    expected = ('Không đủ RAM. Hãy chọn ảnh nhỏ hơn hoặc ít thuật toán hơn.'
                if isinstance(error, MemoryError) else
                'Thiếu checkpoint. Xem scripts/download_models.py.'
                if isinstance(error, FileNotFoundError) else
                'Không thể xử lý ảnh. Mở chi tiết để kiểm tra.')
    assert errors[0].message == expected
    if isinstance(error, FileNotFoundError):
        assert 'scripts/download_models.py' in errors[0].message
    assert str(error) in caplog.text
    assert backend.released and len(backend.calls) == 2
    assert (events[-1].completed, events[-1].errors, events[-1].total) == (2, 2, 2)
    backend.error = None
    controller.start(lr, ['Bicubic'])
    pump(root, controller)
    assert events[-2].kind == 'result'


@pytest.mark.parametrize('algorithms', [[], ['unknown'], ['Bicubic'] * 6,
                                      ['Bicubic', 'Bicubic']])
def test_selection_validation(algorithms):
    controller = DemoController(Backend())
    with pytest.raises(ValueError):
        controller.start(np.zeros((6, 7, 3), dtype=np.uint8), algorithms)
    assert not controller.busy


def test_large_lr_confirmation_is_required_before_start():
    backend = Backend()
    controller = DemoController(backend)
    lr = np.zeros((2, 257, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match='xác nhận'):
        controller.start(lr, ['Bicubic'])
    assert not controller.busy and not backend.started.is_set()
    backend.finish.set()
    root = FakeRoot()
    controller.attach(root, lambda event: None)
    controller.start(lr, ['Bicubic'], confirmed_large=True)
    pump(root, controller)


def test_actual_tk_after_remains_responsive_during_background_prediction():
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    backend = Backend()
    controller = DemoController(backend, poll_ms=5)
    events, heartbeats = [], []
    owner = threading.get_ident()
    try:
        controller.attach(root, lambda event: events.append((event, threading.get_ident())))
        controller.start(np.zeros((6, 7, 3), dtype=np.uint8), ['Bicubic'])
        assert backend.started.wait(2)
        def heartbeat():
            heartbeats.append((threading.get_ident(), controller.busy, backend.active))
            backend.finish.set()
        root.after(15, heartbeat)
        deadline = time.perf_counter() + 3
        while controller.busy and time.perf_counter() < deadline:
            root.update()
            time.sleep(.002)
        assert not controller.busy
        assert heartbeats == [(owner, True, True)]
        assert all(thread == owner for _, thread in events)
        assert any(e.kind == 'result' for e, _ in events)
    finally:
        backend.finish.set()
        controller.detach()
        root.destroy()
        # Collect Tk callback cycles on the main thread before later workers allocate.
        del root
        gc.collect()


def test_controller_created_on_worker_cannot_attach_tk_on_worker():
    errors = []
    def wrong_thread():
        try:
            DemoController(Backend()).attach(FakeRoot(), lambda event: None)
        except RuntimeError as error:
            errors.append(str(error))
    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join(timeout=2)
    assert len(errors) == 1
    assert 'luồng chính' in errors[0]


def test_release_error_still_delivers_done_and_restores_busy(caplog):
    backend = Backend()
    backend.finish.set()
    def broken_release():
        raise RuntimeError('synthetic release failure')
    backend.release = broken_release
    controller = DemoController(backend)
    root, events = FakeRoot(), []
    controller.attach(root, events.append)
    controller.start(np.zeros((6, 7, 3), dtype=np.uint8), ['Bicubic'])
    pump(root, controller)
    assert [event.kind for event in events] == ['status', 'result', 'error', 'done']
    assert 'synthetic release failure' in caplog.text


def test_bad_callback_cannot_prevent_done_delivery(caplog):
    backend = Backend()
    backend.finish.set()
    controller = DemoController(backend)
    root, events = FakeRoot(), []
    def callback(event):
        events.append(event)
        if event.kind == 'result':
            raise ValueError('synthetic callback failure')
    controller.attach(root, callback)
    controller.start(np.zeros((6, 7, 3), dtype=np.uint8), ['Bicubic'])
    pump(root, controller)
    assert events[-1].kind == 'done'
    assert 'synthetic callback failure' in caplog.text


def test_detach_stops_polling_without_touching_worker():
    controller = DemoController(Backend())
    root = FakeRoot()
    controller.attach(root, lambda event: pytest.fail('detached callback'))
    controller.detach()
    root.tick()
    assert not root.callbacks


@pytest.mark.parametrize('error', [MemoryError('synthetic RAM exhaustion'),
                                  ValueError('synthetic hash mismatch'),
                                  FileNotFoundError('synthetic missing checkpoint')])
def test_failed_algorithm_releases_then_continues_to_success(error):
    backend = Backend()
    backend.finish.set()
    original_predict = backend.predict
    trace = []

    def predict(algorithm, lr):
        trace.append(algorithm)
        backend.error = error if algorithm == 'SRCNN' else None
        return original_predict(algorithm, lr)

    def release():
        assert not backend.active
        trace.append('release')

    backend.predict, backend.release = predict, release
    controller = DemoController(backend)
    root, events = FakeRoot(), []
    controller.attach(root, events.append)
    controller.start(np.zeros((6, 7, 3), dtype=np.uint8), ['SRCNN', 'Bicubic'])
    pump(root, controller)

    assert trace == ['SRCNN', 'release', 'Bicubic', 'release']
    assert [event.kind for event in events] == ['status', 'error', 'status', 'result', 'done']
    assert events[-2].algorithm == 'Bicubic'
    assert (events[-1].completed, events[-1].errors, events[-1].total) == (2, 1, 2)
    assert events[-1].message == 'Đã xử lý 2/2. Thành công 1, lỗi 1.'
    assert not events[-1].stopped


def test_failure_and_broken_release_still_continue_and_finish(caplog):
    backend = Backend()
    backend.finish.set()
    original_predict = backend.predict
    releases = []

    def predict(algorithm, lr):
        backend.error = ValueError('synthetic first refusal') if algorithm == 'SRCNN' else None
        return original_predict(algorithm, lr)

    def release():
        assert not backend.active
        releases.append(True)
        raise RuntimeError('synthetic cleanup failure')

    backend.predict, backend.release = predict, release
    controller = DemoController(backend)
    root, events = FakeRoot(), []
    controller.attach(root, events.append)
    controller.start(np.zeros((6, 7, 3), dtype=np.uint8), ['SRCNN', 'Bicubic'])
    pump(root, controller)

    assert len(releases) == 2
    assert [name for name, _ in backend.calls] == ['SRCNN', 'Bicubic']
    assert any(event.kind == 'result' and event.algorithm == 'Bicubic' for event in events)
    assert events[-1].kind == 'done'
    assert (events[-1].completed, events[-1].errors, events[-1].total) == (2, 1, 2)
    assert 'synthetic cleanup failure' in caplog.text


def test_stop_after_failed_current_algorithm_reports_processed_counts():
    backend = Backend(ValueError('synthetic first refusal'))
    controller = DemoController(backend)
    root, events = FakeRoot(), []
    controller.attach(root, events.append)
    controller.start(np.zeros((6, 7, 3), dtype=np.uint8), ['SRCNN', 'Bicubic'])
    assert backend.started.wait(2)
    controller.stop()
    backend.finish.set()
    pump(root, controller)
    assert [name for name, _ in backend.calls] == ['SRCNN']
    assert events[-1].stopped
    assert (events[-1].completed, events[-1].errors, events[-1].total) == (1, 1, 2)
    assert events[-1].message == 'Đã dừng. Đã xử lý 1/2. Thành công 0, lỗi 1.'


def test_event_additions_preserve_existing_defaults():
    event = ControllerEvent('status')
    assert event.detail == ''
    assert (event.completed, event.errors, event.total) == (0, 0, 0)
