"""A detached controller must cancel its scheduled Tk callback."""
from sr_compare.gui.controller import DemoController


class ScheduledRoot:
    def __init__(self):
        self.pending = {}

    def after(self, milliseconds, callback):
        key = 'scheduled-'+str(len(self.pending))
        self.pending[key] = callback
        return key

    def after_cancel(self, key):
        del self.pending[key]


def test_detach_cancels_tk_callback_and_clears_root_reference():
    root = ScheduledRoot()
    controller = DemoController()
    controller.attach(root, lambda event: None)
    assert root.pending
    controller.detach()
    assert not root.pending
