"""Real Tkinter M10 demo view; inference and export stay in the accepted core."""
from datetime import datetime
import gc
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .. import esrgan
from ..inference import ALGORITHMS
from .controller import DEMO_TIME_LABEL, METRIC_LABEL, DemoController, hr_caveat, load_input, save_result
from .styles import COLORS, configure_styles
from .viewport import Viewport


DISPLAY_NAMES = {name: 'EDSR-baseline' if name == 'EDSR' else name for name in ALGORITHMS}
HINTS = ('Nội suy, không học', '3 lớp tích chập, kênh Y',
         'Tích chập chuyển vị, kênh Y', '16 khối dư, RGB',
         '23 khối RRDB, RGB, chậm nhất')
LARGE_WARNING = 'Ảnh lớn: ESRGAN có thể chạy lâu trên CPU'
FSRCNN_NOTICE = (
    'Trọng số lấy từ bản đăng lại của gói FSRCNN_test (Chao Dong, ECCV 2016); '
    'hai bản sao độc lập trùng SHA-256; nguồn gốc trực tiếp từ CUHK và '
    'cách chọn checkpoint chưa được xác minh; chỉ sử dụng cho mục đích học thuật.'
)


def _visible(text):
    return str(text).replace(chr(0x2014), ';').replace(chr(0x2013), '-')


class DemoApp:
    def __init__(self, root, registry=None):
        self.root = root
        self.controller = DemoController(registry)
        self.viewport = Viewport()
        self.lr_image = None
        self.hr_image = None
        self.results = {}
        self.errors = {}
        self._error_details = {}
        self._selected = ()
        self._running = False
        self._photos = [None, None]
        self._display_images = {}
        self._drag = None
        self._closing = False
        self._save_counter = 0
        self.font_family = configure_styles(root)
        root.title('So sánh Super Resolution ×4')
        root.geometry('1360x820')
        root.minsize(1100, 700)
        self.lr_info = tk.StringVar(root, 'Chưa chọn ảnh')
        self.hr_info = tk.StringVar(root, 'Không có HR: không tính PSNR/SSIM')
        self.status = tk.StringVar(root, 'Sẵn sàng')
        self.left_choice = tk.StringVar(root, 'Bicubic')
        self.right_choice = tk.StringVar(root, 'EDSR')
        self.algorithm_vars = {name: tk.BooleanVar(root, name != 'ESRGAN') for name in ALGORITHMS}
        self.algorithm_checks = {}
        self.zoom_buttons = {}
        self._build_sidebar()
        self._build_main()
        self._bind_shortcuts()
        self.left_choice.trace_add('write', lambda *_: self._render())
        self.right_choice.trace_add('write', lambda *_: self._render())
        self.controller.attach(root, self._on_event)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self._update_actions()

    def _build_sidebar(self):
        side = ttk.Frame(self.root, style='Side.TFrame', padding=(20, 18))
        side.place(x=0, y=0, width=330, relheight=1)
        tk.Frame(self.root, bg=COLORS['border']).place(x=330, y=0, width=1, relheight=1)
        ttk.Label(side, text='Super Resolution ×4', style='Title.TLabel').pack(anchor='w')
        ttk.Label(side, text='Chương trình mô phỏng (demo)', style='Muted.TLabel').pack(
            anchor='w', pady=(2, 18))
        ttk.Label(side, text='Ảnh đầu vào (LR)', style='H.TLabel').pack(anchor='w')
        self.load_lr_button = ttk.Button(side, text='Chọn ảnh…', style='Secondary.TButton', command=self.load_lr)
        self.load_lr_button.pack(anchor='w', pady=(6, 2))
        self.lr_label = ttk.Label(side, textvariable=self.lr_info, style='Muted.TLabel',
                                  wraplength=286, justify='left')
        self.lr_label.pack(anchor='w', pady=(4, 4))
        self.large_warning = ttk.Label(side, text=LARGE_WARNING, style='Warn.TLabel',
                                       wraplength=266, justify='left')
        ttk.Frame(side, style='Side.TFrame', height=12).pack()
        ttk.Label(side, text='Ảnh HR tham chiếu (không bắt buộc)', style='H.TLabel').pack(anchor='w')
        self.load_hr_button = ttk.Button(side, text='Chọn HR…', style='Secondary.TButton', command=self.load_hr)
        self.load_hr_button.pack(anchor='w', pady=(6, 2))
        self.hr_label = ttk.Label(side, textvariable=self.hr_info, style='Muted.TLabel',
                                  wraplength=286, justify='left')
        self.hr_label.pack(anchor='w', pady=(4, 4))
        self.hr_warning = ttk.Label(side, text=hr_caveat(), style='Warn.TLabel', wraplength=266, justify='left')
        self.algorithm_heading = ttk.Label(side, text='Thuật toán', style='H.TLabel')
        self.algorithm_heading.pack(anchor='w', pady=(12, 4))
        for name, hint in zip(ALGORITHMS, HINTS):
            row = ttk.Frame(side, style='Side.TFrame')
            row.pack(fill='x', pady=1)
            check = ttk.Checkbutton(row, text=DISPLAY_NAMES[name], variable=self.algorithm_vars[name],
                                    style='Side.TCheckbutton', command=self._update_actions)
            check.pack(side='left')
            self.algorithm_checks[name] = check
            check.bind('<Double-1>', lambda event, algorithm=name: self._source_doubleclick(algorithm))
            ttk.Label(row, text=hint, style='Muted.TLabel').pack(side='right')
        bottom = ttk.Frame(side, style='Side.TFrame')
        bottom.pack(side='bottom', fill='x', before=side.winfo_children()[0])
        self.run_button = ttk.Button(bottom, text='Chạy ×4', style='Primary.TButton', command=self.run_selected)
        self.run_button.pack(fill='x')
        row = ttk.Frame(bottom, style='Side.TFrame')
        row.pack(fill='x', pady=(8, 0))
        self.stop_button = ttk.Button(row, text='Dừng sau bước này', style='Secondary.TButton', command=self.stop)
        self.stop_button.pack(side='left', fill='x', expand=True, padx=(0, 4))
        self.save_button = ttk.Button(row, text='Lưu kết quả…', style='Secondary.TButton', command=self.save_selected)
        self.save_button.pack(side='left', fill='x', expand=True, padx=(4, 0))

    def _build_main(self):
        main = ttk.Frame(self.root, padding=(20, 16))
        main.place(x=331, y=0, relwidth=1, width=-331, relheight=1)
        bar = ttk.Frame(main)
        bar.pack(fill='x')
        controls = ttk.Frame(bar)
        controls.grid(row=0, column=0, sticky='w')
        bar.columnconfigure(1, weight=1)
        self.selectors = []
        for label, choice in (('Trái', self.left_choice), ('Phải', self.right_choice)):
            ttk.Label(controls, text=label).pack(side='left', padx=(0, 6))
            selector = ttk.Combobox(controls, values=('LR', *ALGORITHMS, 'HR'),
                                    textvariable=choice, width=9, state='readonly')
            selector.pack(side='left', padx=(0, 18))
            self.selectors.append(selector)
        ttk.Label(controls, text='Phóng').pack(side='left', padx=(12, 6))
        for zoom in (1, 2, 4, 8):
            button = ttk.Button(controls, text=f'{zoom}×', width=3,
                                style='SegOn.TButton' if zoom == self.viewport.zoom else 'Seg.TButton',
                                command=lambda value=zoom: self.set_zoom(value))
            button.pack(side='left', padx=(0, 2))
            self.zoom_buttons[zoom] = button
        nearest = ttk.Label(bar, text='Hiển thị nearest, không làm mượt', style='MainMuted.TLabel')
        nearest.grid(row=0, column=1, sticky='e')
        def fit_toolbar(event):
            if event.width < controls.winfo_reqwidth() + nearest.winfo_reqwidth() + 12:
                nearest.grid_configure(row=1, column=0, columnspan=2, sticky='e', pady=(4, 0))
            else:
                nearest.grid_configure(row=0, column=1, columnspan=1, sticky='e', pady=0)
        bar.bind('<Configure>', fit_toolbar)
        view = ttk.Frame(main)
        view.pack(fill='both', expand=True, pady=(12, 10))
        view.columnconfigure(0, weight=1, uniform='panes')
        view.columnconfigure(1, weight=1, uniform='panes')
        view.rowconfigure(1, weight=1)
        self.pane_titles = []
        canvases = []
        for index in range(2):
            padding = (0, 6) if index == 0 else (6, 0)
            title = ttk.Label(view, style='Pane.TLabel')
            title.grid(row=0, column=index, sticky='w', pady=(0, 4), padx=padding)
            self.pane_titles.append(title)
            canvas = tk.Canvas(view, bg=COLORS['canvas'], highlightthickness=1,
                               highlightbackground=COLORS['border'], takefocus=True)
            canvas.grid(row=1, column=index, sticky='nsew', padx=padding)
            canvas.bind('<Configure>', lambda event: self._render())
            canvas.bind('<ButtonPress-1>', self._start_drag)
            canvas.bind('<B1-Motion>', self._pan)
            canvas.bind('<ButtonRelease-1>', lambda event: setattr(self, '_drag', None))
            canvas.bind('<MouseWheel>', self._wheel)
            canvas.bind('<Button-4>', lambda event: self._wheel(event, 1))
            canvas.bind('<Button-5>', lambda event: self._wheel(event, -1))
            canvases.append(canvas)
        self.canvases = tuple(canvases)
        ttk.Label(main, text=DEMO_TIME_LABEL, style='MainMuted.TLabel', wraplength=680).pack(
            anchor='w', pady=(0, 4))
        self.metric_caption = ttk.Label(main, text='', style='MainMuted.TLabel', wraplength=680)
        self.metric_caption.pack(anchor='w', pady=(0, 4))
        columns = ('algo', 'status', 'psnr_y', 'ssim_y', 'psnr_rgb', 'ssim_rgb', 'time')
        headings = ('Thuật toán', 'Trạng thái', 'PSNR Y (dB)', 'SSIM Y', 'PSNR RGB', 'SSIM RGB', 'Thời gian demo')
        widths = (145, 160, 110, 90, 105, 90, 135)
        table_frame = ttk.Frame(main)
        table_frame.pack(fill='x')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=5)
        for index, (column, heading, width) in enumerate(zip(columns, headings, widths)):
            anchor = 'w' if index < 2 else 'e'
            self.tree.heading(column, text=heading, anchor=anchor)
            self.tree.column(column, width=width, minwidth=width, anchor=anchor, stretch=True)
        self.tree.tag_configure('error', foreground=COLORS['danger'])
        self.tree.bind('<Double-1>', self._row_doubleclick)
        self.tree.pack(fill='x')
        self.table_scroll = ttk.Scrollbar(table_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(xscrollcommand=self.table_scroll.set)
        def fit_table(event):
            if event.width < sum(widths):
                if not self.table_scroll.winfo_manager():
                    self.table_scroll.pack(fill='x')
            else:
                self.table_scroll.pack_forget()
        table_frame.bind('<Configure>', fit_table)
        status = ttk.Frame(main)
        status.pack(fill='x', pady=(10, 0))
        ttk.Label(status, textvariable=self.status, style='MainMuted.TLabel').pack(side='left')
        self.progress = ttk.Progressbar(status, length=160, maximum=1, value=0)
        self.progress.pack(side='left', padx=12)
        ttk.Label(status, text='CPU · PyTorch 4/1 luồng', style='MainMuted.TLabel').pack(side='right')

    def _bind_shortcuts(self):
        for sequence, command in (('<Control-o>', self.load_lr), ('<Control-h>', self.load_hr),
                                  ('<Control-Return>', self.run_selected), ('<Escape>', self.stop),
                                  ('<Control-s>', self.save_selected)):
            self.root.bind(sequence, lambda event, callback=command: self._shortcut(callback))
        for zoom in (1, 2, 4, 8):
            self.root.bind(str(zoom), lambda event, value=zoom: self._zoom_shortcut(event, value))

    @staticmethod
    def _shortcut(callback):
        callback()
        return 'break'

    def _zoom_shortcut(self, event, value):
        if not isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox)):
            self.set_zoom(value)
            return 'break'

    def _choose_path(self, title):
        return filedialog.askopenfilename(parent=self.root, title=title,
                                         filetypes=[('Ảnh PNG/JPG/BMP', '*.png *.jpg *.jpeg *.bmp')])

    def load_lr(self, path=None):
        if self._running:
            return
        path = path if path is not None else self._choose_path('Chọn ảnh LR')
        if not path:
            return
        try:
            loaded = load_input(Path(path))
        except Exception as error:
            messagebox.showerror('Không mở được ảnh LR', 'Không thể đọc ảnh PNG/JPG/BMP hợp lệ.\n' + _visible(error), parent=self.root)
            return
        self.lr_image = loaded
        self._clear_results()
        height, width = loaded.rgb.shape[:2]
        info = f'{loaded.path.name}   {width} × {height} px, RGB'
        if loaded.original_mode != 'RGB':
            info += f'\nĐã chuyển sang RGB từ {loaded.original_mode}'
        self.lr_info.set(info)
        if loaded.requires_confirmation:
            self.large_warning.pack(fill='x', after=self.lr_label, pady=(2, 4))
        else:
            self.large_warning.pack_forget()
        self.viewport.image_size = (width * 4, height * 4)
        self.viewport.offset = (0.0, 0.0)
        self._update_hr_info()
        self.status.set('Sẵn sàng')
        self._update_actions()
        self._render()

    def load_hr(self, path=None):
        if self._running:
            return
        path = path if path is not None else self._choose_path('Chọn ảnh HR tham chiếu')
        if not path:
            return
        try:
            loaded = load_input(Path(path))
        except Exception as error:
            messagebox.showerror('Không mở được ảnh HR', 'Không thể đọc ảnh PNG/JPG/BMP hợp lệ.\n' + _visible(error), parent=self.root)
            return
        self.hr_image = loaded
        self._clear_results()
        self._update_hr_info()
        self.status.set('Sẵn sàng')
        self._update_actions()
        self._render()

    def _clear_results(self):
        self.results.clear()
        self.errors.clear()
        self._error_details.clear()
        self._display_images.clear()
        self.tree.delete(*self.tree.get_children())
        self.progress.configure(value=0)

    def _update_hr_info(self):
        self.metric_caption.configure(text='')
        self.hr_warning.pack_forget()
        self.hr_label.configure(style='Muted.TLabel')
        if self.hr_image is None:
            self.hr_info.set('Không có HR: không tính PSNR/SSIM')
            return
        height, width = self.hr_image.rgb.shape[:2]
        text = f'{self.hr_image.path.name}   {width} × {height} px'
        if self.lr_image is None:
            text += '\nChọn LR để kiểm tra kích thước HR'
        elif self.hr_image.rgb.shape != (self.lr_image.rgb.shape[0] * 4, self.lr_image.rgb.shape[1] * 4, 3):
            expected_w, expected_h = self.viewport.image_size
            text += f'\nHR phải đúng 4 × LR (cần {expected_w} × {expected_h}); không tính PSNR/SSIM'
            self.hr_label.configure(style='Danger.TLabel')
        else:
            text += ', đúng 4 × LR'
            self.hr_warning.pack(fill='x', before=self.algorithm_heading, pady=(2, 4))
            self.metric_caption.configure(text=METRIC_LABEL)
        if self.hr_image.original_mode != 'RGB':
            text += f'\nĐã chuyển sang RGB từ {self.hr_image.original_mode}'
        self.hr_info.set(text)

    def set_algorithms(self, algorithms):
        if self._running:
            return
        if any(name not in ALGORITHMS for name in algorithms):
            raise ValueError('Unknown algorithm selection.')
        for name, variable in self.algorithm_vars.items():
            variable.set(name in algorithms)
        self._update_actions()

    def _update_actions(self):
        selected = any(variable.get() for variable in self.algorithm_vars.values())
        self.run_button.state(['!disabled'] if self.lr_image is not None and selected and not self._running else ['disabled'])
        self.stop_button.state(['!disabled'] if self._running else ['disabled'])
        self.save_button.state(['!disabled'] if self.results and not self._running else ['disabled'])
        for widget in (self.load_lr_button, self.load_hr_button, *self.algorithm_checks.values(), *self.zoom_buttons.values()):
            widget.state(['disabled'] if self._running else ['!disabled'])
        for selector in self.selectors:
            selector.configure(state='disabled' if self._running else 'readonly')

    def run_selected(self):
        if self._running or self.lr_image is None:
            return
        algorithms = tuple(name for name in ALGORITHMS if self.algorithm_vars[name].get())
        if not algorithms:
            return
        confirmed = False
        if self.lr_image.requires_confirmation:
            confirmed = messagebox.askyesno('Xác nhận ảnh lớn', LARGE_WARNING + '\nTiếp tục chạy ảnh này?', parent=self.root)
            if not confirmed:
                return
        self._clear_results()
        self._selected = algorithms
        for name in algorithms:
            self.tree.insert('', 'end', iid=name, values=(DISPLAY_NAMES[name], 'Chờ', '-', '-', '-', '-', ''))
        self.progress.configure(maximum=len(algorithms), value=0)
        try:
            self.controller.start(self.lr_image.rgb, algorithms,
                                  self.hr_image.rgb if self.hr_image is not None else None,
                                  confirmed_large=confirmed)
        except Exception as error:
            messagebox.showerror('Không thể chạy', _visible(error), parent=self.root)
            self._update_actions()
            return
        self._running = True
        self.status.set(f'Đang chạy (0/{len(algorithms)})')
        self._update_actions()
        self._render()

    def stop(self):
        if self._running:
            self.controller.stop()
            self.stop_button.state(['disabled'])
            self.status.set('Sẽ dừng sau bước này')

    def _on_event(self, event):
        name = event.algorithm
        total = len(self._selected)
        if event.kind == 'status' and name:
            self.tree.set(name, 'status', 'Đang chạy…')
            position = self._selected.index(name) + 1
            self.status.set(f'Đang chạy {DISPLAY_NAMES[name]} ({position}/{total})')
        elif event.kind == 'result' and event.result is not None:
            result = event.result
            self.results[name] = result
            values = ['-'] * 4
            if result.metrics is not None:
                values = [f"{result.metrics['y']['psnr']:.2f}", f"{result.metrics['y']['ssim']:.4f}",
                          f"{result.metrics['rgb']['psnr']:.2f}", f"{result.metrics['rgb']['ssim']:.4f}"]
            self.tree.item(name, values=(DISPLAY_NAMES[name], 'Xong', *values, f'{result.demo_seconds:.2f} s'))
            self.progress.configure(value=sum(algorithm in self.results or algorithm in self.errors
                                              for algorithm in self._selected))
            self._render()
        elif event.kind == 'error':
            key = name if name is not None else 'CPU'
            self.errors[key] = _visible(event.message)
            self._error_details[key] = _visible(getattr(event, 'detail', '') or event.message)
            if name and self.tree.exists(name):
                self.tree.set(name, 'status', 'Lỗi: ' + _visible(event.message).split('.')[0])
                self.tree.item(name, tags=('error',))
            self.progress.configure(value=sum(algorithm in self.results or algorithm in self.errors
                                              for algorithm in self._selected))
        elif event.kind == 'done':
            self._running = False
            processed = getattr(event, 'completed', len(self.results) + len(self.errors))
            errors = getattr(event, 'errors', len(self.errors))
            completed = processed - errors
            total = getattr(event, 'total', total)
            for algorithm in self._selected:
                if algorithm not in self.results and algorithm not in self.errors:
                    self.tree.set(algorithm, 'status', 'Đã dừng' if event.stopped else 'Chờ')
            text = f'Đã dừng {completed}/{total}' if event.stopped else f'Hoàn tất {completed}/{total}'
            if errors:
                text += f', {errors} lỗi'
            self.status.set(text)
            self.progress.configure(value=processed)
            self._update_actions()
            if self._closing:
                self._destroy()

    def save_algorithm(self, algorithm, output_dir):
        if self._running:
            raise RuntimeError('Chờ thuật toán hiện tại hoàn tất trước khi lưu.')
        if algorithm not in self.results or self.lr_image is None:
            raise ValueError('Chưa có kết quả của thuật toán đã chọn.')
        return save_result(self.results[algorithm], self.lr_image, Path(output_dir), hr_image=self.hr_image)

    def save_selected(self):
        if self._running or not self.results:
            return
        selected = self.tree.selection()
        algorithm = selected[0] if selected and selected[0] in self.results else self.right_choice.get()
        if algorithm not in self.results:
            algorithm = next(iter(self.results))
        folder = filedialog.askdirectory(parent=self.root, title=f'Lưu kết quả {DISPLAY_NAMES[algorithm]}')
        if not folder:
            return
        self._save_counter += 1
        directory = Path(folder) / f'{algorithm}_{datetime.now():%Y%m%d_%H%M%S_%f}_{self._save_counter}'
        try:
            self.save_algorithm(algorithm, directory)
        except Exception as error:
            messagebox.showerror('Không lưu được kết quả', _visible(error), parent=self.root)
            return
        self.status.set(f'Đã lưu {DISPLAY_NAMES[algorithm]}: {directory.name}')

    def _source_doubleclick(self, name):
        if not self._running:
            self.show_source(name)
        return 'break'

    def show_source(self, algorithm):
        result = self.results.get(algorithm)
        notice = FSRCNN_NOTICE if algorithm == 'FSRCNN' else (result.source_notice if result is not None else None)
        if not notice:
            notice = {'ESRGAN': esrgan.SOURCE_NOTICE}.get(algorithm)
        messagebox.showinfo('Thông tin nguồn', notice or 'Thuật toán sử dụng pipeline đã khóa trong dự án.', parent=self.root)

    def _row_doubleclick(self, event):
        if self._running:
            return
        name = self.tree.identify_row(event.y)
        if name in self.errors:
            messagebox.showerror(f'Chi tiết lỗi {DISPLAY_NAMES.get(name, name)}',
                                 self.errors[name] + '\n\n' + self._error_details[name], parent=self.root)
        elif name:
            self.show_source(name)

    def set_zoom(self, zoom):
        if self._running:
            return
        self.viewport.set_zoom(zoom)
        for value, button in self.zoom_buttons.items():
            button.configure(style='SegOn.TButton' if value == zoom else 'Seg.TButton')
        self._render()

    def _start_drag(self, event):
        if not self._running:
            self._drag = (event.x, event.y)
            event.widget.focus_set()

    def _pan(self, event):
        if self._drag is None or self._running:
            return
        self.viewport.pan(event.x - self._drag[0], event.y - self._drag[1])
        self._drag = (event.x, event.y)
        self._render()

    def _wheel(self, event, direction=None):
        if self._running:
            return 'break'
        direction = direction if direction is not None else (1 if event.delta > 0 else -1)
        levels = (1, 2, 4, 8)
        index = max(0, min(3, levels.index(self.viewport.zoom) + direction))
        self.set_zoom(levels[index])
        return 'break'

    def _render(self):
        if not hasattr(self, 'canvases'):
            return
        self.viewport.view_size = tuple(min(canvas.winfo_width() - 2 if axis == 0 else canvas.winfo_height() - 2
                                            for canvas in self.canvases) for axis in (0, 1))
        self.viewport.clamp()
        for index, (canvas, choice) in enumerate(zip(self.canvases, (self.left_choice.get(), self.right_choice.get()))):
            self.pane_titles[index].configure(text=DISPLAY_NAMES.get(choice, choice))
            canvas.delete('all')
            size = (max(1, canvas.winfo_width() - 2), max(1, canvas.winfo_height() - 2))
            source, scale = None, 1
            if choice == 'LR' and self.lr_image is not None:
                source, scale = self.lr_image.rgb, 4
            elif choice == 'HR' and self.hr_image is not None:
                source = self.hr_image.rgb
            elif choice in self.results:
                source = self.results[choice].rgb
            if source is None:
                text = 'Chọn ảnh LR để bắt đầu' if self.lr_image is None else (
                    'Chưa chọn ảnh HR' if choice == 'HR' else 'Chưa có kết quả')
                canvas.create_text(size[0] / 2, size[1] / 2, text=text,
                                   fill=COLORS['muted'], font=(self.font_family, -15))
                self._photos[index] = None
                continue
            cached = self._display_images.get(choice)
            if cached is None or cached[0] is not source:
                cached = (source, Image.fromarray(source))
                self._display_images[choice] = cached
            rendered = self.viewport.render(cached[1], size, virtual_scale=scale)
            if rendered is not None:
                self._photos[index] = ImageTk.PhotoImage(rendered, master=self.root)
                canvas.create_image(1, 1, anchor='nw', image=self._photos[index])

    def close(self):
        if self._running:
            self._closing = True
            self.stop()
            return
        self._destroy()

    def _destroy(self):
        self.controller.detach()
        self._photos[:] = [None, None]
        self.root.destroy()
        gc.collect()
