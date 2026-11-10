"""The approved light, square ttk palette for the M10 desktop demo."""
import ctypes
import tkinter.font as tkfont
from tkinter import ttk


COLORS = {
    'bg': '#F3F4F6', 'surface': '#FBFBFC', 'text': '#1F2328',
    'muted': '#57606A', 'border': '#D0D7DE', 'accent': '#1F5FBF',
    'accent_hover': '#184C99', 'accent_soft': '#E8F0FB',
    'warn_bg': '#FFF6DD', 'warn_text': '#7A5200',
    'danger': '#B42318', 'canvas': '#E2E5E9',
}


def enable_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


def configure_styles(root):
    c = COLORS
    family = 'Segoe UI' if 'Segoe UI' in tkfont.families(root) else tkfont.nametofont('TkDefaultFont').actual('family')
    style = ttk.Style(root)
    style.theme_use('clam')
    # Normalize 14/10/9 pt to the approved 96-DPI physical layout, even at high Tk scaling.
    style.configure('.', background=c['bg'], foreground=c['text'], font=(family, -13),
                    bordercolor=c['border'], lightcolor=c['bg'], darkcolor=c['bg'])
    style.configure('Side.TFrame', background=c['surface'])
    style.configure('Side.TLabel', background=c['surface'])
    style.configure('Title.TLabel', background=c['surface'], font=(family, -19, 'bold'))
    style.configure('H.TLabel', background=c['surface'], font=(family, -13, 'bold'))
    style.configure('Muted.TLabel', background=c['surface'], foreground=c['muted'], font=(family, -12))
    style.configure('MainMuted.TLabel', foreground=c['muted'], font=(family, -12))
    style.configure('Danger.TLabel', background=c['surface'], foreground=c['danger'], font=(family, -12))
    style.configure('Warn.TLabel', background=c['warn_bg'], foreground=c['warn_text'],
                    font=(family, -12), padding=(10, 6))
    style.configure('Side.TCheckbutton', background=c['surface'], font=(family, -13),
                    indicatorbackground=c['surface'], indicatorforeground=c['accent'],
                    indicatormargin=(0, 0, 8, 0))
    style.map('Side.TCheckbutton', background=[('active', c['surface'])])
    style.configure('Primary.TButton', background=c['accent'], foreground=c['surface'],
                    font=(family, -13, 'bold'), padding=(14, 8), borderwidth=0)
    style.map('Primary.TButton', background=[('disabled', c['border']), ('active', c['accent_hover'])],
              foreground=[('disabled', c['muted'])])
    style.configure('Secondary.TButton', background=c['surface'], foreground=c['text'],
                    padding=(12, 7), bordercolor=c['border'])
    style.map('Secondary.TButton', background=[('active', c['accent_soft'])])
    style.configure('Seg.TButton', background=c['surface'], padding=(6, 3),
                    font=(family, -12), bordercolor=c['border'])
    style.configure('SegOn.TButton', background=c['accent_soft'], foreground=c['accent'],
                    padding=(6, 3), font=(family, -12, 'bold'), bordercolor=c['accent'])
    style.configure('Pane.TLabel', font=(family, -12, 'bold'))
    style.configure('Treeview', background=c['surface'], fieldbackground=c['surface'],
                    rowheight=28, font=(family, -13), bordercolor=c['border'])
    style.configure('Treeview.Heading', background=c['bg'], font=(family, -12, 'bold'), relief='flat')
    style.configure('TProgressbar', background=c['accent'], troughcolor=c['border'], thickness=6)
    root.configure(bg=c['bg'])
    return family
