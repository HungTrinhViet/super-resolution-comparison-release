
from pathlib import Path
from tkinter import Tk, filedialog
from PIL import Image, ImageOps

# Chọn ảnh gốc từ máy tính
root = Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Chọn ảnh gốc",
    filetypes=[
        ("Ảnh", "*.jpg *.jpeg *.png")
    ]
)

root.destroy()

if not file_path:
    raise SystemExit("Bạn chưa chọn ảnh!")

# Thư mục chứa ảnh đã xử lý
output = Path("demo_local")
output.mkdir(exist_ok=True)

# Đọc ảnh gốc, cắt giữa thành ảnh vuông và giảm kích thước
with Image.open(file_path) as img:
    img = img.convert("RGB")

    hr = ImageOps.fit(
        img,
        (512, 512),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5)
    )

# Tạo LR bằng cách giảm 4 lần mỗi chiều
lr = hr.resize(
    (128, 128),
    resample=Image.Resampling.BICUBIC
)

# Lưu ảnh PNG
hr.save(output / "HR_512.png")
lr.save(output / "LR_128.png")

print("Đã tạo ảnh thành công!")
print("HR:", (output / "HR_512.png").resolve())
print("LR:", (output / "LR_128.png").resolve())
