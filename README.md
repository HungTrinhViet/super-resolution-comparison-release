# So sánh Super Resolution ×4 với GUI Tkinter

Đồ án Xử lý ảnh số so sánh năm thuật toán tăng độ phân giải một ảnh (Single Image Super Resolution, SISR) ở hệ số ×4. Phiên bản này có mã nguồn, GUI Tkinter, kiểm thử, tài liệu khoa học và bảng kết quả CPU trên năm cặp Set5. Các mô hình dùng trọng số pretrained; không huấn luyện từ đầu.

## Năm thuật toán

| Thuật toán | Đặc điểm của phiên bản sử dụng |
|---|---|
| Bicubic | Nội suy NumPy, kernel a = -0.5; đối chiếu LR SelfExSR, không tuyên bố có MATLAB oracle độc lập. |
| SRCNN | 9-5-5 ImageNet ×4, xử lý kênh Y sau phóng to, trọng số tác giả. |
| FSRCNN | (56,12,4) ×4, xử lý trong không gian LR, deconvolution cuối mạng; checkpoint bản đăng lại. |
| EDSR-baseline | 16 residual block, 64 kênh, không batch normalization, PixelShuffle; không phải EDSR full. |
| Original ESRGAN | 23 RRDB, residual scaling 0.2 hai cấp; checkpoint GAN BasicSR chuyển đổi, không phải Real-ESRGAN. |

Các quy trình đổi màu, lượng tử, padding và metric được mô tả chi tiết trong tài liệu kết quả liên kết bên dưới. GUI gọi lại đúng thư viện, không triển khai lại thuật toán.

## Cấu trúc repository

```text
src/sr_compare/       Thuật toán, metric, I/O, inference registry và GUI
scripts/             CLI tải/kiểm tra trọng số, thực nghiệm và đối chiếu
tests/               Unit test, regression và reference NumPy độc lập
docs/                Kết quả khoa học, provenance và hướng dẫn GUI
docs/comparison/     Bảng CSV/Markdown, biểu đồ SVG và crop manifest
docs/report/         Lưu đồ và ánh xạ claim để biên tập báo cáo
requirements*.txt    Các phiên bản dependency đã khóa
tao_anh_demo.py       Tạo cặp ảnh dùng thử cục bộ
```

Dataset, checkpoint, môi trường Python, kết quả ảnh, log, tài liệu nội bộ và ảnh demo người dùng không nằm trong repository công khai. Hai ảnh chụp GUI trong `docs/gui/` chỉ dùng ảnh tổng hợp.

## Hệ thống và môi trường

Môi trường thực nghiệm đã ghi nhận: Windows 10 build 19045, Intel Core i7-1185G7, RAM 16,889,188,352 byte, Python 3.12.10 và PyTorch 2.10.0+cpu. GUI dùng Tkinter, CPU FP32, batch 1, 4 intra-op / 1 inter-op thread. Không có CUDA trong cấu hình này; ảnh lớn có thể không đủ RAM. Đây là cấu hình đã kiểm chứng, không phải cam kết tương thích mọi hệ điều hành.

Từ thư mục gốc repository, dùng PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple --only-binary=:all: -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --require-hashes --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements-m2-cpu.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` khóa 13 package nền; `requirements-m2-cpu.txt` bổ sung 9 package với hash, giúp kiểm tra đúng wheel tải về. Không cần OpenCV, TensorFlow hoặc cài thêm gói GUI. Bản Python Windows phải có Tkinter. Không cài package vào Python hệ thống.

## Tải và kiểm tra checkpoint

```powershell
.\.venv\Scripts\python.exe scripts/download_models.py --item srcnn
.\.venv\Scripts\python.exe scripts/inspect_checkpoint.py data/quarantine/srcnn/SRCNN_v1.zip --extract-x4
.\.venv\Scripts\python.exe scripts/download_models.py --item edsr
.\.venv\Scripts\python.exe scripts/download_models.py --item esrgan
```

Mỗi lệnh tải một item, không ghi đè; kiểm kích thước, tên, HTML, chữ ký ZIP hoặc legacy torch đầy đủ và SHA-256. File `.part` còn sót phải được người dùng kiểm tra trước khi xóa thủ công. `.pth` chỉ đọc bằng `torch.load(weights_only=True, map_location="cpu")`; `.mat` qua `scipy.io.loadmat`. Xem `--help` của `scripts/inspect_checkpoint.py` để kiểm tra file riêng.

FSRCNN không có item tải tự động: lấy đúng [x4.mat, yaBorn commit c6b50c5](https://raw.githubusercontent.com/yaBorn/Super-resolution/c6b50c57f33b044eed1b4d5c20344848842d65e9/mat_FSRCNN/Model/FSRCNN/FSRCNN/x4.mat), lưu thành `data/quarantine/fsrcnn/yaBorn_c6b50c5_FSRCNN_x4.mat`. SHA-256 phải là `af005f2ba6e90ef45f1807045c3f78132e15b57d9056e4691c79e34863dbdd4f`.

So sánh với [manifest trọng số](docs/MODELS_MANIFEST.sha256) và [provenance](docs/MODELS_PROVENANCE.md). Hash tự tính bảo vệ tính toàn vẹn sau khi tải; không nguồn nào công bố SHA-256 đầy đủ. Tám ký tự trong tên hai checkpoint BasicSR chỉ là kiểm tra một phần. License riêng của trọng số chưa xác minh; SRCNN Readme giới hạn học thuật, không thương mại và yêu cầu trích dẫn. License code không tự động cấp quyền cho trọng số.

## Chạy GUI

```powershell
.\.venv\Scripts\python.exe scripts/run_gui.py --check-startup
.\.venv\Scripts\python.exe scripts/run_gui.py
```

Chọn LR, có thể chọn HR, chọn từ một đến năm thuật toán rồi chạy. Hai khung ảnh đồng bộ zoom/pan; kết quả lưu PNG + JSON vào thư mục mới. Một thuật toán lỗi không ngăn các thuật toán sau. Dừng có hiệu lực sau bước hiện tại, không ngắt forward giữa chừng. Xem [GUI_GUIDE](docs/GUI_GUIDE.md) cho trạng thái, phím tắt, trường metadata và ảnh chụp tổng hợp.

**PSNR/SSIM chỉ có ý nghĩa khi HR là ảnh gốc đúng của LR và được căn chỉnh khớp. Chương trình chỉ kiểm được kích thước.** HR phải đúng 4 × LR theo từng chiều; không tự crop/resize cặp tham chiếu. Nhãn metric: “Theo giao thức M1; không phải kết quả benchmark”. Thời gian demo trong GUI có thể gồm nạp checkpoint lần đầu, không phải benchmark đã nghiệm thu. CLI mới là công cụ benchmark chính thức.

## Tạo ảnh LR/HR dùng thử

```powershell
.\.venv\Scripts\python.exe tao_anh_demo.py
```

Người dùng chọn JPG/JPEG/PNG trong hộp thoại. Script chuyển ảnh về RGB, cắt vùng vuông giữa và resize Lanczos thành HR 512 × 512; sau đó dùng Pillow Bicubic tạo LR 128 × 128. Hai file `demo_local/HR_512.png` và `demo_local/LR_128.png` được lưu tương đối với thư mục chạy. Script **có thể ghi đè hai file demo khi chạy lại**, nhưng không sửa ảnh gốc. Hủy chọn ảnh thì thoát.

Đây chỉ là công cụ demo: không tái tạo quy trình tạo LR của SelfExSR và không thay thế benchmark Set5. Pillow Bicubic ở script này không được coi là đồng nhất với baseline NumPy. Kết quả một ảnh tổng hợp hoặc ảnh người dùng không quyết định thứ hạng trung bình trên Set5. Không commit ảnh cá nhân hoặc `demo_local/`.

## Kiểm thử và dữ liệu tùy chọn

Sau khi chuẩn bị checkpoint, chạy:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -rs
```

Các test mạng dùng checkpoint và archive SRCNN cục bộ; clone sạch chưa có các file đó nên không thể tuyên bố toàn bộ suite đạt. Regression Set5 mặc định SKIP với thông báo `NOT VERIFIED: Set5 missing`; `--require-set5` yêu cầu dữ liệu thật và sẽ FAIL khi thiếu. Để chạy phần không cần checkpoint mà không sửa test:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -rs --ignore=tests/test_srcnn.py --ignore=tests/test_fsrcnn.py --ignore=tests/test_edsr.py --ignore=tests/test_esrgan.py -k "not test_all_five_match_direct_library_calls_exactly and not test_cache_reuses_one_model_releases_before_next_load and not test_inference_value_errors_propagate"
```

Lệnh lọc bỏ bốn module có kiểm chứng dùng trọng số/Set14 archive và phép parity năm mô hình thật; hai test cache và truyền lỗi cũng cần checkpoint nên được loại khỏi tập không dữ liệu. Các phần bị loại này chưa được kiểm chứng trên clone thiếu dữ liệu, không phải test PASS. Các test Tkinter cần phiên desktop có Tk. Không tự cài dependency hoặc tải dataset trong test.

Nếu được phép dùng Set5, có thể tải bằng `scripts/download_set5.py`, rồi kiểm bằng `scripts/verify_set5.py` theo [provenance Set5](docs/SET5_PROVENANCE.md). Mã nguồn cung cấp CLI `scripts/run_bicubic.py`, `run_srcnn.py`, `run_fsrcnn.py`, `run_edsr.py`, `run_esrgan.py`; dùng `--help`. Các runner mạng kiểm biên bản khóa cục bộ, vốn không được đưa vào repository công khai. Không tạo giả biên bản hoặc coi việc chạy GUI là một benchmark mới. Không cần chạy lại benchmark để sử dụng GUI.

## Kết quả Set5 ×4 đã nghiệm thu kỹ thuật

Nguồn: [mean.csv](docs/comparison/tables/mean.csv), tính trung bình số đầy đủ của năm ảnh; bảng dưới chỉ làm tròn hiển thị (PSNR 4 chữ số, SSIM 6 chữ số).

| Thuật toán | PSNR Y (dB) | SSIM Y | PSNR RGB (dB) | SSIM RGB |
|---|---:|---:|---:|---:|
| Bicubic | 28.4318 | 0.811262 | 26.6960 | 0.773369 |
| SRCNN | 30.4806 | 0.862561 | 28.4808 | 0.823883 |
| FSRCNN | 30.7016 | 0.865449 | 28.6690 | 0.826462 |
| EDSR | 32.0979 | 0.893816 | 30.1840 | 0.863086 |
| ESRGAN | 30.4742 | 0.851801 | 28.5433 | 0.814520 |

EDSR-baseline đứng đầu PSNR/SSIM Y trung bình trên tập này. ESRGAN có kết quả kém Bicubic ở img_001 và img_004 trên cả hai metric Y; không bỏ qua các trường hợp bất lợi. Chất lượng cảm nhận khác độ chính xác tái tạo: texture do GAN sinh không được coi mặc nhiên là chi tiết đúng. Năm ảnh không đủ để tổng quát hóa sang mọi dữ liệu.

Protocol chính: lượng tử SR RGB uint8 theo từng pipeline đã khóa, đổi sang Y BT.601 float64 với `Y = 16 + (65.481R + 128.553G + 24.966B)/255`, giữ Y float, kiểm cùng kích thước rồi bỏ 4 pixel mỗi cạnh. MSE/PSNR dùng data_range 255; SSIM Gaussian 11 × 11, sigma 1.5, K1 0.01, K2 0.03, population covariance, data_range 255. RGB là metric bổ sung. Không resize ngầm để tính metric.

Thời gian CPU pipeline và chỉ mạng được tách riêng trong tài liệu; Bicubic dùng 3 warm-up + 30 lần đo, mạng dùng 3 + 10, báo median/IQR. Các phiên chạy khác nhau, cùng các bất thường timing đã ghi, hạn chế so sánh trực tiếp. Không dùng thời gian GUI để thay các bảng đó.

## Nguồn, quyền sử dụng và giới hạn khoa học

- Set5 HR/LR lấy sẵn từ SelfExSR commit `8f6dd8c1d20cb7e8792a7177b4f6fd677633f598`. Công cụ/phương pháp giảm mẫu cụ thể chưa xác định; không tuyên bố toàn bộ LR do nhóm tạo. License Set5 chưa xác định, chỉ dùng học thuật nội bộ. Không phân phối dataset, ảnh SR hoặc contact sheet Set5 trong repo công khai.
- SRCNN dùng bản 9-5-5 ImageNet ×4 của tác giả; FSRCNN dùng bản đăng lại FSRCNN_test, hai bản sao trùng SHA-256, nhưng nguồn trực tiếp CUHK và cách chọn checkpoint chưa xác minh.
- EDSR/ESRGAN dùng checkpoint BasicSR; chuỗi chuyển đổi từ file tác giả gốc chưa được xác minh độc lập. Kiến trúc Original ESRGAN không chứng minh trọng số gốc đã xác minh. License trọng số và trùng dữ liệu huấn luyện với Set5 vẫn UNVERIFIED.
- Không có MATLAB oracle độc lập; giữ cảnh báo Pearson, hạn chế crop, các bất thường thời gian và kết quả bất lợi trong tài liệu nguồn. Crop exploratory v2 không thay kết quả chính, 3/5 vùng trùng v1. Nhóm đã xem SR trước khi xây dựng quy tắc crop, không coi đây là đánh giá mù hay human study.

## Tài liệu kỹ thuật và nguồn tham khảo

- [Bicubic và protocol M1](docs/M1_RESULTS.md)
- [SRCNN](docs/M3_SRCNN_RESULTS.md), [FSRCNN](docs/M4_FSRCNN_RESULTS.md), [EDSR](docs/M5_EDSR_RESULTS.md), [Original ESRGAN](docs/M6_ESRGAN_RESULTS.md)
- [So sánh chính thức](docs/M7_COMPARISON_RESULTS.md), [exploratory v2](docs/M7_EXPLORATORY_V2.md)
- [Nguồn và license checkpoint](docs/MODELS_PROVENANCE.md), [Set5](docs/SET5_PROVENANCE.md)
- [Bảng từng ảnh](docs/comparison/tables/per_image.csv), [lưu đồ triển khai L1](docs/report/flowcharts/L1.svg), [lưu đồ khái niệm C1](docs/report/flowcharts_concept/C1.svg)
- [Script crosscheck độc lập](scripts/crosscheck_bicubic.py), [checker báo cáo](scripts/check_report_consistency.py). Checker chỉ kiểm claim đã khai báo, không chứng minh toàn bộ tính đúng ngữ nghĩa của báo cáo.

## Quy trình phát triển và khai báo sử dụng AI

Mã nguồn được viết với sự hỗ trợ của công cụ AI: OpenAI Codex hỗ trợ viết code; Claude (Anthropic) điều phối, review và thực hiện kiểm chứng độc lập. Các quyết định kỹ thuật do nhóm sinh viên đưa ra với sự tư vấn của một trợ lý AI hướng dẫn kỹ thuật. Nhóm chịu trách nhiệm kiểm tra, hiểu và giải thích code, số liệu và giới hạn.

Quy trình này **không phải nghiệm thu chính thức của giảng viên môn học hoặc nhà trường**. Bằng chứng kiểm chứng gồm pytest, kiểm tra dữ liệu và checksum, cùng script đối chiếu độc lập. Công cụ AI không thay thế trách nhiệm khoa học của nhóm. Lịch sử Git mới có một initial commit trình bày phiên bản hoàn chỉnh; điều đó không phủ nhận quá trình phát triển và kiểm chứng trước đó.
