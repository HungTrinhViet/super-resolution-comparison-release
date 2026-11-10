# Milestone 1 — Bicubic ×4, kết quả và giới hạn

Mọi tham chiếu `logs/...` trong tài liệu này là **log cục bộ, không công bố
trong repository**. Tài liệu ghi lại protocol, số đo và lịch sử kiểm chứng;
script đối chiếu độc lập được cung cấp để tái lập các kiểm tra trên run có sẵn.

## Quyết định Bicubic baseline

Hai mốc thời gian được giữ nguyên và phân biệt như sau:

- **2026-11-08 16:27:03 UTC** là mốc được ghi là “chốt” trong tài liệu ban đầu,
  còn xuất hiện trong snapshot ở `logs/m1_step1_decision.log`.
  The available logs do not independently establish which event 16:27:03 UTC
  represents; it cannot be confirmed as the time the document was written.
- **2026-11-08 16:27:56.195160 UTC** là lúc ghi receipt
  `BASELINE_DECISION_RECORDED_UTC`, SHA-256 và snapshot phần quyết định vào
  `logs/m1_step1_decision.log`. Receipt xác nhận tài liệu đã chứa quyết định
  trước khi chạy metrics.

Cả hai mốc đều trước **METRICS_RUN_START 2026-11-08 16:27:57.940333 UTC**
trong `logs/m1_step1_run.log`; ý nghĩa độc lập của mốc đầu vẫn chưa xác định.

Backend baseline: **NumPy separable cubic a=-0.5**, half-pixel center mapping,
symmetric extension bao gồm boundary sample, float64 hai pass, clip [0,255]
và round-half-up một lần cuối về uint8. Không cache trọng số ngoài inference.

Lý do: đặc tả tường minh về kernel/alignment/biên và lượng tử hóa; gần với
cách mô tả resize bicubic mà benchmark SR thường tham chiếu. Đã có test tính
chất hằng số, symmetry, kích thước ×4, giá trị kernel, tọa độ half-pixel trên
ramp và giá trị extrapolation ở biên. Các kiểm tra là kiểm tra toán học/tính
chất, **không phải MATLAB oracle độc lập**. Không tuyên bố “tương thích MATLAB”.

Pillow 11.2.1 là phương án đối chiếu hợp lệ: cùng a=-0.5, có antialias khi
giảm và center mapping, nhưng support bị cắt/chuẩn hóa ở biên cùng đường
uint8/fixed-point khác NumPy. Không gọi Pillow mặc nhiên sai. Không chọn
baseline vì downsampling tái tạo LR gần nhất, cũng không chọn theo PSNR/SSIM
upsampling cao hơn. Trước thời điểm này **chưa tính PSNR/SSIM HR-vs-SR Set5**;
chỉ có kiểm tra LR degradation và synthetic unit tests.

Tham khảo: [MathWorks imresize](https://www.mathworks.com/help/matlab/ref/imresize.html),
[BasicSR cubic/coordinate/symmetric implementation](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/matlab_functions.py),
[Pillow 11.2.1 Resample.c](https://github.com/python-pillow/Pillow/blob/11.2.1/src/libImaging/Resample.c).
Source khảo sát đã lưu trong `logs/m1_step0_*` (log cục bộ, không công bố
trong repository); protocol đầy đủ được đặc tả trong tài liệu này.
Downsampling chỉ là **đối chiếu với LR SelfExSR**, không quyết định upsampling.

## Kiểm tra trước thực nghiệm

- Python 3.12.10; cài đúng 13 pin trong `.venv`; pip check PASS.
- Pytest: **22 passed**, log `logs/m1_step1_pytest_green.log`.
- Download: đúng 10 RGB PNG, không chuyển mode; Git blob SHA-1 so nguồn PASS;
  SHA-256 tự tính lưu `docs/SET5_MANIFEST.sha256`.
- Dimensions: **PASS 5/5**; không crop/resize sửa cặp benchmark.
- Content: **PASS 5/5**; raw matched LR PSNR 55.0811–58.0171 dB,
  MAE 0.10238–0.14730 (thang 0–255), ngưỡng sàng lọc 35 dB/3.
- Negative controls: **20 ordered swaps, 0/20 vượt ngưỡng**;
  PSNR 6.69679–10.37807 dB, MAE 63.03125–99.28125.
  Do các ảnh khác kích thước, đo nội dung trên views 32×32 dùng cùng Pillow
  BICUBIC cho cặp đúng và sai; có lưu kiểm tra shape gốc. Đây là screening
  bổ sung có regrid **được ghi rõ**, không sửa ảnh benchmark, không dùng cho
  metrics SR chính. Raw matched values vẫn so trên LR grid gốc.
- LR generation method: **UNVERIFIED**. README chỉ xác nhận bicubic kernel
  downsampling; thiếu tool/version/settings provenance. NumPy và Pillow đều
  không exact pixel-match LR ở cả 5 ảnh, nên không đủ điều kiện PASS(c).
  Xem toàn bộ số matched/control/degradation ở `docs/SET5_VERIFICATION.json`.

## Protocol thực nghiệm đã khóa

SR float64 → clip và round-half-up RGB uint8 → chuyển BT.601 Y float
`16+(65.481R+128.553G+24.966B)/255` → crop 4 pixel mỗi cạnh HR và SR → metrics.
Strict equal shape trước crop, data_range 255, SSIM Gaussian 11×11 sigma 1.5,
K1=.01/K2=.03 population covariance. RGB là bổ sung. Mean Set5 = mean
per-image. Tất cả ảnh tải thực tế là RGB nên không có ngoại lệ grayscale.

CPU timing: perf_counter_ns; warmup 3/repeats 30; chỉ upsample và lượng tử SR;
median/IQR, không I/O/tạo LR/metrics. Environment và các sample lưu trong run.

## Kết quả Set5 thật

Run ID: `set5_numpy_x4_20261108_1627`; bắt đầu UTC **2026-11-08T16:27:57.940333+00:00**.

Nguồn số: `results/set5_numpy_x4_20261108_1627/metrics.json` và `metrics.csv`.
Dữ liệu là 5 cặp PNG thật từ SelfExSR; không phải synthetic fixtures.

| Ảnh | PSNR Y (dB) | SSIM Y | PSNR RGB (dB) | SSIM RGB | Median ms | IQR ms |
|---|---:|---:|---:|---:|---:|---:|
| img_001_SRF_4 | 31.78644620 | 0.85765643 | 30.36962896 | 0.82878895 | 66.83700000 | 40.52487500 |
| img_002_SRF_4 | 30.18703920 | 0.87380864 | 28.21878366 | 0.84875447 | 15.68900000 | 0.95692500 |
| img_003_SRF_4 | 22.10101285 | 0.73748384 | 20.86513731 | 0.70122683 | 12.28355000 | 3.04385000 |
| img_004_SRF_4 | 31.61503165 | 0.75466398 | 28.89386920 | 0.67347603 | 18.04720000 | 6.94730000 |
| img_005_SRF_4 | 26.46922783 | 0.83269942 | 25.13235539 | 0.81459887 | 15.53820000 | 2.73422500 |
| **Mean per-image** | 28.43175155 | 0.81126246 | 26.69595490 | 0.77336903 | — | — |

MSE bình quân per-image: Y=139.533223492786; RGB=194.771784366072.

CPU: **11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz**, logical CPUs=8.
OS: `Windows-10-10.0.19045-SP0`; Python `3.12.10`.
Đúng 13 pin theo `requirements.txt`; thread env OMP/MKL/OPENBLAS/NUMEXPR/VECLIB đều unset.
Tất cả 30 sample/ảnh và warm-up 3 lưu trong metrics.json; environment.json chứa phiên bản đầy đủ.
IQR lớn ở img_001 được giữ nguyên, không loại mẫu hay chạy lại để làm đẹp số.
Tải nền, turbo/power plan chưa được kiểm soát; đây là phép đo trên máy hiện tại.

Bằng chứng thứ tự: `logs/m1_step1_decision.log` ghi backend lúc
**2026-11-08T16:27:56.195160+00:00**, trước METRICS_RUN_START
**2026-11-08T16:27:57.940333+00:00** trong `logs/m1_step1_run.log`.
Log decision lưu SHA256 và snapshot phần quyết định trước khi bổ sung bảng số vào tài liệu này.

## Reference và hạn chế

[VDSR, Kim et al., CVPR 2016 Table 3](https://arxiv.org/html/1511.04587v2) báo Bicubic Set5 ×4 **28.42/0.8104**.
Đó là số tác giả công bố, không phải kết quả nhóm, không là tiêu chí PASS/FAIL.
Y float, lượng tử hóa, padding hoặc degradation khác có thể làm số khác nhau.
Không có MATLAB oracle độc lập; LR generator tool/settings chưa xác định; license ảnh chưa xác định.
Ảnh chỉ dùng học thuật nội bộ, không commit/phân phối. Chưa kiểm tra Máy B/CUDA hay model pretrained.

## Kiểm thử và review sau chạy

Final pytest: **25 passed**, log `logs/m1_step1_pytest_final_code.log`.
Review phát hiện invalid repeats/warmup có thể tạo output directory trước báo lỗi;
đã thêm regression tests thấy FAIL rồi đưa validation lên trước tạo output; các tests đã PASS.
Không đổi kernel/inference sau phép chạy Set5; chỉ thêm preflight validation và chỉnh trình bày/logging.
Các test là synthetic software checks, tách riêng hoàn toàn khỏi bảng Set5 thật ở trên.

## Kiểm chứng độc lập

Ngày 2026-11-08, một bản cài đặt độc lập theo đặc tả `imresize` đã tạo lại
5 ảnh SR trùng từng pixel với run `set5_numpy_x4_20261108_1627`. Bản đối chiếu
tự tính Y BT.601, crop 4 pixel và PSNR; SSIM kiểu Wang dùng Gaussian 11×11,
sigma 1.5 và `scipy.signal.convolve2d` mode `valid`. PSNR Y và SSIM Y khớp
`metrics.csv` đến 8 chữ số thập phân; trung bình **28.43175155 / 0.81126246**.
Đây là đối chiếu độc lập theo đặc tả, không phải MATLAB oracle.

Ngày chạy script công bố: **2026-11-09 (Asia/Bangkok)**; receipt UTC
**2026-11-08T18:12:55.489038+00:00**. Lệnh:

```powershell
.\.venv\Scripts\python.exe scripts/crosscheck_bicubic.py --run-dir results/set5_numpy_x4_20261108_1627
```

Kết quả: **PASS 5/5** ảnh, trùng từng pixel; PSNR Y khớp tuyệt đối trong
precision được in, max abs delta PSNR bằng 0; max abs delta SSIM **1.132e-14**.
Dung sai kiểm tra là **1e-6 tuyệt đối, relative tolerance 0**, không làm tròn
score trước khi so. Trung bình **28.43175155 / 0.81126246**. Script không import
package ứng dụng và không ghi vào dataset/run. Log `logs/docsA_crosscheck.log`
là log cục bộ, không công bố trong repository. Pytest sau bổ sung test hàm
thuần của checker: **26 passed**, giữ riêng lịch sử **25 passed** ở trên.

Known execution notes (English):

- Initial pytest collection failed as expected because the package had not been implemented; red logs are retained.
- Preflight regression initially failed for invalid warmup/repeats; the validation order is fixed and all tests pass.
- Early pytest runs used its default Windows temporary directory outside the project. Later runs are confined to `logs/pytest_tmp` via pytest.ini. No outside-project cleanup was attempted.
- The pip install wrapper reported exit 1 from PowerShell stderr handling of a pip upgrade notice. Installation completed; all 13 exact versions import and `pip check` passes. Pip was not upgraded.
- Initial INFO logging went to stderr and PowerShell wrapped it as NativeCommandError despite native exit 0. Scripts now log INFO to stdout; the receipts retain the actual native exit code.

## Lệnh và log kiểm chứng

| Lệnh | Kết quả | Log cục bộ (không công bố trong repository) |
|---|---|---|
| `py -3.12 -m venv .venv` | PASS | logs/m1_step1_venv.log |
| `.venv/Scripts/python.exe -m pip install --index-url https://pypi.org/simple --only-binary=:all: --no-cache-dir -r requirements.txt` | Installed; wrapper notice explained above | logs/m1_step1_install.log |
| `.venv/Scripts/python.exe -m pip check` | PASS | logs/m1_step1_pip_check.log |
| Version/import check (13 pins) | PASS | logs/m1_step1_import.log |
| `.venv/Scripts/python.exe -m pytest -v` (before implementation) | Expected FAIL, missing package | logs/m1_step1_pytest_red.log |
| `.venv/Scripts/python.exe -m pytest tests/test_verification.py -v` (before implementation) | Expected FAIL, missing module | logs/m1_step1_verify_pytest_red.log |
| `.venv/Scripts/python.exe -m pytest tests/test_core.py -k invalid_experiment -v` | Expected regression FAIL; fixed | logs/m1_step1_preflight_red.log |
| `.venv/Scripts/python.exe -m pytest -v` (final code) | PASS 25, 1.93 s | logs/m1_step1_pytest_final_code.log |
| `.venv/Scripts/python.exe scripts/download_set5.py` | PASS 10 verified files | logs/m1_step1_download.log |
| `.venv/Scripts/python.exe scripts/download_set5.py` (existing files) | PASS, no image redownload/overwrite | logs/m1_step1_download_existing.log |
| `.venv/Scripts/python.exe scripts/verify_set5.py` | Dimensions/content PASS, method UNVERIFIED | logs/m1_step1_verify.log |
| `.venv/Scripts/python.exe scripts/run_bicubic.py --data-dir data/set5/image_SRF_4 --output-dir results/set5_numpy_x4_20261108_1627` | PASS 5 real pairs | logs/m1_step1_run.log |
| `.venv/Scripts/python.exe scripts/run_bicubic.py --help` | PASS | logs/m1_step1_cli_help.log |
| `.venv/Scripts/python.exe logs/m1_step1_artifact_review.py` | PASS 14 file AST/style, 10 hashes, 5 saved-image metrics, 30 timing samples/image | logs/m1_step1_artifact_review_final.log |
| `git status --short --branch --untracked-files=all` | PASS tại thời điểm kiểm tra | logs/m1_step1_git_status.log |

Artifact review ban đầu phát hiện một dòng exception dài 82 ký tự; đã ngắt
dòng và chạy lại PASS. Không đổi thuật toán hoặc số liệu. Source/manual review
và kiểm chứng artifact được bổ sung bởi đối chiếu độc lập mô tả ở mục
“Kiểm chứng độc lập”. Các helper trong `logs/` là công cụ kiểm tra cục bộ,
không công bố trong repository; script crosscheck công bố ở `scripts/`.

Phạm vi kết quả hiện tại là Bicubic. SRCNN chưa được triển khai; nguồn
checkpoint, license và preprocessing cho các mô hình học sâu chưa được kiểm thử.
