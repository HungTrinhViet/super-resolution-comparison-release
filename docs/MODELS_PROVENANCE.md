# Nguồn checkpoint và kiểm tra môi trường M2

Ngày ghi nhận: 2026-11-09 (Asia/Bangkok). Đây là kiểm tra nguồn và dữ liệu
trọng số, chưa phải kết quả suy luận. Không mô hình nào trong tài liệu này
đã được chạy trên Set5. Không có PSNR, SSIM hoặc timing mô hình.

## Trạng thái các cổng

| Cổng | Trạng thái | Bằng chứng |
|---|---|---|
| G0: regression M1 trước cài torch | PASS | Hai chế độ pytest: 33 passed; crosscheck: PASS 5/5 |
| G1: môi trường torch CPU | PASS | Đúng 9 package mới; 13 pin M1 không đổi; pip check đạt; regression 33 passed và crosscheck PASS 5/5 |
| G2: archive SRCNN | PASS | Đủ byte, magic, CRC, đường dẫn an toàn; đọc được hai MAT ×4 |
| G3: khảo sát FSRCNN | PASS về khảo sát; checkpoint UNVERIFIED | Bảng nguồn và rủi ro dưới đây; chưa chọn hoặc tải trọng số |
| G4: EDSR-baseline | PASS sau sửa validator định dạng | 6,084,475 byte; legacy torch; params; 1,517,571 scalar |
| G5: Original ESRGAN | PASS | 66,922,040 byte; legacy torch; params; 16,697,987 scalar |

Log chi tiết là log cục bộ, không công bố trong repository. Mọi checksum
trong [MODELS_MANIFEST.sha256](MODELS_MANIFEST.sha256) là **SHA-256 tự tính**
sau khi tải hoặc trích file; không phải checksum độc lập do tác giả công bố.
Chúng xác định bytes đã kiểm tra tại thời điểm này, không chứng minh lịch sử
huấn luyện hay giấy phép của checkpoint.

## Môi trường CPU và bảo vệ M1

Python 3.12; torch `2.10.0+cpu`; CUDA available = False; MKLDNN available =
True. Số thread thực tế mặc định là 4 intra-op và 4 inter-op; chưa đổi cấu
hình thread. Chỉ chạy một phép `conv2d` trên tensor tổng hợp để kiểm backend,
không khởi tạo hay chạy kiến trúc mô hình SR.

`requirements-m2-cpu.txt` khóa phiên bản và SHA-256 wheel cho 9 gói:
torch 2.10.0+cpu, filelock 3.18.0, typing_extensions 4.13.2, sympy 1.14.0,
mpmath 1.3.0, Jinja2 3.1.6, MarkupSafe 3.0.2, fsspec 2025.5.1,
setuptools 80.9.0. Freeze tăng từ 13 lên 22 dòng, không xóa hay đổi pin M1.
Dry-run và cài thật đều dùng `--require-hashes --only-binary=:all:`, index
`https://download.pytorch.org/whl/cpu` và extra index `https://pypi.org/simple`.
Cache đặt ở `logs/pip-cache` trong project.

Golden regression lấy trực tiếp từ PNG và metrics của run đã lưu
`set5_numpy_x4_20261108_1627`, source commit `100c90b`, không sinh golden
bằng hàm đang kiểm thử. File `tests/fixtures/m1_bicubic_expected.json` lưu
5 SHA-256 PNG cùng MSE/PSNR/SSIM Y và RGB; dung sai metric tuyệt đối là
1e-9. Chỉ chạy forward Bicubic, không đo lại timing. Thiếu Set5: pytest
thường skip với `NOT VERIFIED: Set5 missing`; `--require-set5` yêu cầu fail.
Hai nhánh này được kiểm bằng đường dẫn tạm, không xóa dữ liệu thật.

## SRCNN: archive của tác giả

Nguồn: Chao Dong và cộng sự, [trang SRCNN](https://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html),
[SRCNN_v1.zip](https://mmlab.ie.cuhk.edu.hk/projects/SRCNN/SRCNN_v1.zip).
Archive này không có revision bất biến hoặc checksum đầy đủ công bố được
xác minh. SHA-256 tự tính được ghi dưới đây và trong manifest.

| Thuộc tính | Kết quả thực tế |
|---|---|
| Đường dẫn | `data/quarantine/srcnn/SRCNN_v1.zip` |
| Kích thước | 7,558,018 byte, khớp kỳ vọng |
| Magic | `PK\x03\x04`, PASS |
| SHA-256 tự tính | `bfa68ca613c1326a59e0c34353205a254ab2b67e34df7f04e28eef567980af30` |
| UTC hoàn tất tải | `2026-11-08T18:43:25.890787+00:00` |
| Tổng bytes giải nén theo ZIP directory | 11,931,756 |
| Kiểm tra ZIP | CRC `testzip` PASS; không có đường dẫn tuyệt đối, `..`, ký tự ổ đĩa hoặc symlink |
| Giới hạn đặt trước khi kiểm | Tổng ≤ 200,000,000 byte; tỷ lệ nén mỗi entry ≤ 100 |
| Trích xuất | Chỉ Readme.txt và hai MAT ×4; không trích ảnh hoặc chạy MATLAB |
| MAT | Classic MAT đọc bằng `scipy.io.loadmat`; không phải v7.3/HDF5 |

Hai biến thể ×4 thực sự có trong archive:

| Biến MATLAB | 9-1-5 (91 images) | 9-5-5 (ImageNet) | dtype |
|---|---|---|---|
| `weights_conv1` | (81, 64) | (81, 64) | float64 |
| `biases_conv1` | (64, 1) | (64, 1) | float64 |
| `weights_conv2` | (64, 1, 32) | (64, 25, 32) | float64 |
| `biases_conv2` | (32, 1) | (32, 1) | float64 |
| `weights_conv3` | (32, 25) | (32, 25) | float64 |
| `biases_conv3` | (1, 1) | (1, 1) | float64 |
| Tổng số scalar weights + bias | 8,129 | 57,281 | |
| Byte của MAT | 38,489 | 269,001 | |

Ánh xạ đặc tả: conv1 1→64, kernel 9×9; conv2 64→32, kernel
1×1 hoặc 5×5; conv3 32→1, kernel 5×5. Có bias ở cả ba lớp, ReLU
sau hai lớp đầu. Khi chuyển đổi trong bước sau phải giữ thứ tự reshape
column-major của MATLAB và ánh xạ sang OIHW; chưa thực hiện chuyển đổi.

Nguồn preprocessing đọc trực tiếp dưới dạng text từ archive, không thực thi:

- `SRCNN/demo_SR.m` dòng 32–33 nêu model ImageNet 9-5-5 ×4; dòng 37–38
  dùng `rgb2ycbcr` và lấy Y; dòng 40 `modcrop`; dòng 41 `single/255`;
  dòng 44–45 `imresize(..., 'bicubic')` xuống rồi lên; dòng 51–53 chuyển
  `uint8(*255)` và shave theo scale.
- `SRCNN/SRCNN.m` dòng 17–18, 25–28, 34–39 dùng `imfilter` với
  `'same', 'replicate'`, ReLU và bias. Không thay padding replicate bằng
  zero padding. `imfilter` ở đây dùng correlation mặc định.
- `SRCNN/Readme.txt` xác định 9-1-5 (91 images) là biến thể ECCV 2014 và
  các biến thể ImageNet thuộc arXiv:1501.00092 (bài SRCNN mở rộng).

Biến thể chốt có điều kiện cho bước tích hợp: **9-5-5 ImageNet ×4**, theo bài SRCNN
mở rộng/TPAMI, vì archive tác giả có đúng biến thể này. Đây là quyết định
theo nguồn và kiến trúc trước mọi PSNR mô hình, chưa phải lựa chọn đã chạy.
9-1-5 (91 images) ×4 được giữ để đối chiếu nguồn; chưa chạy.
M3 phải kiểm chuyển trọng số, layout, padding replicate, tiền xử lý và hậu
xử lý trước khi đánh giá. Số tham số của bản 9-5-5 là 57,281.

**Điều khoản đã đọc:** Readme yêu cầu trích dẫn ECCV 2014 hoặc bài arXiv
khi dùng/thích nghi code; chỉ dùng cho mục đích học thuật, không dùng cho
hoạt động thương mại/công nghiệp. License riêng của trọng số: **không nêu
riêng**. Không suy diễn thành giấy phép phân phối lại checkpoint.
Chưa xác minh trùng/near-duplicate giữa tập huấn luyện và Set5.

## FSRCNN: khảo sát, chờ quyết định checkpoint

Không tải bất kỳ checkpoint FSRCNN nào. Các kích thước dưới đây đến từ
HEAD hoặc release metadata, không phải file đã tải. Không có SHA-256 tự tính.

| Nguồn | Bằng chứng nguồn/train/selection | Truy cập và byte | License và kết luận |
|---|---|---|---|
| [Tác giả FSRCNN](https://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html) | Trang công khai liệt kê MATLAB test code, Caffe training code và General-100; chưa đọc được nội dung các gói Drive | MATLAB Drive chuyển sang trang đăng nhập trong khảo sát HEAD trước; không đăng nhập, không truy cập tự động được; kích thước và biến thể trọng số UNVERIFIED | License code/trọng số và best-vs-last UNVERIFIED; ưu tiên làm rõ nguồn này trước khi chọn |
| [Lornatang 1.0](https://github.com/Lornatang/FSRCNN-PyTorch/tree/84b440d2570d730fb91c46b939bc5f7a7c8fb78d) | train.py mặc định BSDS300 train, Set5 eval; lưu PSNR tốt nhất vào đúng tên file release; README có ví dụ T91 train/Set5 eval | [fsrcnn_X4.pth](https://github.com/Lornatang/FSRCNN-PyTorch/releases/download/1.0/fsrcnn_X4.pth), HEAD 54,998 byte; release digest null, release notes trống | Header source Apache-2.0 đã xác minh; license trọng số riêng chưa xác định; rủi ro selection bias trên Set5; binary thực tế UNVERIFIED |
| [Lornatang 1.1](https://github.com/Lornatang/FSRCNN-PyTorch/releases/tag/1.1) | Release notes chỉ ghi “A fast super division algorithm”; không xác minh được lịch sử train của binary; khác kiến trúc 1.0 (64 thay 56 và khác vị trí PReLU), không thay thế trực tiếp | [fsrcnn_4x.pth](https://github.com/Lornatang/FSRCNN-PyTorch/releases/download/1.1/fsrcnn_4x.pth), HEAD 61,411 byte, digest null | Source LICENSE Apache-2.0 tại revision `6e9dcb1c9246077d5ef083a3f87799722966d93b`; trọng số riêng chưa xác định; UNVERIFIED |
| [yjn870](https://github.com/yjn870/FSRCNN-pytorch/tree/3eaab297f4b77b3c2877aac71b2a9497eaa839a4) | README 91-image train/Set5 eval; train.py lưu best theo PSNR eval thành best.pth; không có bằng chứng nối binary công bố với epoch cụ thể | [Dropbox ×4](https://www.dropbox.com/s/vsvumpopupdpmmu/fsrcnn_x4.pth?dl=1), HEAD 54,998 byte | Không tìm thấy LICENSE ở revision khảo sát; license code/trọng số chưa xác định; rủi ro selection bias; binary thực tế UNVERIFIED |

Ba liên kết chính xác trên trang tác giả (chỉ đọc metadata trang công khai):

- MATLAB test: `https://drive.google.com/open?id=0B7tU5Pj1dfCMVktYZUN2aV8xVTQ`.
- Caffe training: `https://drive.google.com/open?id=0B7tU5Pj1dfCMWjhhaE1HR3dqcGs`.
- General-100: `https://drive.google.com/open?id=0B7tU5Pj1dfCMVVdJelZqV0prWnM`.

Nguồn tái lập cho kết luận selection bias:

- [Lornatang train.py](https://github.com/Lornatang/FSRCNN-PyTorch/blob/84b440d2570d730fb91c46b939bc5f7a7c8fb78d/train.py#L31-L38)
  dòng 31–38: BSDS300 train và Set5 eval;
  [dòng 167–171](https://github.com/Lornatang/FSRCNN-PyTorch/blob/84b440d2570d730fb91c46b939bc5f7a7c8fb78d/train.py#L167-L171)
  lưu mỗi epoch và lưu best PSNR thành `fsrcnn_X{scale_factor}.pth`.
  [README dòng 159](https://github.com/Lornatang/FSRCNN-PyTorch/blob/84b440d2570d730fb91c46b939bc5f7a7c8fb78d/README.md#L159)
  lại minh họa T91. Không đủ bằng chứng xác định train dataset của binary.
- [yjn870 train.py dòng 106–112](https://github.com/yjn870/FSRCNN-pytorch/blob/3eaab297f4b77b3c2877aac71b2a9497eaa839a4/train.py#L106-L112)
  chọn best trên eval PSNR; [README dòng 22–37](https://github.com/yjn870/FSRCNN-pytorch/blob/3eaab297f4b77b3c2877aac71b2a9497eaa839a4/README.md#L22-L37)
  mô tả 91-image/Set5. Ví dụ train trong README là ×3, không phải log
  huấn luyện trực tiếp của binary ×4.
- [Release API](https://api.github.com/repos/Lornatang/FSRCNN-PyTorch/releases?per_page=15)
  không cung cấp log huấn luyện hoặc checksum digest cho các asset này.

**Chưa chốt checkpoint FSRCNN:** ưu tiên checkpoint tác giả hoặc nguồn có
quy trình chọn trọng số minh bạch. Lornatang 1.0 không được chọn làm
checkpoint đánh giá chính. Cần làm rõ điều khoản trọng số. yjn870 có rủi ro
tương tự và thiếu license rõ ràng. Không chọn checkpoint bằng PSNR Set5.
Không khẳng định binary là best chỉ từ code train; cũng không khẳng định
Set5 chưa từng được dùng để chọn checkpoint. Việc ảnh train trùng Set5 vẫn
chưa xác định, là câu hỏi khác với selection bias trên validation. FSRCNN
vẫn thuộc mục tiêu so sánh 5 thuật toán.

## BasicSR: nguồn EDSR và Original ESRGAN dự kiến

Revision cố định: `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a`.
[download_pretrained_models.py](https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/scripts/download_pretrained_models.py#L40-L41)
dòng 40–41 gắn tên ESRGAN với Drive ID;
[dòng 69](https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/scripts/download_pretrained_models.py#L69)
gắn tên EDSR với Drive ID.

| Item | URL cố định theo ID nguồn | Byte kỳ vọng | Số scalar kỳ vọng (chưa đo) |
|---|---|---|---|
| EDSR-baseline 16 khối/64 kênh ×4 | `https://drive.google.com/uc?export=download&id=1bCK6cFYU01uJudLgUUe-jgx-tZ3ikOWn` | 6,084,475 | 1,517,571, không gồm mean ngoài state_dict |
| Original ESRGAN ×4 | `https://drive.google.com/uc?export=download&id=1b3_bWZTjNO3iL2js1yWkJfjZykcQgvzT` | 66,922,040 | 16,697,987 |

Filename lần lượt là `EDSR_Mx4_f64b16_DIV2K_official-0c287733.pth`
và `ESRGAN_SRx4_DF2KOST_official-ff704c30.pth`. Drive object có thể thay
bytes; ID không phải một bảo đảm bất biến bằng mật mã.

**Checksum do nguồn công bố:** chưa xác minh được SHA-256 đầy đủ. Hai
suffix 8 hex là thông tin trong tên file do nguồn công bố. Chưa tìm thấy
căn cứ trong ba file source đã đọc để khẳng định BasicSR tạo suffix bằng
SHA-256: `download_util.py` dòng 97 truyền `hash_prefix=None`, còn nhánh
Drive không kiểm hash. Vì vậy đối chiếu 8 hex với đầu SHA-256 chỉ là kiểm
tra một phần theo giả thuyết quy ước tên kiểu torch.hub, không gọi là
checksum đầy đủ hoặc kiểm chứng độc lập nguồn. SHA-256 tự tính sau khi
tải sẽ được ghi riêng, chỉ khi tải và kiểm tra thành công.

[convert_models.py](https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/scripts/model_conversion/convert_models.py)
mô tả ánh xạ EDSR từ implementation tác giả và ESRGAN từ
`RRDB_ESRGAN_x4.pth`. Source này chỉ được đọc, không thực thi. EDSR dự
kiến có 16 residual block/64 kênh và hai bước PixelShuffle ×2; ESRGAN dự
kiến có 23 RRDB × 3 RDB × 5 conv. Schema, key/shape và tổng tham số
thực tế được kiểm riêng dưới đây. Kiến trúc RRDB
tương tự cũng xuất hiện trong Real-ESRGAN, nên key/shape đơn lẻ không
chứng minh nguồn huấn luyện Original ESRGAN; phải kết hợp mapping nguồn.

License code BasicSR: Apache-2.0 đã xác minh trong
[LICENSE.txt](https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/LICENSE.txt).
License riêng của các trọng số chuyển đổi: chưa xác định. Chưa kiểm tra
trùng/near-duplicate giữa DIV2K hoặc DF2K/OST và Set5.

## G4/G5: định dạng thực tế và kiểm tra lại

Lần G4 đầu **FAIL** với `ValueError: Artifact magic mismatch: expected
PK\x03\x04; stop.` Validator lúc đó giả định mọi checkpoint torch đều là
ZIP. Log cục bộ `logs/m2_g4_download.log` được giữ nguyên. Không đổi nguồn
hoặc bỏ kiểm tra; định dạng legacy được bổ sung sau khi chẩn đoán chữ ký.

Validator hiện chấp nhận chính xác ZIP `50 4B 03 04` hoặc legacy torch
protocol 2 `80 02 8A 0A 6C FC 9C 46 F9 20 6A A8 50 19` đủ 14 byte.
Hai file tải lại thực sự có chữ ký legacy này, tương ứng pickle protocol 2
và torch serialization MAGIC_NUMBER `0x1950a86a20f9469cfc6c`. Không dùng
unpickle tổng quát để kiểm chữ ký. Bất kỳ prefix khác hoặc HTML đều bị từ
chối. Không thay URL/ID. Tên Content-Disposition nếu có phải khớp tên đã
khóa; giữ kiểm kích thước, SHA-256, `.part` và không ghi đè.

| Kết quả thực đo | EDSR-baseline ×4 | Original ESRGAN ×4 |
|---|---|---|
| Byte | 6,084,475 | 66,922,040 |
| UTC tải xong | 2026-11-08T19:00:07.143044+00:00 | 2026-11-08T19:00:45.030255+00:00 |
| Định dạng | torch legacy protocol 2 | torch legacy protocol 2 |
| SHA-256 tự tính | `0c287733e70d1b8b8fc5885613ecbe451e5f3010bcae0307612ef5e4aa08dd5f` | `ff704c30ff560305e48ed1f4db895e525ab6bc81a46fafe80c0094a271c806d9` |
| 8 hex đầu khớp tên (kiểm một phần) | PASS `0c287733` | PASS `ff704c30` |
| Wrapper/top-level key | `params` | `params` |
| Tensor key | 74 | 702 |
| Dtype mọi tensor | torch.float32 | torch.float32 |
| Tổng scalar thực tế | 1,517,571 | 16,697,987 |
| Đối chiếu toàn bộ tên/shape/dtype với đặc tả | PASS | PASS |

Tất cả key và shape được in trong log cục bộ `logs/m2_g4r_inspect.log`
và `logs/m2_g5_inspect.log`. Không nạp bằng `weights_only=False`. Hai lần
đọc đều dùng `torch.load(weights_only=True, map_location='cpu')`, không
khởi tạo model. Kiểm bộ key chính xác, không chỉ kiểm tổng scalar.

Đặc tả đầy đủ bộ key EDSR (weight OIHW, bias có chiều output):

| Key weight (mỗi key có bias tương ứng) | Shape weight | Shape bias |
|---|---|---|
| `conv_first` | (64, 3, 3, 3) | (64,) |
| `body.i.convj`, i=0…15, j=1,2 | (64, 64, 3, 3) | (64,) |
| `conv_after_body` | (64, 64, 3, 3) | (64,) |
| `upsample.0`, `upsample.2` | (256, 64, 3, 3) | (256,) |
| `conv_last` | (3, 64, 3, 3) | (3,) |

Đặc tả đầy đủ bộ key ESRGAN:

| Key weight (mỗi key có bias tương ứng) | Shape weight | Shape bias |
|---|---|---|
| `conv_first` | (64, 3, 3, 3) | (64,) |
| `body.i.rdbj.conv1`, i=0…22, j=1…3 | (32, 64, 3, 3) | (32,) |
| `body.i.rdbj.conv2`, i=0…22, j=1…3 | (32, 96, 3, 3) | (32,) |
| `body.i.rdbj.conv3`, i=0…22, j=1…3 | (32, 128, 3, 3) | (32,) |
| `body.i.rdbj.conv4`, i=0…22, j=1…3 | (32, 160, 3, 3) | (32,) |
| `body.i.rdbj.conv5`, i=0…22, j=1…3 | (64, 192, 3, 3) | (64,) |
| `conv_body`, `conv_up1`, `conv_up2`, `conv_hr` | (64, 64, 3, 3) | (64,) |
| `conv_last` | (3, 64, 3, 3) | (3,) |

Nguồn chọn đúng artifact ESRGAN là mapping filename/Drive ID và hàm
`convert_esrgan_model` đọc `RRDB_ESRGAN_x4.pth` trong BasicSR revision
đã nêu, kết hợp kích thước và prefix hash. Không chọn checkpoint PSNR
hoặc Real-ESRGAN. Shape RRDB không tự nó phân biệt được lịch sử huấn luyện
với Real-ESRGAN; chưa có full publisher hash hoặc chuỗi chứng thực quá
trình chuyển đổi độc lập. Không trình bày kiểm schema là kiểm inference.

## Lệnh kiểm chứng chính

Các lệnh thực thi tại thư mục gốc; log đầy đủ lưu cục bộ, không công bố
trong repository:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q --require-set5
.\.venv\Scripts\python.exe scripts/crosscheck_bicubic.py --run-dir results/set5_numpy_x4_20261108_1627
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/download_models.py --item srcnn
.\.venv\Scripts\python.exe scripts/inspect_checkpoint.py data/quarantine/srcnn/SRCNN_v1.zip
.\.venv\Scripts\python.exe scripts/inspect_checkpoint.py data/quarantine/srcnn/SRCNN_v1.zip --extract-x4
```

Chỉ chạy từng item tải riêng. Không ghi đè file có sẵn; bytes phải đi qua
`.part`, kiểm HTML/magic/tên file nếu có Content-Disposition/kích thước/hash
rồi mới đổi tên. SRCNN chỉ nhận ZIP; EDSR/ESRGAN nhận đúng ZIP hoặc chữ ký
legacy đầy đủ, không chỉ hai byte đầu `80 02`. Không nạp bằng
`weights_only=False`. Script kiểm `.pth` dùng
`torch.load(weights_only=True, map_location='cpu')`; đây chỉ là kiểm
schema tensor, chưa instantiate model hay kiểm `strict=True` trên kiến trúc.

Trong quá trình lập báo cáo G3, helper cục bộ yêu cầu dòng vượt độ dài
source (IndexError), sau đó đọc log UTF-16 bằng UTF-8 (UnicodeDecodeError).
Hai log lỗi được giữ nguyên; helper cuối bỏ bước đọc log cũ, clamp theo số
dòng thực tế và đã chạy thành công. Các lỗi báo cáo này không thay đổi
artifact hoặc kết quả G0–G2.
