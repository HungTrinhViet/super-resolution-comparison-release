# M7 — EXPLORATORY v2: low-gradient, mid-luminance crop

Đây là thí nghiệm khám phá riêng, được đề xuất **sau khi nhóm đã xem kết quả v1 và ảnh SR**. Vì vậy, bằng chứng này là **exploratory, không phải confirmatory**; quy tắc thuật toán không khôi phục được trạng thái quan sát mù. V2 không thay thế kết quả chính thức [M7](M7_COMPARISON_RESULTS.md), không dùng để đổi thứ hạng thuật toán hoặc sửa kết luận đã viết. Không chạy lại suy luận hay benchmark.

**3/5 vùng v2 trùng tọa độ với `hr_lowgrad64_v1` (img_001, img_003, img_005); chỉ img_002 và img_004 cho vùng mới.** V2 không tạo ra năm vùng mới.

Kết luận chính thức giữ nguyên: EDSR đứng đầu PSNR Y và SSIM Y trung bình; ESRGAN có trường hợp kém Bicubic, nhất là img_001 và img_004. Nền gần đen mà `hr_lowgrad64_v1` chọn ở img_002 và img_004 là hạn chế của phương pháp chọn crop. Đây không phải lý do thay đổi manifest v1; cả 15 crop, receipt và các sản phẩm v1 được giữ nguyên.

## Quy tắc và phạm vi

Tên quy tắc là `hr_lowgrad64_v2`, phiên bản thuật toán `low_gradient_mid_luminance_integer_v2`. Tên mô tả là **low-gradient, mid-luminance crop**; điều kiện cường độ và gradient không chứng minh loại vật thể hay vật liệu trong cửa sổ.

Chỉ đọc năm HR để chọn cửa sổ 64 × 64, quét toàn ảnh với bước 1 pixel. Với RGB uint8, dùng int64:

```text
V = 65481R + 128553G + 24966B
Y = 16 + (V + 127500) // 255000
N = 4096
variance_numerator = N·ΣY² − (ΣY)²
```

Y nguyên này chỉ là feature chọn crop, không thay đổi metric M1. Cửa sổ hợp lệ khi đồng thời `variance_numerator ≥ 4·N²` (population std ≥ 2) và `60·N ≤ ΣY ≤ 200·N` (mean Y trong [60, 200], bao gồm hai đầu). Hai điều kiện được kiểm bằng số nguyên chính xác.

Điểm là tổng bình phương chênh lệch Y của các cặp pixel kề ngang và dọc **nằm hoàn toàn trong cửa sổ**. Chọn điểm nhỏ nhất; hòa điểm chọn y nhỏ hơn, rồi x nhỏ hơn. Không có cửa sổ hợp lệ thì báo lỗi, không chọn tay. Sau khi hoàn tất lựa chọn trên HR mới đọc manifest v1 để ghi nhận tọa độ trùng; không tìm vùng thay thế.

## Manifest, khóa và tọa độ

[Manifest v2](comparison/exploratory_v2/crop_manifest_v2.json) có năm bản ghi, chứa hash HR, kích thước, x/y/w/h, điểm, `sum_y`, `sum_y2`, `variance_numerator`, mean, phiên bản, thời điểm khóa và `exploratory: true`. Tọa độ là gốc 0, tính từ góc trên trái HR.

| Ảnh | x | y | w × h | Điểm gradient | Mean Y | Trùng tọa độ v1 |
|---|---:|---:|---|---:|---:|---|
| img_001 | 352 | 264 | 64 × 64 | 15069 | 191.75830078125 | Có: hr_lowgrad64_v1 |
| img_002 | 0 | 88 | 64 × 64 | 114108 | 75.359619140625 | Không |
| img_003 | 0 | 192 | 64 × 64 | 338148 | 109.31103515625 | Có: hr_lowgrad64_v1 |
| img_004 | 216 | 86 | 64 × 64 | 388121 | 79.048828125 | Không |
| img_005 | 0 | 258 | 64 × 64 | 158266 | 71.007568359375 | Có: hr_lowgrad64_v1 |
| Tổng hợp | — | — | — | — | — | **3/5 trùng v1; chỉ img_002 và img_004 là vùng mới** |

- Khóa lúc `2026-11-09T07:25:53.282192+00:00`.
- SHA-256 manifest v2: `740c9915bfde18e57a140b6e2f9dea0df352cf88cd9ebdce2da24b14ffd41e60`.
- Receipt: `logs/m7_v2_crop_lock.log`, SHA-256 `bd6821e05133201dcdefb7e8ea04531ac151a5dd0d62aa8d415edc2a507cc39e`.
- Receipt ghi hash hai file code v2, hai module v1 được import và năm HR. Receipt là log cục bộ, không công bố trong repository.
- Render đầu bắt đầu lúc `2026-11-09T07:25:58.114039+00:00`, sau khóa; render đối chiếu bắt đầu lúc `2026-11-09T07:26:06.190878+00:00`.
- SHA-256 manifest v1 vẫn là `0f37d1901fae432e896aba6e4b8470a84cb9bc174a570a8fe9a14e08fb8e1b22`.

Các sheet cục bộ nằm trong `results/comparison/m7_exploratory_v2/crop_img_00{1..5}_SRF_4_hr_lowgrad64_v2.png`; bản render thứ hai nằm trong thư mục con `repeat_check/`. Mỗi sheet mang nhãn **EXPLORATORY v2**, thứ tự HR | Bicubic | SRCNN | FSRCNN | EDSR | ESRGAN, có run_id, cùng cửa sổ và phóng nearest ×4. Không chỉnh màu, độ sáng hoặc tương phản. Các PNG chứa pixel Set5/phái sinh nên không commit hoặc phân phối trong repository.

## Quan sát exploratory theo từng ảnh

Các quan sát sau được thực hiện trên cả năm sheet đã sinh, không chấm điểm, không xếp hạng và không phải human study. Phân biệt độ sắc biên, texture, artifact, màu và độ khớp với HR; một cửa sổ không đại diện toàn ảnh.

**img_001 — Quan sát:** cửa sổ có biến thiên màu sáng nhẹ; Bicubic và SRCNN làm mịn, FSRCNN và EDSR có biến thiên nhẹ hơn so với vùng phẳng của Bicubic; ESRGAN có texture lốm đốm rõ, không trùng hoàn toàn với các mảng nhỏ trong HR. Không thấy lệch hình học lớn trong cửa sổ. **Giả thuyết:** các mô hình tạo cách biểu diễn khác nhau cho tín hiệu biên độ thấp; texture GAN có thể làm vùng này ít phẳng hơn. **Giới hạn:** crop trùng v1 và có ít biên rõ, không đủ kiểm tra hình học hay kết luận texture là chi tiết phục hồi đúng; kết quả bất lợi toàn ảnh img_001 vẫn giữ nguyên.

**img_002 — Quan sát:** crop mới chứa vùng xanh có các dải tối và mảng màu; các đầu ra đều giữ bố cục lớn, Bicubic mịn, ESRGAN có nhiều biến thiên hạt nhỏ hơn các đầu ra còn lại. Sắc xanh nhìn chung được giữ nhưng mức chuyển màu và ranh giới khác HR. **Giả thuyết:** ràng buộc mean đã loại vùng gần đen và đưa vào vùng có biến thiên hữu hình hơn; khác biệt tần số cao phụ thuộc bộ lọc và ánh xạ học được. **Giới hạn:** cửa sổ vẫn mềm và độ tương phản thấp; không chứng minh các hạt nhỏ là đúng hay đủ đánh giá artifact trên toàn ảnh.

**img_003 — Quan sát:** vùng có các dải đỏ, trắng và chuyển sắc; Bicubic làm mềm các ranh giới, các mô hình học sâu thể hiện độ chuyển khác nhau. ESRGAN thêm biến thiên nhỏ trong vùng màu, trong khi HR có cấu trúc hạt riêng không khớp từng điểm. **Giả thuyết:** vùng có màu và biên mềm làm lộ đồng thời tác động làm mịn và tạo texture. **Giới hạn:** tọa độ trùng v1, không phải bằng chứng độc lập mới; không dùng độ sắc hoặc màu nhìn bằng mắt để suy ra độ chính xác pixel.

**img_004 — Quan sát:** crop mới là vùng xanh chứa texture dày trong HR; Bicubic, SRCNN, FSRCNN và EDSR thể hiện các mảng mềm hơn, ESRGAN có texture nhỏ rõ nhưng không trùng các chấm và cấu trúc của HR. Không thấy hoán đổi màu lớn; sai khác texture vẫn tồn tại. **Giả thuyết:** thông tin tần số cao đã mất trong LR khiến tái tạo và tổng hợp texture cho kết quả khác nhau. **Giới hạn:** đây không phải vùng texture da đã được quan sát ở M6; crop này không giải thích hoặc loại bỏ việc ESRGAN kém Bicubic về PSNR Y và SSIM Y toàn img_004. Texture sinh không được coi là chi tiết đúng.

**img_005 — Quan sát:** cửa sổ có ranh giới tối–sáng và các dải chéo trên vùng màu hồng; Bicubic làm mềm dải, các mô hình giữ mức nét khác nhau. ESRGAN thể hiện các dải và hạt nhỏ có hình dạng khác HR. Không thấy dịch chuyển lớn của ranh giới chính. **Giả thuyết:** các dải lặp và ranh giới là vị trí nhạy với cách tăng cường tần số cao. **Giới hạn:** crop trùng v1 và chỉ bao phủ một phần ảnh; nhìn rõ dải không đồng nghĩa tái tạo đúng màu hoặc texture trên toàn ảnh.

## Kiểm thử và tái lập

TDD ghi nhận lần RED đầu do module v2 chưa tồn tại; sau triển khai, tám unit test tổng hợp PASS. Test gồm mean 60/200, tổng nguyên lệch đúng 1 đơn vị ra ngoài hai biên, std đúng ngưỡng, vùng hằng không hợp lệ, tie y/x, trùng tọa độ không đổi vùng, ảnh quá nhỏ và đối chiếu chọn điểm bằng duyệt trực tiếp trên ảnh nhỏ. Mảng tổng hợp chỉ dùng kiểm thử.

`pytest -q --require-set5`: **142 passed**. Crosscheck Bicubic: **PASS 5/5**, trung bình PSNR Y `28.43175155`, SSIM Y `0.81126246`; không chạy lại benchmark.

Kiểm tra độc lập đọc HR, tính lại điều kiện nguyên và điểm trên mọi cửa sổ: năm lựa chọn đều khớp. Hai lần render có **5/5 hash byte PNG và 5/5 hash pixel giống hệt**. Cả 30 panel crop bằng chính xác pixel nguồn phóng nearest ×4, không biến đổi màu. Hash **5 HR + 25 SR** và **28 file commit M7 chính thức** trước/sau đều không đổi. [Bản ghi tái lập v2](comparison/exploratory_v2/reproducibility_v2.json) lưu hash, thứ tự khóa/render và số lượng kiểm chứng; không chứa pixel ảnh.

| Lệnh | Kết quả | Bằng chứng cục bộ |
|---|---|---|
| `.\.venv\Scripts\python.exe -m pytest -q tests/test_exploratory_crops_v2.py` | PASS 8 | `logs/m7_v2_unit_boundaries.log` |
| `.\.venv\Scripts\python.exe -m pytest -q --require-set5` | PASS 142 | `logs/m7_v2_pytest.log` |
| `.\.venv\Scripts\python.exe scripts/build_exploratory_crops_v2.py lock` | PASS | `logs/m7_v2_lock_command.log` |
| `.\.venv\Scripts\python.exe scripts/build_exploratory_crops_v2.py render` | PASS | `logs/m7_v2_render.log` |
| `.\.venv\Scripts\python.exe scripts/build_exploratory_crops_v2.py render --output-dir results/comparison/m7_exploratory_v2/repeat_check --report logs/m7_v2_render_repeat.json` | PASS | `logs/m7_v2_render_repeat.log` |
| `.\.venv\Scripts\python.exe logs/m7_v2_audit.py` | PASS | `logs/m7_v2_audit.log` |
| `.\.venv\Scripts\python.exe scripts/crosscheck_bicubic.py --run-dir results/set5_numpy_x4_20261108_1627` | PASS 5/5 | `logs/m7_v2_crosscheck.log` |

Các log/helper trong bảng là bằng chứng cục bộ, không công bố trong repository. `lock` chỉ chạy khi manifest và receipt chưa tồn tại; `render` yêu cầu receipt đúng và thư mục output mới. Không chạy lại các lệnh trên với cùng đường dẫn output đã có. Không sửa manifest hoặc code đã khóa để tạo một lựa chọn khác.
