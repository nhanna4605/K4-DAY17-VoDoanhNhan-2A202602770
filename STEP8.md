# STEP 8 — Phân tích kết quả benchmark (Day 17: Memory Systems for AI Agent)

Sinh viên: Võ Doãn Nhân · MSSV 2A202602770

Mọi số liệu dưới đây lấy từ `python src/benchmark.py` (chế độ offline, tất định: chạy hai lần trên `state/` sạch cho đúng cùng một bảng). Estimator token là `len(text)//4`, nên con số tuyệt đối chỉ có ý nghĩa **so sánh giữa hai agent**, không phải số token của một tokenizer thật.

## Kết quả

**Standard Benchmark** (10 hội thoại × ~10 lượt, user `dungct`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 594 | 9807 | 0.00 | 0.20 | 0 | 0 |
| Advanced | 808 | 18111 | 1.00 | 1.00 | 349 | 0 |

**Long-Context Stress Benchmark** (1 hội thoại 16 lượt rất dài, user `dungct_stress`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 99 | 20822 | 0.00 | 0.20 | 0 | 0 |
| Advanced | 180 | 12207 | 1.00 | 1.00 | 252 | 2 |

**Ablation — tắt compact** (`COMPACT_THRESHOLD_TOKENS=1000000`, giữ nguyên mọi thứ khác): bảng stress của Advanced thành `180 | 22185 | 1.00 | 1.00 | 252 | 0`.

Ba lớp memory trong code: **short-term** = danh sách message của một `thread_id` (cả hai agent có); **persistent** = `User.md` ở `state/profiles/<user>/User.md` do `UserProfileStore` quản lý (chỉ Advanced); **compact** = `CompactMemoryManager` nén message cũ thành `summary` (chỉ Advanced).

## 1. Vì sao Advanced có recall tốt hơn Baseline?

Số liệu: `Cross-session recall` của Advanced là **1.00** ở cả hai bảng, của Baseline là **0.00**; `Memory growth (bytes)` của Baseline bằng **0** (nó không ghi file nào), của Advanced là 349 và 252 byte.

Cơ chế: mọi `recall_questions` được hỏi ở **thread mới**. Baseline khóa `self.sessions` theo `thread_id`, nên thread mới trống và nó trả "chưa có thông tin". Advanced đi qua `extract_profile_updates()` ở từng lượt, ghi fact ổn định vào `User.md` bằng `upsert_fact()`, rồi `_offline_response()` đọc lại chính file đó ở thread mới. Cột `Response quality` của Baseline là 0.20 chứ không phải 0 vì công thức heuristic cho 20% điểm "độ gọn" (câu trả lời ngắn), còn độ phủ fact và độ tự tin bằng 0.

Giới hạn: Baseline **không bị làm kém đi cố ý** — trong cùng thread nó trả lời đúng (có test `test_cross_session_recall` kiểm cả vế đó); nó chỉ quên khi sang thread mới.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Số liệu (bảng Standard): Advanced tốn **808** agent tokens (Baseline 594) và **18111** prompt tokens (Baseline 9807), tức gần gấp đôi ở cột prompt, trong khi `Compactions = 0`.

Cơ chế: hội thoại ~10 lượt chưa bao giờ vượt `compact_threshold_tokens = 1200`, nên compact không chạy và không tiết kiệm được gì. Trong lúc đó, mỗi lượt Advanced vẫn phải mang `User.md` vào prompt (`_estimate_prompt_context_tokens()` cộng `User.md` + summary + recent) và còn sinh thêm câu xác nhận "Đã lưu vào User.md: …" nên agent tokens cũng cao hơn. Đây là chi phí cố định của persistent memory; nó chỉ được "trả lại" nhờ recall 1.00 chứ không nhờ token.

## 3. Vì sao compact giúp Advanced có lợi thế ở hội thoại dài?

Số liệu (bảng Stress): `Prompt tokens processed` của Baseline là **20822**, Advanced là **12207** (thấp hơn ~41%) với **2 compactions**. Ablation cho thấy chính compact tạo ra khoảng cách đó: tắt compact thì Advanced vọt lên **22185** — còn *cao hơn* Baseline, vì vẫn phải mang thêm `User.md`. Nghĩa là compact tiết kiệm ~45% prompt (22185 → 12207) và bù được cả chi phí profile.

Cơ chế: Baseline mang toàn bộ lịch sử vào mọi lượt nên prompt tăng gần tuyến tính theo số lượt; các turn trong stress dài ~1000 ký tự. Advanced giữ `compact_keep_messages = 4` message gần nhất nguyên văn, phần cũ thành tối đa 8 dòng summary (mỗi dòng ≤ 120 ký tự).

Lưu ý quan trọng: compact **chủ yếu tối ưu `Prompt tokens processed`, không phải `Agent tokens only`**. Ở bảng stress, Agent tokens only của Advanced (180) thậm chí cao hơn Baseline (99): nó sinh ra thêm câu xác nhận và câu trả lời recall có nội dung, trong khi Baseline chỉ trả lời "chưa có thông tin". Compact không làm câu trả lời của agent ngắn đi; nó làm *ngữ cảnh đưa vào* nhỏ đi.

## 4. File memory tăng trưởng ra sao, rủi ro gì?

Số liệu: `User.md` của `dungct` tăng thêm **349 byte** sau 10 hội thoại (8 dòng fact), của `dungct_stress` tăng **252 byte** sau 16 lượt rất dài. Tăng trưởng chậm vì ghi theo *khóa*: fact đơn trị (`name`, `location`, `profession`, `drink`, `food`, `pet`) bị ghi đè bằng `edit_text()` khi có correction; fact đa trị (`style`, `interests`) được gộp, không trùng và chặn ở 8 mục.

Rủi ro quan sát được:
- **Lưu thừa/sai.** Sau 10 hội thoại, `interests` chứa `markdown` — chỉ là một câu nói thoáng qua ("Mình thích markdown vì dễ mở…"), không phải sở thích kỹ thuật ổn định. Bộ trích xuất bằng regex không phân biệt được "ổn định" và "nhất thời".
- **Compact làm mất chi tiết.** Summary chỉ giữ 120 ký tự/message và tối đa 8 dòng, nên chi tiết cụ thể của các tin tức trong stress test (số liệu, ngày tháng) không còn sau khi nén; chỉ các fact đã được tách vào `User.md` là còn.
- **Chỉ đúng với dữ liệu mẫu.** Recall 1.00 đến từ bộ regex tiếng Việt viết cho kiểu câu trong `data/`; với cách diễn đạt khác nó sẽ bỏ sót. Đây là heuristic offline, không phải bằng chứng về chất lượng với LLM thật.

## Bonus — Conflict handling + Confidence threshold

**Bài toán.** Dữ liệu cố tình có correction (Đà Nẵng ↔ Huế; backend → MLOps engineer) và nhiễu ("Hà Nội" chỉ là nơi đi họp, "product manager" chỉ là câu đùa). Ghi bừa sẽ khiến `User.md` giữ đồng thời hai nơi ở/nghề nghiệp, hoặc ghi cả câu đùa.

**Cách làm.**
- *Conflict handling:* `upsert_fact()` với khóa đơn trị gọi `edit_text()` để **sửa đúng dòng cũ** thay vì thêm dòng mới; bộ trích xuất bỏ mệnh đề đã bị phủ định (`không còn …`, `chứ không …`, `lúc đầu …`, `dù trước đó …`) nên giá trị cũ không được ghi lại.
- *Confidence threshold:* `extract_profile_candidates()` gán độ tin cậy cho từng fact; câu hỏi (`?`, `là gì`, `nhắc lại…`) bị bỏ, câu có từ giảm chắc chắn (`đùa`, `cân nhắc`, `có lẽ`, `giả sử`…) bị nhân 0.5, và chỉ fact đạt `MIN_CONFIDENCE = 0.6` mới vào `User.md`.

**Cải thiện đo được.** Test `test_correction_replaces_old_fact_in_user_md` kiểm chứng `User.md` chỉ còn đúng một dòng `location` (Huế) và một `profession` (MLOps engineer) sau correction; `test_noise_and_questions_are_not_written` kiểm chứng "đi họp ở Hà Nội", "chuyển sang product manager" (câu đùa) và câu hỏi không bị ghi. Cần nói thẳng: **bảng benchmark không nhìn thấy lợi ích này** — `expected_contains` chỉ kiểm chuỗi *có xuất hiện*, nên một agent giữ cả "Huế" lẫn "Đà Nẵng" vẫn được điểm recall đầy đủ. Lợi ích nằm ở độ đúng của câu trả lời và độ gọn của `User.md`, không ở con số recall.

**Rủi ro tạo thêm.** (1) Ngưỡng 0.6 và danh sách từ giảm chắc chắn là hằng số viết tay: đặt quá cao thì bỏ sót fact thật, quá thấp thì lọt nhiễu. (2) Cắt mệnh đề phủ định bằng chuỗi (`chứ không`, `không còn`) có thể xóa nhầm một fact đúng nằm sau từ khóa đó. (3) "Giá trị mới nhất thắng" nghĩa là một câu nói nhầm hoặc bị hiểu sai sẽ ghi đè fact đúng mà không có lịch sử để khôi phục — hệ thống mạnh hơn nhưng cần thêm guardrail (lưu lịch sử thay đổi, xác nhận với người dùng trước khi ghi đè fact quan trọng).

## Quy ước cấu hình để chạy lại

Chế độ mặc định là **offline**, không cần biến môi trường nào. Chế độ live chỉ bật khi `LAB_LIVE=1` và có đủ SDK + API key; nhánh live **chưa được kiểm thử trong bài này** (không có API key), nên mọi số liệu ở trên đều là offline. Biến môi trường do `load_config()` đọc: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE`, `JUDGE_PROVIDER`, `JUDGE_MODEL`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `CUSTOM_BASE_URL`, `CUSTOM_API_KEY`, `OLLAMA_BASE_URL`, `COMPACT_THRESHOLD_TOKENS` (mặc định 1200), `COMPACT_KEEP_MESSAGES` (mặc định 4), `LAB_LIVE`.
