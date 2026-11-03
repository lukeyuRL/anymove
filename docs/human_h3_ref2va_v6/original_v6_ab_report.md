# AnyMove V6 FOURCAM — fal H3 Ref2VA 400-Step Held-out A/B

日期：2026-11-03
狀態：**COMPLETE / 4 of 4 fal AI videos completed, downloaded, decoded and validated**
類型：fal.ai 原生託管 `minimax/h3/reference-to-video/lora`，不是自架 Serverless。

## 1. Controlled evaluation

模型來自 V6 **28 支**訓練影片、400 Steps、Rank 16、觸發詞 `FOURCAM`。

- 訓練 Request ID: `01a1a04e-19bd-7340-8b68-c3bfd4566976`
- 權重 SHA256: `cc837f9122caa0619b5357a25563e319b8438102f07128ad057d2dbe70add6a2`
- Dataset: 28 Train, 8 Held-out; this test uses 2 previously unseen clip IDs.
- Endpoint: `minimax/h3/reference-to-video/lora`
- Source: 每個動作 107 Frames × 24 FPS，原生 896×896 2×2 四視角。
- Output: 124 Frames × 24 FPS，768×768，實測 5.184 秒。
- Common prompt: `FOURCAM` + 對應 Heldout 訓練格式 caption，明確標示四視角與手臂顏色。
- Common Seed: 42. Resolution: 768P, aspect ratio: 1:1, requested duration: 5s.
- 同一動作的 A/B 使用完全相同的 Pose Video、Human 首幀、Prompt、LoRA 權重和 Endpoint；只改 `loras[0].scale` = 0 或 1。
- 全程僅**新增四筆使用者批准的 fal 付費生成任務**；沒有額外訓練、其他 Seed 或重複提交。

四視角：
| Quadrant | Camera |
|---|---|
| TL | Front-left 45° orthographic |
| TR | Front-right 45° orthographic |
| BL | Rear / Back |
| BR | True top-down overhead |

## 2. Results

Fal 回傳兩個動作 × Scale 0/1，皆 `COMPLETED`，四支影片已下載並驗證可解碼。

### Controlled action and correct 3D source truth (not generated motion)

| 動作 | Blender GT：動作手的最大 3D 位移 | Blender GT：另一手 |
|---|---:|---:|
| Forward Left Reach | Left Hand 0.4968 m | Right Hand 0 m |
| Backward Right Reach | Right Hand 0.4055 m | Left Hand 0 m |

來源真值為 Blender 55-bone FK。這項結果僅保證合成資料正確，**不是**生成影片的 3D Joint 誤差。

### RGB color-mask IoU, active frames 24–96

使用 2×2 各 384×384 影像區塊的紅色左上臂／藍色右上臂遮罩，與 Blender Source RGB 的對應顏色逐影格比較。Source 107 Frames 與 Generated 124 Frames 以 **normalized motion phase** 對齊。IoU=1 完全重疊；數字越大代表 2D 畫面越接近 Source。

| Held-out 動作 | Scale 0 | Scale 1 | Scale 1 − Scale 0 |
|---|---:|---:|---:|
| Forward Left Reach, 4Views×2Arm average | 0.66831 | **0.69689** | +0.02858 |
| Backward Right Reach, 4Views×2Arm average | 0.64090 | **0.67148** | +0.03058 |
| Combined 2 cases | 0.65460 | **0.68419** | +0.02959 |

16 個 view×arm 配對中，13 個 Scale1 全片平均 IoU 優於 Scale0。

**只看動作手、四視角平均 IoU：**

| Target Arm | Scale 0 | Scale 1 | Difference |
|---|---:|---:|---:|
| Forward Left RED | 0.76380 | 0.76772 | +0.00392 |
| Backward Right BLUE | 0.51892 | 0.57673 | **+0.05781** |

重要：Forward Left 的動作手改善 **非常小**，即使整體 IoU 增加 0.02858。不能將總體 RGB Mask 改善誤認為骨架 Motion 精度等量改善。

### Visual review

- 四支影片的 2×2 layout、視角順序、Avatar 外觀與紅/藍上臂身份標記維持可辨識。
- Forward Left：左紅色上臂／橘色前臂執行動作，右藍／綠手臂主要維持放下。Scale0 的右上視角（front-right 45°）在部分影格出現更明顯的手臂方向偏移；Scale1 在該視角與 Source 的 IoU 較好。
- Backward Right：右藍／綠手臂向後，左紅／橘手臂維持相對靜止。Scale1 在動作手四視角的 RGB Mask IoU 明顯改善，但仍有前左 45°角度遮擋／輪廓差異。
- 對同一動作、相同 Seed，模型產生的兩個變體具有相近的時序、人體身份與場景布局。

### Important limitations

1. **2D 紅／藍上臂 pixel-mask IoU 不等於 3D Joint Error，也不是 multiview reprojection loss。** 橋接 AnyMove 應另做 3D Skeleton Reconstruction 和 FK 骨長／camera projection 檢查。
2. Camera 的 2×2 tile 在同一影片並存，不代表 4 個 Camera 真正共享一致的 3D Motion。尤其上肢遮擋時，左右手可能發生 identity ambiguity。
3. fal H3 Ref2VA 的 Reference Video 經過稀疏/soft conditioning 前處理；並不使用原始數值 camera extrinsics，也不是逐影格硬約束。
4. GT 107 Frames 和生成 124 Frames 的 normalized-phase 比較不保證時間準確一致。
5. 只有兩個 Held-out Action，各一個 Seed=42，且 Forward/Backward 方向在 V6 Train 已有其他幅度實例。不能據此宣稱可泛化到完全未知 Motion。

## 3. Evidence and results

Results root:
`F:\anymove\runs\human\lora_synth_A\conditioned_backend\fal_h3_ref2va\remote\h3_ref2va_v6_28_fourview_400step\heldout_ab_v1`

- `forward_left/scale0/h3_ref2va_scale0.mp4`
- `forward_left/scale1/h3_ref2va_scale1.mp4`
- `backward_right/scale0/h3_ref2va_scale0.mp4`
- `backward_right/scale1/h3_ref2va_scale1.mp4`
- `forward_left/analysis/forward_left_scale0_vs_scale1.mp4`
- `backward_right/analysis/backward_right_scale0_vs_scale1.mp4`
- `forward_left/analysis/forward_left_source_vs_scale0_vs_scale1.jpg`
- `backward_right/analysis/backward_right_source_vs_scale0_vs_scale1.jpg`
- `forward_left/analysis/forward_left_metrics.json`
- `backward_right/analysis/backward_right_metrics.json`
- `forward_left/analysis/forward_left_color_pixel_iou.json`
- `backward_right/analysis/backward_right_color_pixel_iou.json`
- `FINAL_4CLIP_SUMMARY.json`

Fal original output request IDs:
- Forward Left Scale0: `01a1a10d-983c-7220-a919-5e5ee98b04ca`
- Forward Left Scale1: `01a1a10d-9b80-72a3-a037-0f3306ae1800`
- Backward Right Scale0: `01a1a10e-23f0-77b2-a801-fd0ac2fe2460`
- Backward Right Scale1: `01a1a10e-26f9-7fc1-a523-300f9c6c2e7a`

## 4. Decision / next stage (not executed)

**目前確認 V6 FOURCAM LoRA 能在 fal 原生 H3 Ref2VA 工作，且 Scale=1 在這兩組的 color-mask alignment 改善約 0.03 IoU。**

要用於 AnyMove 回推任意骨架 3D Motion，仍應先建立最關鍵的評測：
- 使用已知 Blender Camera extrinsics，從每個生成 Camera 的 2D Joint 或 color-coded landmarks 做 **joint association**。
- 做 4-view triangulation 或 3D Solver 後，用 reprojection error、bone length、left/right limb identity 與 per-frame temporal jitter 獨立驗證。
- 做多 Seed 的 heldout A/B 之後，才決定是否新增 Motion Prior、增加 LoRA data/Steps，或改成具硬性 Pose Control 的模型。

**未自動啟動額外付費 AI 推論或新的訓練。**
