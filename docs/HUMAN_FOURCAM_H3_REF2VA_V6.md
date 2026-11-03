# Human V6 FOURCAM — H3 Ref2VA 訓練及四視角 A/B 驗收

**日期：** 2026-11-03
**範圍：** AnyMove Human V5→V6 四視角資料、fal.ai 原生 H3 Ref2VA LoRA 400-Step Training、兩組 Held-out A/B 評估。
**狀態：** 模型訓練完成、權重驗證通過、四支 A/B 影片完成；尚未通過 3D Skeleton / Joint Consistency 驗收。

本文件是受 Git 管理的可重現程式與證據快照。大型模型、影片、資料集仍在工作站 F:\anymove\runs\human，依 .gitignore 不納入 Git。本快照不是含有模型權重的獨立專案。

## 1. 資料集與鏡頭契約

| 項目 | V6 |
|---|---|
| 角色 | Human SkinMax V3，55-bone Blender FK |
| 訓練資料 | 28 Clips = V5 12 + V6 新增 16 |
| 動作組成 | 8 Lean + 20 Reach |
| Held-out | 8 Clips（本輪選 2 組） |
| 訓練畫布 | 896 × 896，1:1，2×2 |
| RGB / Pose | 各 107 Frames × 24 FPS，同步 |
| Anatomical colors | 左上臂 RED、左前臂 ORANGE、右上臂 BLUE、右前臂 GREEN |

| 2×2 位置 | Camera |
|---|---|
| Top-left | Front-left 45° orthographic |
| Top-right | Front-right 45° orthographic |
| Bottom-left | Rear / Back orthographic |
| Bottom-right | True overhead / Top-down |

原始 ZIP SHA-256：

    76faf527f9e5cd7f9521b8a42f306b7f07fda04f6c2d7cc2d4d233f57f102141

ZIP 16,154,035 bytes，共 112 個成員（28 × RGB、Caption、Pose Reference、Identity Frame）。本輪是 **fal 託管標準 H3 Ref2VA**，不是 H3 Max；不能假設這份 LoRA 可直接跨 Endpoint 使用。

## 2. 訓練與模型

| 項目 | 結果 |
|---|---|
| 1-Step fal Preprocess Debug | PASS（28 / 28） |
| Training API | minimax/h3/ref2va/trainer |
| Training Request ID | 01a1a04e-19bd-7340-8b68-c3bfd4566976 |
| Training Steps | 400，從 H3 Base 重新訓練 |
| Rank / Learning Rate | 16 / 0.0001 |
| Reference Conditioning Probability | 0.95 |
| Trigger Phrase | FOURCAM |
| Training Status | COMPLETED |
| Model | SafeTensors 65,628,056 bytes / 208 tensors；PASS |
| 訓練估算費用 | US$6；最終帳單以 fal 為準 |

模型 SHA-256：

    cc837f9122caa0619b5357a25563e319b8438102f07128ad057d2dbe70add6a2

模型位於工作站：

    F:\anymove\runs\human\lora_synth_A\conditioned_backend\fal_h3_ref2va\remote\h3_ref2va_v6_28_fourview_400step\model\human_v6_fourcam_h3_400.safetensors

fal Debug 對 Reference Video 做的是 9 張影格的稀疏預處理，而非逐格硬控制。fal 也未將 Blender Camera Extrinsics 作為數值幾何約束輸入。

## 3. 四支 Held-out 推論

| 動作 | Scale 0 Request ID | Scale 1 Request ID |
|---|---|---|
| Forward Left Reach | 01a1a10d-983c-7220-a919-5e5ee98b04ca | 01a1a10d-9b80-72a3-a037-0f3306ae1800 |
| Backward Right Reach | 01a1a10e-23f0-77b2-a801-fd0ac2fe2460 | 01a1a10e-26f9-7fc1-a523-300f9c6c2e7a |

推論使用 minimax/h3/reference-to-video/lora，Seed=42、Duration=5 s、768P、1:1、Prompt Expansion disabled。每組 A/B 共用相同 Pose Video、首幀、Prompt、權重與 Endpoint，只改 Scale 0 / 1。四個完成結果皆為 **768×768、124 Frames、24 FPS、5.184 秒**。

Blender 3D FK 真值：Forward Left 的左手最大 3D 位移 **0.4968 m**，右手固定；Backward Right 的右手最大 3D 位移 **0.4055 m**，左手固定。

## 4. RGB Color Mask IoU 驗證

對 2×2 四個視角中的 RED 左上臂及 BLUE 右上臂作影像分割，Blender 107 Frames 與生成 124 Frames 以 normalized clip phase 對齊，計算動作區間 Frames 24–96 的 RGB Mask IoU。

| 動作／全部四視角×雙上臂 | Scale 0 | Scale 1 | 差異 |
|---|---:|---:|---:|
| Forward Left Reach | 0.66831 | **0.69689** | +0.02858 |
| Backward Right Reach | 0.64090 | **0.67148** | +0.03058 |
| 兩動作平均 | 0.65460 | **0.68419** | +0.02959 |

| 動作手自身（四視角） | Scale 0 | Scale 1 |
|---|---:|---:|
| Forward Left RED | 0.76380 | 0.76772 |
| Backward Right BLUE | 0.51892 | 0.57673 |

16 組 Camera×Arm 配對中 13 組 Scale 1 IoU 較高。**Forward Left 動作手改善很小；Backward Right 改善較明顯。** 整體影像對齊較好，不能等同 3D Motion 正確。

## 5. 版本控制與重現

- **Source code**：tools/human_h3_ref2va_v6/，27 份經編譯檢查的原始腳本位元組快照。原先來源在 runs/human，被 .gitignore 排除；來源 SHA 存於證據快照。
- **Evidence**：docs/human_h3_ref2va_v6/，包含訓練前離線 QA、fal Debug Audit、模型權重 Hash、受控 A/B Sanitized Request、IoU、色塊中心指標及兩張縮圖。
- **注意 Windows 固定路徑**：歷史腳本引用 F:\anymove 與本機 runs/ 資料。移動到不同機器時，需要補齊原始資料、調整 Root 路徑及 Python / Blender / ffmpeg 環境。
- **不入 Git**：FAL_KEY / .env、模型 SafeTensors、原始資料 ZIP、RGB/Pose PNG 序列與生成 MP4；全數保留在工作站。
- **付費防護**：請不要重新執行 run_v6_28_fourcam_1step_debug.py、run_v6_28_fourcam_400step.py、run_v6_fourcam_*_ab.py 的 submit 命令。它們是歷史付費工作的可重現程式，不應重新支付同一輪任務。poll 與 collect 只查詢既有 Request，並不提交新的付費生成。
- **快照自檢**：C:\Python314\python.exe -m unittest discover -s tests -p test_human_ref2va_v6_snapshot.py -v

## 6. 尚未解決

1. 四視角同時存在並不保證同一套 3D 關節身分，先前 V2 雙視角 Reach 已出現上臂錯配。
2. 需要 Reader / Visibility-aware Joint Tracking、Known Camera Triangulation、Bone-Length 與 Reprojection 驗證。
3. 只有兩組 Held-out 和單一 Seed 42；Forward / Backward 在訓練集中已出現其他幅度，不可宣稱完全未知動作泛化。
4. LoRA 視覺參考是 soft/sparse，不是逐影格硬約束；不能把 IoU 當 3D Joint Accuracy。

**歷史狀態註記：** dataset_offline_acceptance.json 是付費訓練前的離線 QA 快照，當中的「未執行訓練」只是當時狀態，不是最新訓練狀態。
