# Dino 全身 Motion Accuracy — Blender Ground Truth 第一階段

Branch: `feature/dino-motion-quality-p123`。本文件為 **可執行的合成 Ground Truth 基準**，
不是已完成的 Dino 實際影片 3D 準確率認證。

## 目標與前提

已驗證的 Dino Static Pose Dataset 是 2,200 poses，92 keypoints、67 bones；
其 88 個驗證姿勢能測 2D Reader，**不能**直接用來評估連續 3D Motion。
之前 P1 Guarded、P2 Weighted 的差異是兩個 Solver 輸出間的距離，不能當成
Ground Truth Error。因此先建立一套保留 AnyMove 主幹、不改 iClone Motion JSON 的
全身連續動畫評估。

**重用現有程式，沒有另建相互矛盾的 Blender FK 模擬器：**

1. `mocap.exam.make` 建立隨機種子固定、關節角在 Rig Limits 內的連續 Euler／Root 動畫
   （控制測試，**不是**真實 Dino 步態、不是 Mocap、不是已驗證的 AI Video 動作）。
2. `blender/apply_motion.py` 對原始 Dino `rigged.blend` 插入動畫 Key，
   再用 Blender 自己評估 Pose Bone 之世界座標，輸出
   `gt_render/joints.json`，**不使用** `mocap.solve.Rig.forward` 產生 GT。
3. 同步渲染 Front／Side 兩個正交視角，組合成標準 1280×720 動畫。
   `tools.audit_dino_gt_camera` 會獨立計算 Blender Camera JSON 與 AnyMove
   原有幾何的投影差，超過 0.2px 就停止，避免錯誤相機製造偽差。
4. `mocap.detect` 使用原始 Dino `reader_hr_v1.pt`，加 `--no-calib`
   （本來就是標準 layout，不再跑第一張畫面的 ECC），保持與 Blender GT 同 FPS。
5. `mocap.solve --no-align` 求解 Baseline；使用完全相同視頻與基礎 Reader，
   `mocap.track --guard-large-disagreements` 產生 P1 Guarded 再求解。
   兩組皆 `--no-align`，因為 Blender 兩個相機本來就是同步的，避免額外 DTW 混淆。
6. 可選擇已完成的 P2 Quality Reader snapshot，使用獨立的
   `tools.make_dino_p2_weighted_detection` 僅調整 Solver 權重、不改 2D Reader
   觀測，再產生 P2 Weighted Motion。
7. `tools.evaluate_dino_full_body` 讀取每個 Motion + **Blender 真值**，
   嚴格檢查 92 keypoints／Bone 順序、T、FPS、Detection SHA-256，
   再輸出全身／分群／每關節／每影格的真實 3D 位置及時間導數誤差。

## Windows Git Bash：第 1 步 — 24-frame Smoke

**先等目前任何訓練程序結束／停止，再更新正在使用的 Repo。**
不要在正在執行的 `mocap.train` 同一 Working Tree 做 `git pull`，
也不要同時執行兩個 Full Training Writer。
如果 GPU 已經被正在進行的 P2 訓練占用，先以現有結果為準，不急著並行 Cycles 渲染。

```bash
cd /f/anymove
git switch feature/dino-motion-quality-p123
git pull --ff-only

export PYTHON=python.exe
export BLENDER="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
export MOCAP_CREATURE=assets/dino
export DINO_GT_RENDER_ENGINE=cycles
export DINO_GT_SAMPLES=8

# 將 GT、渲染、Detection、Motion、QA 全放在獨立的新資料夾
bash tools/eval_dino_full_body.sh \
  runs/dino/gt_experiments/seed7_smoke_24 \
  24 7
```

輸入的 `assets/dino/rigged/rigged.blend`、
`assets/dino/rigged/rig_info.json`、原本 2,200 poses、
`test_walk` AI Video、`reader_hr_v1.pt` 及任何舊 `motion.json` 均不會被覆寫。

若 Cycles 渲染環境失敗，可改 `DINO_GT_RENDER_ENGINE=workbench`，
並使用**新的輸出資料夾**重新執行 Smoke；但 Workbench 與訓練用 Cycles 的
影像風格可能不同，Reader Accuracy 不可直接跨渲染 Domain 比較。

先檢查：

- `camera_layout_audit.json` 狀態應為 `blender_cameras_match_anymove_orthographic_layout`。
- `gt_render/joints.json` 具有 24×92 個 Blender World Joint。
- `gt_video.mp4` 是同步 Front/Side 的 24 FPS layout。
- `full_body_quality/full_body_report.json` 包含 Baseline + P1 Guarded 兩組，
  每組全身和分部位 3D Ground Truth Error。
- `full_body_quality/by_joint.csv` 應有 184 行（2×92）；
  `by_frame.csv` 有 48 行（2×24）。均不計表頭。

若某些 Dino Rig 的 Bone 命名與分類關鍵字不符，所有未分類 Joint
仍保留在 `unclassified`，**絕不默默丟掉**。先檢查
`unclassified_joint_names` 再建立精確 group override。
特別注意 `joint:BoneName:tail` 中的 `:tail` 是 Bone 端點，
**不代表** Dino 尾巴，所以分類會先排除該後綴。

## 第 2 步 — 124-frame + P2 全身對照

在 24-frame 完整走通、確認 joint group／相機與 Blender FK Basis
後，選擇一份**已完整寫入**的 Quality-only Reader checkpoint snapshot。
若 `reader_hr_quality_p2_headonly.pt` 仍被訓練程序持續更新，先等
訓練停止／完成，避免混用不同 Epoch 的模型。

```bash
# snapshot 不可與舊實驗同名，避免 cp -n 靜默保留舊模型
cp -n runs/dino/reader_hr_quality_p2_headonly.pt \
  runs/dino/reader_hr_quality_p2_gt_seed7.pt

bash tools/eval_dino_full_body.sh \
  runs/dino/gt_experiments/seed7_full_124 \
  124 7 \
  runs/dino/reader_hr_quality_p2_gt_seed7.pt
```

產出：

```text
runs/dino/gt_experiments/seed7_full_124/
  gt_motion.json                  # deterministic test motion
  gt_render/
    joints.json                   # independent Blender world-joint GT
    animation.blend               # keyed scene for visual review
    cam_front.json / cam_side.json
    front/f_*.png / side/f_*.png
  camera_layout_audit.json
  gt_video.mp4
  det_baseline.npz
  motion_baseline.json
  det_p1_guarded.npz
  motion_p1_guarded.json
  det_p2_quality.npz               # optional P2 input/output scores
  det_p2_weighted.npz              # original UV + P2 error weights only
  motion_p2_weighted.json
  full_body_quality/
    full_body_report.json
    by_joint.csv
    by_frame.csv
```

P2 的 synthetic-cutout visibility 值**不會**進入 Solver；只有
預測 2D 誤差經保守權重啟發式降權，而且會檢查 2D UV／Reader Peak
與 Baseline 一致。若 P2 Reader 和 Baseline 的 keypoints 不一致，
直接 Fail，不會誤報成「Quality Head 改善」。

## 全身品質報告的正確解讀

所有誤差都相對 **同一個 Blender GT**，不是 A/B 兩次 Solver
互相減：

| 組別 | 要檢查的項目 |
| --- | --- |
| head_neck | 頭頸／下顎 3D MAE、P95 和轉頭影格 |
| torso_root | Spine／Pelvis 與 Root Keypoint World Error |
| forelimbs | 前肢相對 3D 真值的 MAE 與速度誤差 |
| hindlimbs | 後肢相對 3D 真值的 MAE 與加速度誤差 |
| feet_toes | 腳掌／腳趾 MAE、P95、速度誤差 |
| tail | 尾巴各節 MAE、P95、尾端連續性 |
| unclassified | 未能從命名分類的 Keypoints，必須人工檢查 |

`velocity_error_cm_s` 與 `acceleration_error_cm_s2`
測量與 Ground Truth 的時間導數差異，不是只看是否「更平滑」。
`root_world_keypoint_position_error_cm` 用原始 Blender Root 骨骼 Head
真值比對，避免直接假設 Local Root Translation 與 World Space 數值
相等。如果 GT 動作本身沒有地面接觸段，不能宣稱「Foot Sliding 已改善」；
應改用真實步態／有接觸標註的 Blender 動作驗證。

請回傳 `full_body_quality/full_body_report.json` 和
`full_body_quality/by_joint.csv`。這份報告能指出哪個部位需要
下一步 Multi-view／Temporal／Uncertainty／Motion Prior 改善。
**合成 3D 真值上的好結果，不保證在 H3Max 自然步態、相機不同步、
自遮擋或真實材質的影片上會同樣好。**

## 後續（不在第一階段擅自開啟）

先比較 2 個以上獨立隨機 Seed，再進入經認證的連續 Blender／iClone
真實步態動作。若尾巴／腳趾持續有高誤差，才評估對**同一段 GT
動畫**渲染 Top／Elevated 第三視圖，與 Front/Side 在
同一份 Blender 3D 真值下比較；不可用不同動作假裝同步第三視角。
未驗證之前，不改 iClone FK/Root Writer、不啟用第三視角為預設。

## 2026-11-01 live Dino 24-frame smoke (LukeYu-2024)

**實際執行完成（非推測）**：Remote Desktop Commander 連線到
`LukeYu-2024`，只停掉了使用原始
`runs/dino/reader_hr_quality_p2_headonly.pt` 的 P2 訓練 Python。
最近完成的訓練備份為 `epoch_013_full.pt`（Epoch 0–12），原始
`F:\\anymove` Working Tree 不變。GT 測試用獨立 repo
`F:\\anymove_gt_eval_20261101`，Dino Rig 僅使用本機連結，
`reader_hr_v1.pt` 與固定的
`reader_hr_quality_p2_eval_snapshot.pt` 僅作唯讀推論；
新 Synthetic GT／影片／Detection／Solver／Report 在隔離目錄
`runs/dino/gt_experiments/seed7_smoke_24/`。

- 24 Frames、24 FPS、92 keypoints、67 bones，Seed 7，Blender Cycles。
- 正交 Camera Audit **PASS**：Front 最大投影差 0.000048px，
  Side 0.000057px；Front/Side 最佳 Lag 0 Frames。
- 原始 Reader 2D peak 平均 Front 0.90、Side 0.93。
- P2 新 Reader 的 UV 與 Heatmap Peaks 相對 Baseline 完全相同（兩者
  Max Difference 都是 0）；P2 只在原始座標上對 84/2208 Front 和
  40/2208 Side Joint-Frames 降權。
- 最終 Report 使用附檔
  `dino_joint_group_overrides.json`，將 17 個 Dino Calvicle／Index／Mid
  Keypoints 正確歸入 Forelimbs，**全部 92 個關節都已分類**。
- 以下是對 **相同 Blender World-joint GT** 的位置誤差（cm），
  不是兩組 Solver 相互之間的距離。

| Body part | Keypoints | Baseline MAE (cm) | P1 Guarded MAE (cm) | P2 error-weighted MAE (cm) |
|---|---:|---:|---:|---:|
| 全身 | 92 | **3.35111** | 3.77252 | 3.30758 |
| Head / Neck | 17 | **4.02197** | 4.30098 | 4.03594 |
| Torso / Root | 5 | **2.07318** | 2.62201 | 2.06410 |
| Forelimbs (incl. fingers) | 21 | **2.47506** | 2.92241 | 2.46643 |
| Hindlimbs | 6 | **2.40445** | 3.10417 | 2.39938 |
| Feet / Toes | 34 | **3.10201** | 3.42297 | 3.08030 |
| Tail | 9 | **6.41011** | 7.16315 | 6.04932 |

**重點判斷**：P1 Guarded 在此受控 24-frame 測試的全身 MAE
增加 **約 12.6%**，且**所有身體群組**皆變差；
Root World Keypoint MAE 2.02491→3.66479cm。
因此目前不能以單純平滑、2D Reprojection 來推動 P1 預設啟用。

P2 Weighting 全身 MAE 降 **約 1.3%**，其中 Tail MAE 約降 **5.6%**，
但 Head／Neck 4.02197→4.03594cm 小幅惡化，
Root World Keypoint 2.02491→2.01953cm 幾乎沒有變化。
全身 P95：Baseline 9.37482cm、P1 10.31365cm、P2 9.22854cm。
模型對尾巴有受控小幅改善的證據，但**不能宣稱自然 Dino
走路或全身 Motion 已獲驗證改善**。

**尚待驗證**：另一個獨立 Seed／124 frames、實際有接地標註的
Walk／Turn 真實 Blender 動作、AI Video 前後視角同步、
自遮擋、模型的跨場景泛化，以及 Blender/iClone
Motion 視覺驗收。若需第三視角，必須基於同一段 GT 動作做
Top／Elevated 消融實驗，不可把不相關的生成畫面當同步真值。

實測 Report 與每關節、每影格 CSV 只保留在使用者本機測試目錄；
本次 GitHub Commit **不包含**私人 Dino Mesh、
Blender Rig、Checkpoint 或其原始資料。

## 2026-11-01 live Dino 124-frame seed 11 results

**已在使用者 Windows 電腦完成實際執行**，不是僅用 GitHub CI 模擬。
實驗位置：`F:\\anymove_gt_eval_20261101\\runs\\dino\\gt_experiments\\seed11_full_124`。
另一個固定 Seed 11、124 frames、24 FPS、92 Joint 的 Blender Cycles
受控連續動作；使用跟 Seed 7 完全相同的原始 672/192 Joint Reader，
以及 SHA256 固定的 P2 Quality Reader snapshot
`85d51371b8667eb7a2fd465cc85bfb3e28a6398285b78611e66f854dde289458`。
全程沒有恢復或改動原始 P2 訓練 Checkpoint。

Blender Front 與 Side Camera 對投影的最大差異分別
**0.000050px / 0.000059px**，Camera Audit `PASS`。
同步 Front/Side 最佳時間差 `0 frames`。
實際 `gt_video.mp4` 有 124 frames。
全部 92 個 Keypoints 分類完成（0 未分類）。

### 原始 Reader 真值 2D 誤差（px）

下表是 **已知 3D Blender world joints 經固定相機投影的 2D GT**
對原始 Reader 檢測的距離，不是 Solver Reprojection。
特別要區分幾何遮擋與仍能顯示投影位置的內部骨骼點；目前尚未
有 Mesh 自遮擋 Label。

| Seed / View | Head Neck | Torso Root | Forelimbs | Hindlimbs | Feet Toes | Tail |
|---|---:|---:|---:|---:|---:|---:|
| Seed 7 · Front | 1.691 | － | － | － | 0.851 | **10.026** |
| Seed 7 · Side | 1.249 | － | － | － | 1.459 | **0.730** |
| Seed 11 · Front | 1.539 | 1.460 | 0.921 | 1.164 | 0.933 | **7.577** |
| Seed 11 · Side | 1.257 | 0.765 | 1.211 | 1.014 | 1.562 | **0.839** |

兩組 Seed 的 2D 錯誤型態相同：Front Tail 大幅偏高，而
Side Tail 很低；Feet/Toes 的 Side 相對 Front 較弱。
Seed 11 全部 92 Joint 的 Front/Side Reader MAE 分別為
**1.736px / 1.276px**。Tail08 在 Frame 47 的 Front 2D 真值
誤差約 **71.55px**。

### P1 Guarded 2D 真值 A/B（Seed 11）

改用 Optical Flow + Guarded 之後，Front MAE
**1.736 → 1.939px**、Side **1.276 → 1.387px**。
Head/Neck、Torso/Root、Forelimbs、Hindlimbs、
Feet/Toes、Tail **全部六組**在兩個視角的平均 2D GT 誤差皆增加。
這是直接對 Blender 真值的證據，不是同一 Detection 的
Solver Loss，足以否決直接開啟目前 P1 Fusion。

### 三版本 3D 真值比較（Seed 11 · 124 frames）

下列 MAE 單位為 cm，全部投影/骨架/影格與 GT 相同。

| Body part | Joints | Baseline | P1 Guarded | P2 Weighted |
|---|---:|---:|---:|---:|
| All | 92 | **3.52860** | 4.11449 | 3.52536 |
| Head / Neck | 17 | 3.85501 | 4.38021 | **3.85140** |
| Torso / Root | 5 | 2.32611 | 3.11415 | **2.32223** |
| Forelimbs | 21 | 2.77206 | 3.17449 | **2.75711** |
| Hindlimbs | 6 | 2.51774 | 3.37383 | **2.51391** |
| Feet / Toes | 34 | 3.39332 | 4.04297 | **3.36220** |
| Tail | 9 | **6.53030** | 7.12566 | 6.66117 |

- 全身 P95 (cm)：Baseline **8.67428**、P1 **10.01748**、
  P2 **8.60468**。
- Root World Keypoint MAE (cm)：Baseline **2.10476**、
  P1 **3.27619**、P2 **2.10620**。
- 全身最大單點誤差 (cm)：Baseline **69.79396**、P1 **65.26477**、
  P2 **76.33210**。
- Tail08 在 Frame 47 仍是最嚴重單點：
  Baseline 69.79396cm、P2 **76.33210cm**。
- P1 全身平均相對 Baseline 約 **+16.6% 誤差**，與 Seed 7 的
  **+12.6%** 同方向。
- P2 全身 MAE 只比 Baseline 改善 **約 0.092%**；Seed 7 曾改善
  **1.299%**，故不能宣稱 P2 的整體效果已穩定。
  P2 Tail MAE 在 Seed 11 從 **6.53030 → 6.66117 cm（變差）**；
  Seed 7 則為 6.41011 → 6.04932cm（改善）。

### P2 誤差校準在 Seed 11 的已知 2D GT 上

P2 的 predicted error vs actual error Pearson：
Front **0.78505**、Side **0.53304**。
平均 error-calibration gap：Front **0.4074px**、
Side **0.68258px**。
Front Tail 的平均預測 5.821px、實際 7.577px；
Tail08 Frame 47 預測 **29.887px**，實際約 **71.547px**，
證明極端失敗案例仍被明顯低估。

在同一份 Raw Reader 座標、完全沒有重新訓練 Joint Reader 的
前提下，P2 只對 Front **391/11408**、Side **364/11408**
Joint-frames 乘上小於一的權重。這是啟發式公式
`clip(5/max(expected_error_px,5),0.4,1)` 的效果，並非
經真實自遮擋分布校準的統計模型。

### 目前工程結論與下一階段 Gate

1. **P1 Guarded 不得啟用為預設。** 兩組受控 Ground Truth 都呈現
   2D 與 3D 全身平均退步。先調查 LK 可信度、Joint Identity、
   常態小幅漂移與「接受／拒絕」融合，再以跨 Seed GT 驗證新 Fusion。
2. **P2 預測不確定性有用，但 Weighting 暫不升級正式 Solver。**
   平均改善不穩定，Frame 47 Tail08 極端失敗甚至變嚴重。
   應優先驗證特別嚴重的尾巴誤差、負向選擇風險，以及 Head/Neck／Toes
   的不確定性校準。
3. **關節專屬 View Reliability** 比直接讓全部 92 Joints 用相同
   視角權重更值得 A/B。可先試精準受控的
   Tail Front 降權／Side 保持權重，另測接近後腿的 Toe；
   但 Front 有獨特 X 軸資訊，Side 無法單獨補回，
   不應把 Front Tail 強制設零而宣稱能恢復 3D。
4. 若使用 Top／Elevated Third View，需先用 **同一段**
   Blender Ground Truth 動畫產生同步第三相機，才能量測
   Joint Specific Multi-view 的真實收益。
5. 合成動作無接地段／自然步態保證；真正的 Motion Prior、接地與
   自遮擋需後續 verified gait corpus。

所有真實 GT／Checkpoint／Dino Blender Mesh／CSV 留在使用者的
本機獨立 Clone，不會隨本文件上傳 GitHub。
