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
