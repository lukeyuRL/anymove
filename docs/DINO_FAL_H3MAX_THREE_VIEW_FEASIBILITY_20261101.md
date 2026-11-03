# Dino H3 Max — 三視圖同步可行性實測（2026-11-01）

**結論：視覺排版通過，但三視圖 3D 幾何一致性沒有通過。**

目前不能將 H3 Max 單次 Image-to-Video 產生的 Front／Side／Top
影片當成真正同步的三台固定相機，直接投入 AnyMove 正式
Three-view 3D Solver。**不代表 H3 Max 所有其他控制方式都不可能成功。**

## 1. 實際使用的 fal API 和費用

- fal endpoint: `minimax/h3-max/image-to-video`（不是 reference-to-video endpoint）
- 一張包含 **同一隻 Dino 的 Front、Side、Top** 三視圖 Rest Pose 的參考圖
- Request: 5 秒、768P、`prompt_expansion_mode=disabled`、Seed 1 / Seed 2
- H3 Max 輸出：每支 **1344×768、124 frames、24 FPS、約 5.17 秒**
- 官方目前標示 768P 5 秒約 **US$0.40／支**；這一輪共 2 支，
  因此名目生成費約 **US$0.80**，不等於已核對實際帳單。
- 官方 API: https://fal.ai/models/minimax/h3-max/image-to-video/api
- 官方定價說明: https://fal.ai/learn/tools/how-to-generate-ai-videos-with-an-api

試驗沒有修改 `F:\anymove`、任何 Reader / P2 Checkpoint、原始
Front/Side Render 或 Motion JSON。使用獨立目錄
`F:\anymove_gt_eval_20261101\runs\dino\fal_3view_smoke_20261101`。

## 2. 單一三視圖參考畫面配置

原始 Dino Rig 中 92 個關節、67 Bones，Front/Side 與新渲染
Top 使用完全相同 Rest Pose（與 `blender/render_ref.py` 重設局部
Euler 的邏輯一致）。

輸入圖保持 16:9、1280×720：

| View | Panel X | Panel 寬度 | 原始圖大小 | 保持長寬比後之 render 高度 |
|---|---:|---:|---:|---:|
| Front | 0–319 | 320 | 400×720 | 576 |
| Side | 320–799 | 480 | 880×720 | 393 |
| Top | 800–1279 | 480 | 880×720 | 393 |

Top 是正俯視，世界座標 +Y（Dino 長軸）對應畫面 +U，
+X 對應 +V，沒有使用 45° 斜視。分隔線為固定深灰色。
Prompt 強制要求同時拍攝**一隻 Dino**、三台固定相機、
同一姿勢、同一動作、不可變換場景、不可縮放或移動相機。
兩次 Seed 使用完全同一張 reference 和尾巴左右擺動 Prompt。

## 3. 如何區分真正同步與「看起來同步」

初步視覺檢查：兩支影片都保留三個 panel、Dino 造型與彩色標記，
Front 和 Top 都看到有節奏的尾巴運動；**Visual Layout PASS**。

但要支援 3D Solver，理論上同一點應符合（單位為原始 Dino
Layout 的像素，而非 H3 輸出壓縮後的像素）：

```
X 誤差 = (front_u - W_FRONT/2) - (top_v - H/2)
Y 誤差 = (side_u - W_SIDE/2) + (top_u - W_TOP/2)
Z 誤差 = front_v - side_v
```

兩種**獨立**評估來源：

1. Dino 672／192 Joint Reader：各 Panel 經原參考圖
   `ref_3view_layout.json` 的縮放與原點還原、再進入同一
   Reader。只測 9 個尾巴 Keypoints，避免把未針對 Top
   訓練的全身 Reader 當成可靠的全身三視圖資訊。
2. Mesh 粉紅尾端 HSV 色塊追蹤：不依賴 Joint Reader，
   使用每個視角可見的粉紅色尾端區域質心作幾何及相位代理
   （並非同一個精確的 3D 表面點）。

**最重要的獨立控制：**使用已驗證的 Blender Seed 11、
124-frame 完全同步的 Front、Side、Top 影片，用**完全相同
1280×720 Panel 配置、縮放和 1344×768 MP4 編碼**重放，交給
完全相同的 Reader 與色塊量測。這個控制可以分離「縮小
Panel、Reader 本身與編碼造成的誤差」和「生成模型新增的
跨視角不一致」。

## 4. 完成的 Grounded A/B

以下全部是 124 frames、24 FPS 的觀測比較。相機投影差以像素
呈現，**沒有 AI 生成動作的真實 3D Ground Truth**，
所以不能冒充真實 3D MAE。

| 指標 | Blender 完全同步控制 | H3 Max Seed 1 | H3 Max Seed 2 |
|---|---:|---:|---:|
| 三視圖 panel 結構 | PASS | PASS | PASS |
| Reader 全尾巴 X 異視角差異中位數 | **1.83 px** | 19.66 px | 10.38 px |
| Reader 全尾巴 Y 異視角差異中位數 | **3.91 px** | 11.23 px | 9.83 px |
| Reader Front／Side Z 差異中位數 | **1.29 px** | 3.97 px | 2.24 px |
| Reader **Tail08 X** 差異中位數 | **2.90 px** | **74.16 px** | **33.43 px** |
| Reader Tail08 X 差異 P95 | **7.16 px** | 157.27 px | 122.89 px |
| 色塊尾端 X 跨視角差異中位數 | **8.84 px** | 42.98 px | 35.20 px |
| 色塊 Front／Top 最佳相關性位移 | **0 frame** | -4 frames | -3 frames |

補充：
- Reader Tail08 Front／Top 的最佳相關性位移：
  Blender **0**，Seed 1 **-2**，Seed 2 **-1** frame。
- Color 尾端位置不一定精確對應同一 3D 點；
  最佳相關性位移也不等於證實存在固定的時鐘延遲，
  可能反映了生成影片各 View 的動作振幅、形狀及相位不同。
- 第一影格原始 Rest Pose 的 Reader Tail 2D GT 中位／均值
  也可能受合成圖大小及 H3 的首影格重繪影響。
- 兩個互不依賴的量測方法都呈現明顯不一致；與控制組比較後，
  無法把這些偏差簡化為原本 Reader 不準或縮小視角造成。

**判斷：目前這個 H3 Max I2V Prompt + 三欄合成圖方案不滿足
AnyMove 嚴格的 3-view 3D 重建前提。** 不能將兩次生成的
Top View 直接接入既有 3D Solver 並宣稱改善。

## 5. 控制支出與下一階段 Gate

原先規劃 3 種動作 × 3 Seeds = 9 支；因為前兩支
相同動作、不同 Seed 都沒有通過幾何一致性，
**已在 2 支止損**，後續 7 支沒有提交，防止無效生成花費。

對 Next Step 的具體建議：

1. 暫停 Full 2,200 Top Reader Fine-tuning。若 fal 提供的
   視角本身不是相同 3D 動作，Reader 訓練不能補出被破壞的
   動作同步。
2. 先嘗試支援真正 3D-conditioned／multi-camera-conditioned
   生成或保持 Blender 同步結構的 Video-to-Video 改風格方案。
   必須以現有 Blender 同步三視圖 Ground Truth 影片作控制，
   不是三段獨立生成再後製對齊。
3. 任何新生成策略必須使用同一控制組計算
   **Tail08 X 殘差、全尾巴 X/Y/Z 殘差、最佳時間位移**
   並在多 Seed／多動作上評估。當各 View 長時間演變不同，
   不得啟用三視角正式 Pipeline。
4. 獨立研究雙視角 + Motion Prior 仍可繼續，不依賴目前
   失敗的 H3 三視圖生成條件。

## 6. 本機實際資料

全部資料只在使用者 Windows 電腦，Git repo 只提交工具與
定量結論，不上傳私人 Blender Rig、訓練權重或生成影片。

```text
runs/dino/fal_3view_smoke_20261101/
  reference/
    ref_top.png
    ref_3view_1280x720.png
    ref_3view_layout.json
    cam_top.json
    ref_top_pose.json
    preview_640x360.png
  prompts/
    smoke_tail.txt
    smoke_walk.txt                 # prepared, NOT generated
    smoke_roar.txt                 # prepared, NOT generated
  videos/
    smoke_tail_s1/
      h3max.mp4
      h3max.json
      contact_sheet_6frames.jpg
      color_tip_qa/threeview_audit.json
      joint_reader_qa/crossview_report.json
    smoke_tail_s2/
      h3max.mp4
      h3max.json
      contact_sheet_6frames.jpg
      color_tip_qa/threeview_audit.json
      joint_reader_qa/crossview_report.json
  blender_control_seed11/
    blender_threeview_control.mp4
    first_frame_3d_pose.json
    control_manifest.json
    color_tip_qa/threeview_audit.json
    joint_reader_qa/crossview_report.json
```

生成影片來源：
- Seed 1: https://v3b.fal.media/files/b/0ab0b0a7/RNP4r2T6DjfitefGvX_De_minimax-h3.mp4
- Seed 2: https://v3b.fal.media/files/b/0ab0b0ca/nScUDOKXakwYaxfsz57Pu_minimax-h3.mp4

重現工具（不重複呼叫 fal 時免費）：
`blender/render_dino_top_reference.py`,
`tools/build_dino_three_view_ref.py`,
`tools/audit_fal_three_view.py`,
`tools/audit_fal_three_view_reader.py`,
`tools/build_blender_three_view_fal_control.py`。
