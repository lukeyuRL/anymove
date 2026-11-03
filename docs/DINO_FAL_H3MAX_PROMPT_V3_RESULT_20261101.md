# H3 Max 三視圖：Prompt v3 單支授權測試（2026-11-01）

## 實驗限制與來源

使用者**明確批准只有一支**新的 H3 Max Image-to-Video 影片，使用與原本 Seed 1 相同的
Front／Side／Top 並排參考圖、相同 Seed 1、5 秒、768P，僅修改 Prompt 為
`prompt_tail_sway_v3_short.txt`（943 字元）。

本機只呼叫 fal 一次，已完成 1344×768、124 frames、24 FPS 生成，並保存於獨立測試目錄。
沒有自動生成 Seed 2、Walk 或 Roar，也沒有修改原始
`F:\anymove`、Reader 權重、Rig、P2 Checkpoints。

**生成實際媒體與原始結果留在使用者自己的 Windows 電腦**：
`F:\anymove_gt_eval_20261101\runs\dino\fal_3view_smoke_20261101\videos\prompt_short_v3_seed1\`。

其中保留 `h3max.mp4`、`h3max.json`、
`contact_sheet_6frames.jpg`、
`color_tip_qa/threeview_audit.json`、
`joint_reader_qa/crossview_report.json`、`joint_reader_qa/reader_threeview.npz`。
其他實驗資料、Prompt SHA 和跨 Seed 比較在：
`runs\dino\fal_3view_prompt_v3_preflight_20261101\prompt_v3_generated_single_test_result.json`。

## 真正 Grounded 比較

同一張參考圖、同一個 Seed，只有 Prompt 改變。
數值來自實際影片經 92-Joint Reader 的 cross-view 2D 投影對照，
不是 AI 影片的真實 3D GT。

| 檢查 | Blender 真同步控制 | 原 Prompt Seed1 | Prompt v3 Seed1 |
|---|---:|---:|---:|
| 全尾巴 Front／Top X 差異中位數 | 1.83 px | 19.66 px | **11.54 px** |
| 全尾巴 Front／Top X 差異 P95 | 10.66 px | **99.04 px** | 111.14 px |
| 全尾巴 Side／Top Y 差異中位數 | 3.91 px | 11.23 px | **6.73 px** |
| Front／Side Z 差異中位數 | 1.29 px | **3.97 px** | 8.50 px |
| Tail08 X 差異中位數 | 2.90 px | 74.16 px | 69.03 px |
| Tail08 X 差異 P95 | 7.16 px | **157.27 px** | 184.16 px |
| 色塊 Front／Top X 零延遲相關係數 | 0.908 | **0.902** | 0.030 |
| Front 粉紅尾端位移 P95−P5 | 177.70 px | 254.35 px | **50.67 px** |
| Top 粉紅尾端對應位移 P95−P5 | 163.36 px | 249.23 px | 128.88 px |

**判斷：Prompt v3 沒有通過同步驗收。**
雖然全尾巴 X/Y 中位數改善，但極端 X 誤差增加、Front／Side
Z 差異增加，且 Front 尾巴的可見移動幅度大幅縮小，
Top 仍有明顯的尾巴運動。色塊 Front／Top 同步相關性
幾乎消失，並非有意義的三視角 Motion 同步改善。

不能僅憑平均誤差變小選擇模型：這可能是縮小動作幅度
的假改善，對 AnyMove 重建完整 3D Motion 沒有幫助。

### 保留的限制

- fal 生成的動作**沒有獨立的 3D Ground Truth**。
- Joint Reader 的錯誤、外觀重新繪製、表面遮擋、相機視角漂移均可能
  貢獻跨視角差異，不宜把每個像素差都等同於 AI 模型的 3D 座標錯誤。
- 粉紅尾端是表面顏色區域質心，並不一定等於同一個 3D 骨骼端點；
  其時序相關係數只能當獨立診斷訊號。
- Blender 完全同步控制組使用**相同三欄尺寸、縮放、MP4 輸出尺寸與 Reader**，
  能有效觀察非模型因素，但不能保證所有生成物的外觀領域移轉相同。

### 後續 Gate

不自動再生成其他影片。若使用者另行批准下一支，
可以固定原 Prompt 與 Seed，僅測 Reference Layout，
或直接考慮另一個 H3 圖像結構／相機提示方案。
每次均應保留清楚動作幅度、跨視角 X/Y/Z、Tail08 P95
與相位一致性做完整 A/B；所有 fal 生成都必須在提交前
得到使用者明確確認。
