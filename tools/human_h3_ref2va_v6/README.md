# Human H3 Ref2VA V6 — 原始開發程式快照

本目錄保存 AnyMove Human V5→V6 FOURCAM 實驗已驗證過的 Python 腳本；它們是從原始工作區 F:\anymove\runs\human\lora_synth_A\conditioned_backend\fal_h3_ref2va 位元組一致複製，沒有覆寫原始檔案。

目標：讓 V6 四視角合成、資料 QA、fal Debug / Training、Held-out A/B 的程式碼進入 Git，而**不把大型模型權重與影片加入 Repo**。完整版技術紀錄請讀 docs/HUMAN_FOURCAM_H3_REF2VA_V6.md，SHA 快照請看 docs/human_h3_ref2va_v6/SNAPSHOT_MANIFEST.json。

## 流程

1. V5 Four-Camera camera geometry / rendering / QA：derive_fourview_v5_l45_r45_back_top.py 等 V5 腳本。
2. V6 Motion Diversity：prepare_reach_diverse_v6.py、render_reach_diverse_v6_source_batch.py 等。
3. V6 Four-View RGB / Pose 輸出：prepare_fourview_v6_reach_cameras.py、render_fourview_v6_reach_blender.py、build_fourview_v6_reach_media.py。
4. 靜態與影片 QA：qa_fourview_v6_reach_independent.py、qa_fourview_v6_reach_encoded_mp4.py。
5. 28 Clips 打包：merge_v5_v6_reach_augmented.py、finalize_reach_v6_expanded.py。
6. fal 預處理 Debug：run_v6_28_fourcam_1step_debug.py、audit_v6_28_fourcam_debug.py。
7. fal 400-Step：run_v6_28_fourcam_400step.py。
8. 四視角 A/B：stage_v6_fourcam_heldout_ab.py、run_v6_fourcam_*_ab.py、analyze_v6_fourcam_heldout_ab.py、v6_fourcam_pixel_iou_qa.py、summarize_v6_fourcam_ab.py。

## 限制與安全

- **工作站專用**：歷史腳本使用固定 Windows F:\anymove 絕對路徑與 runs/ 下的合成素材。僅複製程式碼不足以在乾淨環境重新訓練。
- 需已存在的 Blender Human rig、資料集、Python packages、ffmpeg，以及使用者自己設置的 fal API 憑證。
- 絕對不要在未另行確認下執行任何 submit 子命令：**這些會產生新付費 API 任務**。這次歷史訓練／推論已完成，不需要再提交。
- 既有 Request ID 在工作站的 run directory 內，可用 poll / collect 只取得原有輸出。
- 禁止將 .env、FAL_KEY、SafeTensors、原始影片及 ZIP 放進原始碼 Git commit。
- 評測是 RGB Color Mask IoU，不是 3D Rig Joint Error；結果不能作為 AnyMove Skeleton Solver 已完成的證據。

對應的靜態驗證單元測試：tests/test_human_ref2va_v6_snapshot.py。
