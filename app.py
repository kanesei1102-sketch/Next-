import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re
import uuid
from skimage.feature import peak_local_max

# ---------------------------------------------------------
# 0. ページ設定と定数
# ---------------------------------------------------------
st.set_page_config(page_title="Bio-Image Quantifier Integrated", layout="wide")
SOFTWARE_VERSION = "Bio-Image Quantifier v2026.12 (Hybrid Edition)"

# セッション状態の初期化
if 'uploader_key_basic' not in st.session_state:
    st.session_state.uploader_key_basic = str(uuid.uuid4())
if 'uploader_key_pro' not in st.session_state:
    st.session_state.uploader_key_pro = str(uuid.uuid4())
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "current_analysis_id" not in st.session_state:
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
    unique_suffix = str(uuid.uuid4())[:8]
    st.session_state.current_analysis_id = f"AID-{date_str}-{unique_suffix}"

# ---------------------------------------------------------
# 1. 共通画像処理エンジン
# ---------------------------------------------------------
COLOR_MAP = {
    # 標準・蛍光用
    "茶色 (DAB染色)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "緑色 (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "赤色 (RFP)": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "青色 (DAPI)": {"lower": np.array([100, 50, 50]), "upper": np.array([140, 255, 255])},
    # Pro用追加定義 (HEなど)
    "ヘマトキシリン (Nuclei)": {"lower": np.array([100, 50, 50]), "upper": np.array([170, 255, 200])},
    "エオジン (Cytoplasm)": {"lower": np.array([140, 20, 100]), "upper": np.array([180, 255, 255])}
}

# マスク生成（コード1とコード2の互換性を維持した統合版）
def get_mask(hsv_img, color_name, sens, bright_min):
    # カラーマップにない場合はデフォルト(DAPI)を使用
    conf = COLOR_MAP.get(color_name, COLOR_MAP["青色 (DAPI)"])
    
    # 赤色やエオジンなどのHueが0/180をまたぐケースの処理
    if color_name == "赤色 (RFP)" or "エオジン" in color_name:
        lower1 = np.array([0, 30, bright_min]); upper1 = np.array([10 + sens//2, 255, 255])
        lower2 = np.array([170 - sens//2, 30, bright_min]); upper2 = np.array([180, 255, 255])
        return cv2.inRange(hsv_img, lower1, upper1) | cv2.inRange(hsv_img, lower2, upper2)
    else:
        # 通常の色
        l = np.clip(conf["lower"] - sens, 0, 255)
        u = np.clip(conf["upper"] + sens, 0, 255)
        l[2] = max(l[2], bright_min)
        return cv2.inRange(hsv_img, l, u)

def get_tissue_mask(hsv_img, color_name, sens, bright_min):
    mask = get_mask(hsv_img, color_name, sens, bright_min)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    valid_tissue = [c for c in cnts if cv2.contourArea(c) > 500]
    cv2.drawContours(mask_filled, valid_tissue, -1, 255, thickness=cv2.FILLED)
    return mask_filled

def get_centroids(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0: pts.append(np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]]))
    return pts

# Pro用: Adaptive Detection Engine
def perform_adaptive_detection(gray_img, block_size=25, c_val=2, min_dist=3):
    if block_size % 2 == 0: block_size += 1
    binary = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_val)
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    coords = peak_local_max(dist, min_distance=min_dist, labels=binary)
    return len(coords), coords, binary

# ---------------------------------------------------------
# 2. バリデーションデータの読み込み
# ---------------------------------------------------------
@st.cache_data
def load_validation_data():
    files = {'C14': 'quantified_data_20260102_201522.csv', 'C40': 'quantified_data_20260102_194322.csv',
             'C70': 'quantified_data_20260103_093427.csv', 'C100': 'quantified_data_20260102_202525.csv'}
    data_list = []; mapping = {'C14': 14, 'C40': 40, 'C70': 70, 'C100': 100}
    for density, filename in files.items():
        try:
            df = pd.read_csv(filename); col = 'Image_Name' if 'Image_Name' in df.columns else 'File Name'
            for _, row in df.iterrows():
                fname = str(row[col]); val = row['Value']
                channel = 'W1' if 'w1' in fname.lower() else 'W2' if 'w2' in fname.lower() else None
                if not channel: continue
                f_match = re.search(r'_F(\d+)_', fname)
                if f_match:
                    focus = int(f_match.group(1)); accuracy = (val / mapping[density]) * 100
                    data_list.append({'Density': density, 'Ground Truth': mapping[density], 'Focus': focus, 'Channel': channel, 'Value': val, 'Accuracy': accuracy})
        except FileNotFoundError: pass
    return pd.DataFrame(data_list)

df_val = load_validation_data()

# ---------------------------------------------------------
# 3. UI サイドバー設定
# ---------------------------------------------------------
st.title("🔬 Bio-Image Quantifier: Hybrid Edition")
st.sidebar.markdown(f"**ID:** `{st.session_state.current_analysis_id}`")

# タブ定義（ここで機能を分離）
tab_basic, tab_pro, tab_val = st.tabs(["🚀 標準解析 (Basic)", "🧪 高度解析 (Pro)", "🏆 精度検証"])

with st.sidebar:
    st.header("共通設定")
    # グループ化戦略（共通変数として定義）
    group_strategy = st.radio("ラベルの決定方法:", ["手動入力", "ファイル名から自動抽出"])
    if group_strategy == "手動入力":
        sample_group = st.text_input("グループ名 (X軸ラベル):", value="Control")
        filename_sep = None
    else:
        filename_sep = st.text_input("セパレーター (例: _ ):", value="_")
        sample_group = "(自動検出中)"
    
    scale_val = st.number_input("空間スケール (μm/px)", value=1.5267, format="%.4f")
    
    st.divider()
    
    # === タブ1（標準解析）用のパラメータ ===
    with st.expander("🔧 標準解析の設定 (タブ1用)", expanded=True):
        mode_raw = st.selectbox("解析モード (Basic):", [
            "1. 面積占有率 (%)", 
            "2. 核カウント / 密度解析", 
            "3. 共局在（Colocalization）解析", 
            "4. 空間距離解析", 
            "5. トレンド・比率解析"
        ])
        mode = mode_raw
        
        # 変数初期化（エラー回避用）
        target_a, sens_a, bright_a = "青色 (DAPI)", 20, 60
        target_b, sens_b, bright_b = "緑色 (GFP)", 20, 60
        bright_count, min_size = 50, 50
        use_roi_norm, roi_color, sens_roi, bright_roi = False, "赤色 (RFP)", 20, 40
        trend_metric, ratio_val, ratio_unit = "面積占有率", 0, "%"
        sens_common, bright_common = 20, 60

        # モード別UI
        if mode.startswith("5."):
            trend_metric = st.radio("指標ターゲット:", ["共局在率", "面積占有率"])
            ratio_val = st.number_input("条件値:", value=0, step=10)
            ratio_unit = st.text_input("単位:", value="%", key="unit_basic")
            if group_strategy == "手動入力": sample_group = f"{ratio_val}{ratio_unit}"
            if trend_metric.startswith("共局在"):
                target_a = st.selectbox("CH-A (基準):", list(COLOR_MAP.keys()), index=3, key="t5_a")
                sens_a = st.slider("A 感度", 5, 50, 20, key="s5_a"); bright_a = st.slider("A 輝度", 0, 255, 60, key="b5_a")
                target_b = st.selectbox("CH-B (対象):", list(COLOR_MAP.keys()), index=2, key="t5_b")
                sens_b = st.slider("B 感度", 5, 50, 20, key="s5_b"); bright_b = st.slider("B 輝度", 0, 255, 60, key="b5_b")
            else:
                target_a = st.selectbox("解析カラー:", list(COLOR_MAP.keys()), index=2, key="t5_single")
                sens_a = st.slider("感度", 5, 50, 20, key="s5_single"); bright_a = st.slider("輝度", 0, 255, 60, key="b5_single")
        elif mode.startswith("1."):
            target_a = st.selectbox("解析カラー:", list(COLOR_MAP.keys()), key="t1"); sens_a = st.slider("感度", 5, 50, 20, key="s1"); bright_a = st.slider("輝度", 0, 255, 60, key="b1")
        elif mode.startswith("2."):
            min_size = st.slider("最小核サイズ (px)", 10, 500, 50, key="m2_size"); bright_count = st.slider("検出閾値", 0, 255, 50, key="m2_th")
            use_roi_norm = st.checkbox("ROIで正規化", value=True, key="m2_roi")
            if use_roi_norm:
                roi_color = st.selectbox("組織カラー:", list(COLOR_MAP.keys()), index=2, key="m2_roicol"); sens_roi = st.slider("ROI感度", 5, 50, 20, key="m2_roisens"); bright_roi = st.slider("ROI輝度", 0, 255, 40, key="m2_roibright")
        elif mode.startswith("3."):
            target_a = st.selectbox("CH-A:", list(COLOR_MAP.keys()), index=3, key="t3_a"); sens_a = st.slider("A 感度", 5, 50, 20, key="s3_a"); bright_a = st.slider("A 輝度", 0, 255, 60, key="b3_a")
            target_b = st.selectbox("CH-B:", list(COLOR_MAP.keys()), index=2, key="t3_b"); sens_b = st.slider("B 感度", 5, 50, 20, key="s3_b"); bright_b = st.slider("B 輝度", 0, 255, 60, key="b3_b")
        elif mode.startswith("4."):
            target_a = st.selectbox("起点 A:", list(COLOR_MAP.keys()), index=2, key="t4_a"); target_b = st.selectbox("終点 B:", list(COLOR_MAP.keys()), index=3, key="t4_b")
            sens_common = st.slider("共通感度", 5, 50, 20, key="s4"); bright_common = st.slider("共通輝度", 0, 255, 60, key="b4")

    # === タブ2（高度解析）用のパラメータ ===
    with st.expander("🧪 高度解析の設定 (タブ2用)", expanded=False):
        mode_pro = st.selectbox("解析モード (Pro):", ["2. 核カウント (Adaptive)", "1. 面積占有率"], key="mode_pro")
        img_type_pro = st.radio("画像タイプ:", ["蛍光 (Fluorescence)", "明視野 (Brightfield/HE)"], key="type_pro")
        
        pro_params = {}
        if mode_pro.startswith("2."):
            if img_type_pro.startswith("蛍光"):
                st.caption("BBBC005推奨設定")
                pro_block = st.slider("ブロックサイズ", 3, 51, 25, step=2, key="p_blk")
                pro_c = st.slider("C値 (感度)", -10, 20, 2, key="p_c")
                pro_dist = st.slider("最小距離 (px)", 1, 20, 3, key="p_dist")
                pro_params = {"block": pro_block, "c": pro_c, "dist": pro_dist}
            else:
                pro_target = st.selectbox("核の色:", list(COLOR_MAP.keys()), index=4, key="p_nuc")
                pro_sens = st.slider("感度", 5, 50, 15, key="p_ns")
                pro_bright = st.slider("輝度", 0, 255, 50, key="p_nb")
                pro_params = {"target": pro_target, "sens": pro_sens, "bright": pro_bright}
            
            pro_roi_norm = st.checkbox("ROI正規化", value=False, key="p_roi")
            if pro_roi_norm:
                pro_roi_col = st.selectbox("ROI色:", list(COLOR_MAP.keys()), index=5, key="p_rc")
                pro_roi_sens = st.slider("ROI感度", 5, 50, 20, key="p_rs")
                pro_roi_bright = st.slider("ROI輝度", 0, 255, 30, key="p_rb")
                pro_params.update({"roi_col": pro_roi_col, "roi_sens": pro_roi_sens, "roi_bright": pro_roi_bright})
            
            # サイズフィルタ
            d_min, d_max = st.slider("核サイズ範囲 (μm)", 0.0, 50.0, (5.0, 20.0), key="p_dia")
            pro_params.update({"d_min": d_min, "d_max": d_max})
        
        elif mode_pro.startswith("1."):
            pro_target = st.selectbox("対象色:", list(COLOR_MAP.keys()), index=2, key="p_t1")
            pro_sens = st.slider("感度", 5, 50, 20, key="p_s1")
            pro_bright = st.slider("輝度", 0, 255, 60, key="p_b1")
            pro_params = {"target": pro_target, "sens": pro_sens, "bright": pro_bright}

    st.divider()
    if st.button("履歴クリア & 新規ID"):
        st.session_state.analysis_history = []
        st.session_state.uploader_key_basic = str(uuid.uuid4())
        st.session_state.uploader_key_pro = str(uuid.uuid4())
        st.rerun()
        
    st.download_button("📥 履歴CSV保存", pd.DataFrame(st.session_state.analysis_history).to_csv(index=False).encode('utf-8'), "data.csv")


# ---------------------------------------------------------
# 4. タブ1: 標準解析 (Code 1のロジックを維持)
# ---------------------------------------------------------
with tab_basic:
    # ユーザー様指定のコードブロック（アップローダーのキーのみ独立させています）
    uploaded_files = st.file_uploader("画像をアップロード (基本モード)", type=["jpg", "png", "tif", "tiff"], accept_multiple_files=True, key=st.session_state.uploader_key_basic)
    if uploaded_files:
        st.success(f"{len(uploaded_files)} 枚の画像を解析中...")
        batch_results = []
        for i, file in enumerate(uploaded_files):
            file.seek(0); file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            if img_raw is not None:
                # --- 自動グループ抽出ロジック ---
                if group_strategy.startswith("ファイル名"):
                    try:
                        detected_group = file.name.split(filename_sep)[0]
                    except:
                        detected_group = "不明"
                    current_group_label = detected_group
                else:
                    current_group_label = sample_group

                # 画像処理 (Code 1 Original Logic)
                img_f = img_raw.astype(np.float32); mn, mx = np.min(img_f), np.max(img_f)
                img_8 = ((img_f - mn) / (mx - mn) * 255.0 if mx > mn else np.clip(img_f, 0, 255)).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR) if len(img_8.shape) == 2 else img_8[:,:,:3]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB); img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                val, unit, res_disp = 0.0, "", img_rgb.copy()
                h, w = img_rgb.shape[:2]; fov_mm2 = (h * w) * ((scale_val / 1000) ** 2)

                extra_data = {}

                if mode.startswith("1.") or (mode.startswith("5.") and trend_metric.startswith("面積")):
                    mask = get_mask(img_hsv, target_a, sens_a, bright_a); val = (cv2.countNonZero(mask) / (h * w)) * 100
                    unit = "% Area"; res_disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB); res_disp[:,:,0]=0; res_disp[:,:,2]=0

                elif mode.startswith("2."):
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY); _, th = cv2.threshold(gray, bright_count, 255, cv2.THRESH_BINARY)
                    blur = cv2.GaussianBlur(gray, (5,5), 0); _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    cnts, _ = cv2.findContours(cv2.bitwise_and(th, otsu), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid = [c for c in cnts if cv2.contourArea(c) > min_size]; val, unit = len(valid), "cells"
                    cv2.drawContours(res_disp, valid, -1, (0,255,0), 2)
                    
                    a_target_mm2 = fov_mm2 
                    roi_status = "視野全体"
                    
                    if use_roi_norm:
                        mask_roi = get_tissue_mask(img_hsv, roi_color, sens_roi, bright_roi)
                        roi_px = cv2.countNonZero(mask_roi)
                        a_target_mm2 = roi_px * ((scale_val/1000)**2) 
                        roi_status = "ROI内"
                        cv2.drawContours(res_disp, cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255,0,0), 3)

                    density = val / a_target_mm2 if a_target_mm2 > 0 else 0
                    extra_data = {
                        "対象面積 (mm2)": round(a_target_mm2, 6),
                        "密度 (cells/mm2)": round(density, 2),
                        "正規化基準": roi_status
                    }

                elif mode.startswith("3.") or (mode.startswith("5.") and trend_metric.startswith("共局在")):
                    mask_a = get_mask(img_hsv, target_a, sens_a, bright_a); mask_b = get_mask(img_hsv, target_b, sens_b, bright_b)
                    coloc = cv2.bitwise_and(mask_a, mask_b); denom = cv2.countNonZero(mask_a)
                    val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0; unit = "% Coloc"; res_disp = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])
                
                elif mode.startswith("4."):
                    ma, mb = get_mask(img_hsv, target_a, sens_common, bright_common), get_mask(img_hsv, target_b, sens_common, bright_common)
                    pa, pb = get_centroids(ma), get_centroids(mb)
                    if pa and pb: val = np.mean([np.min([np.linalg.norm(a - b) for b in pb]) for a in pa]) * (scale_val if scale_val > 0 else 1)
                    unit = "μm 距離" if scale_val > 0 else "px 距離"; res_disp = cv2.addWeighted(img_rgb, 0.6, cv2.merge([ma, mb, np.zeros_like(ma)]), 0.4, 0)

                st.divider()
                st.markdown(f"### 📷 画像 {i+1}: {file.name}")
                st.markdown(f"**検出グループ:** `{current_group_label}`")
                
                if mode.startswith("2.") and "密度 (cells/mm2)" in extra_data:
                    c_m1, c_m2, c_m3 = st.columns(3)
                    c_m1.metric("カウント数", f"{int(val)} cells")
                    c_m2.metric("密度", f"{int(extra_data['密度 (cells/mm2)']):,} /mm²")
                    c_m3.caption(f"面積: {extra_data['対象面積 (mm2)']:.4f} mm² ({extra_data['正規化基準']})")
                else:
                    st.markdown(f"### 解析結果: **{val:.2f} {unit}**")
                
                c1, c2 = st.columns(2); c1.image(img_rgb, caption="元画像"); c2.image(res_disp, caption="解析結果（マスクオーバーレイ）")
                
                row_data = {
                    "ファイル名": file.name, "グループ": current_group_label,
                    "数値": val, "単位": unit, "Mode": "Basic",
                    "解析日時": datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                }
                if extra_data: row_data.update(extra_data)
                batch_results.append(row_data)
        
        if st.button("このバッチの結果を確定する (Basic)", type="primary"):
            st.session_state.analysis_history.extend(batch_results)
            st.success("履歴に保存されました。")
            st.rerun()

# ---------------------------------------------------------
# 5. タブ2: 高度解析 (Code 2の機能)
# ---------------------------------------------------------
with tab_pro:
    uploaded_pro = st.file_uploader("画像をアップロード (Pro: 16-bit Auto-Scale / Adaptive)", type=["jpg", "png", "tif", "tiff"], accept_multiple_files=True, key=st.session_state.uploader_key_pro)
    
    if uploaded_pro:
        st.info("Proエンジンで解析中 (Adaptive Watershed / Auto Contrast)...")
        batch_pro = []
        for i, file in enumerate(uploaded_pro):
            file.seek(0); file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            
            # --- Pro Engine Image Loading (Code 2) ---
            img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            img_bgr = None
            if img_raw is not None:
                # 16bit / Low Contrast Auto Scaling
                is_low = (img_raw.max() < 150); is_16 = (img_raw.dtype == np.uint16) or (img_raw.max() > 255)
                if is_16 or is_low:
                    p_min, p_max = np.percentile(img_raw, (0.5, 99.5))
                    if p_max <= p_min: p_max = np.max(img_raw)
                    scale = 255.0 / (p_max - p_min) if (p_max - p_min) > 0 else 1.0
                    img_8 = np.clip((img_raw.astype(np.float32) - p_min) * scale, 0, 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR) if len(img_8.shape)==2 else img_8
                else:
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # グループ名
            if group_strategy.startswith("ファイル名"):
                try: grp = file.name.split(filename_sep)[0]
                except: grp = "Unknown"
            else: grp = sample_group

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            res_disp = img_rgb.copy()
            h, w = img_rgb.shape[:2]
            
            val, unit = 0, ""
            extra_data = {}

            # --- Pro Logic ---
            if mode_pro.startswith("2."): # Count
                roi_area_mm2 = (h * w) * ((scale_val/1000)**2)
                roi_stat = "FoV"
                coords = []

                if img_type_pro.startswith("蛍光"): # Adaptive
                    val, coords, _ = perform_adaptive_detection(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 
                                                              pro_params["block"], pro_params["c"], pro_params["dist"])
                    for p in coords: cv2.circle(res_disp, (p[1], p[0]), 3, (0,255,0), -1)
                else: # Brightfield (HSV)
                    mask = get_mask(img_hsv, pro_params["target"], pro_params["sens"], pro_params["bright"])
                    cnts, _ = cv2.findContours(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    min_px = (np.pi*((pro_params["d_min"]/2)**2))/(scale_val**2)
                    max_px = (np.pi*((pro_params["d_max"]/2)**2))/(scale_val**2)
                    valid = [c for c in cnts if min_px < cv2.contourArea(c) < max_px]
                    val = len(valid)
                    cv2.drawContours(res_disp, valid, -1, (0,255,0), 2)
                
                # ROI処理
                if pro_params.get("roi_col"):
                    mask_roi = get_tissue_mask(img_hsv, pro_params["roi_col"], pro_params["roi_sens"], pro_params["roi_bright"])
                    roi_area_mm2 = cv2.countNonZero(mask_roi) * ((scale_val/1000)**2)
                    roi_stat = "ROI"
                    cv2.drawContours(res_disp, cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255,0,0), 3)
                    # 蛍光の場合はROI外の点を除外
                    if img_type_pro.startswith("蛍光"):
                        coords = [p for p in coords if mask_roi[p[0], p[1]] > 0]
                        val = len(coords)

                unit = "cells"
                dens = val / roi_area_mm2 if roi_area_mm2 > 0 else 0
                extra_data = {"密度": round(dens, 2), "面積mm2": round(roi_area_mm2, 4), "正規化": roi_stat}

            elif mode_pro.startswith("1."): # Area
                mask = get_mask(img_hsv, pro_params["target"], pro_params["sens"], pro_params["bright"])
                val = (cv2.countNonZero(mask)/(h*w))*100
                unit = "% Area"
                res_disp[mask>0] = (0,255,0)
            
            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"**{file.name}**")
                st.metric(f"結果 ({unit})", f"{val:.2f}")
                if extra_data: st.write(extra_data)
            with c2:
                st.image(res_disp, caption="Pro解析結果")
            
            row = {"ファイル名": file.name, "グループ": grp, "数値": val, "単位": unit, "Mode": "Pro"}
            row.update(extra_data)
            batch_pro.append(row)

        if st.button("このバッチの結果を確定する (Pro)", type="primary"):
            st.session_state.analysis_history.extend(batch_pro)
            st.success("履歴保存完了")
            st.rerun()

# ---------------------------------------------------------
# 6. タブ3: バリデーション (Code 1のBBBC005検証を維持)
# ---------------------------------------------------------
with tab_val:
    st.header("🏆 精度検証サマリー")
    st.markdown("""
    * **ベンチマーク:** BBBC005 (Broad Bioimage Benchmark Collection)
    * **解析規模:** 3,200枚
    * **検証手法:** 密度グループごとのパラメータ個別最適化
    """)

    if not df_val.empty:
        df_hq = df_val[(df_val['Focus'] >= 1) & (df_val['Focus'] <= 5)]
        w1_hq = df_hq[df_hq['Channel'] == 'W1']
        avg_acc = w1_hq['Accuracy'].mean()
        df_lin = w1_hq.groupby('Ground Truth')['Value'].mean().reset_index()
        r2 = np.corrcoef(df_lin['Ground Truth'], df_lin['Value'])[0, 1]**2

        m1, m2, m3 = st.columns(3)
        m1.metric("平均精度", f"{avg_acc:.1f}%")
        m2.metric("直線性 (R²)", f"{r2:.4f}")
        m3.metric("解析済み画像数", "3,200+")

        st.divider()
        st.subheader("📈 1. 計数性能と直線性 (W1 vs W2)")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot([0, 110], [0, 110], 'k--', alpha=0.3, label='理論値')
        ax1.scatter(df_lin['Ground Truth'], df_lin['Value'], color='#1f77b4', s=100, label='W1 (核)')
        w2_lin = df_hq[df_hq['Channel'] == 'W2'].groupby('Ground Truth')['Value'].mean().reset_index()
        ax1.scatter(w2_lin['Ground Truth'], w2_lin['Value'], color='#ff7f0e', s=100, marker='D', label='W2 (細胞質)')
        ax1.set_xlabel('理論値'); ax1.set_ylabel('解析値'); ax1.legend(); ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 2. 密度別精度比較")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            df_bar = df_hq.groupby(['Density', 'Channel'])['Accuracy'].mean().reset_index()
            sns.barplot(data=df_bar, x='Density', y='Accuracy', hue='Channel', ax=ax2)
            ax2.axhline(100, color='red', linestyle='--'); st.pyplot(fig2)
        with c2:
            st.subheader("📉 3. 光学的堅牢性")
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            df_decay = df_val[df_val['Channel'] == 'W1'].copy()
            sns.lineplot(data=df_decay, x='Focus', y='Accuracy', hue='Density', marker='o', ax=ax3)
            ax3.axhline(100, color='red', linestyle='--'); st.pyplot(fig3)
    else:
        st.error("バリデーションデータが見つかりません。")
