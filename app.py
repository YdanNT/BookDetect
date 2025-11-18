# # -*- coding: utf-8 -*-
# """
# ================================================================================
# 📚 BOOK DETECTOR - Application Streamlit
# ================================================================================
# Détection automatique de livres sur étagère avec OCR intelligent.
# """

# import os

# # ================== FIX PYTORCH 2.6+ ==================
# # Correction pour le chargement des modèles YOLO avec PyTorch 2.6+
# try:
#     import torch
#     # Monkey patch pour forcer weights_only=False
#     _original_torch_load = torch.load
    
#     def _patched_torch_load(*args, **kwargs):
#         if 'weights_only' not in kwargs:
#             kwargs['weights_only'] = False
#         return _original_torch_load(*args, **kwargs)
    
#     torch.load = _patched_torch_load
# except Exception:
#     pass

# import streamlit as st
# import math
# import json
# import base64
# import time
# from typing import List, Dict, Any, Optional
# import numpy as np
# import cv2
# import pandas as pd
# import supervision as sv
# from ultralytics import YOLO
# import httpx
# import certifi
# from openai import OpenAI


# # ================== CONFIGURATION API ==================
# # Clé API Scaleway configurée directement (ne pas partager publiquement)
# os.environ["SCW_SECRET_KEY"] = "ae1d12c9-9fd6-434b-aeba-286e9b9702fa"
# SCW_API_KEY = "ae1d12c9-9fd6-434b-aeba-286e9b9702fa"
# SCW_BASE_URL = "https://api.scaleway.ai/v1"

# # Chemin du modèle YOLO
# MODEL_PATH = "models/best.pt"


# # ================== CONFIGURATION PAGE ==================
# st.set_page_config(
#     page_title="📚 Book Detector",
#     page_icon="📚",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # CSS Custom
# st.markdown("""
# <style>
#     .stApp {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     }
#     h1 {
#         color: white;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
#     }
# </style>
# """, unsafe_allow_html=True)


# # ================== CACHE MODÈLE ==================
# @st.cache_resource
# def load_yolo_model(model_path: str):
#     """Charge le modèle YOLO (mis en cache)."""
#     try:
#         if not os.path.exists(model_path):
#             return None, f"Fichier modèle introuvable : {model_path}"
        
#         # Configuration pour PyTorch 2.6+ : autoriser le chargement de modèles personnalisés
#         import torch
        
#         # Ajouter les classes Ultralytics aux globaux sûrs
#         try:
#             from ultralytics.nn.tasks import OBBModel, DetectionModel
#             torch.serialization.add_safe_globals([OBBModel, DetectionModel])
#         except:
#             pass
        
#         # Charger le modèle avec weights_only=False pour les modèles personnalisés
#         model = YOLO(model_path)
#         return model, None
#     except Exception as e:
#         return None, str(e)


# # ================== FONCTIONS GÉOMÉTRIE ==================
# def order_quad_points(pts: np.ndarray) -> np.ndarray:
#     """Ordonne 4 points en [TL, TR, BR, BL]."""
#     pts = np.asarray(pts, dtype=np.float32)
#     s = pts.sum(axis=1)
#     d = np.diff(pts, axis=1).ravel()
    
#     tl = pts[np.argmin(s)]
#     br = pts[np.argmax(s)]
#     tr = pts[np.argmin(d)]
#     bl = pts[np.argmax(d)]
    
#     quad = np.array([tl, tr, br, bl], dtype=np.float32)
    
#     v1 = quad[1] - quad[0]
#     v2 = quad[2] - quad[1]
#     if np.cross(v1, v2) < 0:
#         quad = np.array([tl, bl, br, tr], dtype=np.float32)
    
#     return quad


# def obb_to_corners(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
#     """Convertit OBB en 4 coins."""
#     angle_rad = math.radians(angle_deg)
#     cos_a = math.cos(angle_rad)
#     sin_a = math.sin(angle_rad)
    
#     dx, dy = w / 2.0, h / 2.0
    
#     local_corners = np.array([
#         [-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]
#     ], dtype=np.float32)
    
#     rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
#     rotated = local_corners @ rotation_matrix.T
#     rotated[:, 0] += cx
#     rotated[:, 1] += cy
    
#     return order_quad_points(rotated)


# def extract_crop(image_bgr: np.ndarray, quad: np.ndarray, pad_ratio: float = 0.03, min_side: int = 20):
#     """Extrait un crop depuis un quadrilatère."""
#     tl, tr, br, bl = quad
    
#     width_top = np.linalg.norm(tr - tl)
#     width_bottom = np.linalg.norm(br - bl)
#     out_width = int(max(1, round(max(width_top, width_bottom))))
    
#     height_right = np.linalg.norm(tr - br)
#     height_left = np.linalg.norm(tl - bl)
#     out_height = int(max(1, round(max(height_right, height_left))))
    
#     if out_width < min_side or out_height < min_side:
#         return None
    
#     pad_w = int(round(out_width * pad_ratio))
#     pad_h = int(round(out_height * pad_ratio))
#     out_width_padded = out_width + 2 * pad_w
#     out_height_padded = out_height + 2 * pad_h
    
#     dst_points = np.array([
#         [pad_w, pad_h],
#         [pad_w + out_width - 1, pad_h],
#         [pad_w + out_width - 1, pad_h + out_height - 1],
#         [pad_w, pad_h + out_height - 1]
#     ], dtype=np.float32)
    
#     M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst_points)
    
#     crop = cv2.warpPerspective(
#         image_bgr, M,
#         (out_width_padded, out_height_padded),
#         flags=cv2.INTER_CUBIC,
#         borderMode=cv2.BORDER_CONSTANT,
#         borderValue=(255, 255, 255)
#     )
    
#     return crop


# # ================== OCR ==================
# def create_ocr_client(api_key: str, base_url: str):
#     """Crée client OCR Scaleway."""
#     if not api_key:
#         return None
    
#     try:
#         http_client = httpx.Client(
#             timeout=60.0,
#             verify=certifi.where(),
#             http2=False,
#             limits=httpx.Limits(max_connections=5)
#         )
        
#         return OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
#     except Exception as e:
#         st.error(f"Erreur création client OCR : {e}")
#         return None


# def encode_crop_to_base64(crop: np.ndarray, quality: int = 85) -> str:
#     """Encode crop en base64."""
#     if len(crop.shape) == 3:
#         crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
#     if len(crop.shape) == 2:
#         crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    
#     h, w = crop.shape[:2]
#     if max(h, w) > 1000:
#         scale = 1000 / max(h, w)
#         crop = cv2.resize(crop, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    
#     encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality, int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]
#     success, buffer = cv2.imencode(".jpg", crop, encode_params)
    
#     if not success:
#         raise RuntimeError("Échec encodage JPEG")
    
#     b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
#     return f"data:image/jpeg;base64,{b64}"


# PROMPT_OCR = """Tu es un expert en OCR de tranches de livres français.

# CONTEXTE: Chaque image montre UNE tranche de livre (texte vertical ou horizontal).

# INSTRUCTIONS: Pour CHAQUE image, extrais:
# 1. **lines** (array): TOUTES les lignes visibles, de HAUT en BAS
# 2. **title** (string): Le titre principal
# 3. **author** (string): L'auteur complet (Prénom NOM)
# 4. **publisher** (string): L'éditeur si visible

# EXEMPLES:

# Exemple 1: "LA TABLE | DES LOUPS | Adam Rapp | RIVAGES"
# → {{"lines": ["LA TABLE", "DES LOUPS", "Adam Rapp", "RIVAGES"], "title": "La Table des Loups", "author": "Adam Rapp", "publisher": "RIVAGES"}}

# Exemple 2: Image illisible
# → {{"lines": [], "title": "", "author": "", "publisher": ""}}

# RÈGLES: N'invente RIEN. Garde les accents. Si incertain → ""

# FORMAT: {{"results": [{{"lines": [...], "title": "...", "author": "...", "publisher": "..."}}, ...]}}

# Traite les {num_images} images."""


# def process_ocr_batch(crops: List[np.ndarray], client, max_tokens: int = 1800, max_retries: int = 4):
#     """Traite un batch OCR."""
#     num_crops = len(crops)
    
#     content = [{"type": "text", "text": PROMPT_OCR.format(num_images=num_crops)}]
    
#     for crop in crops:
#         content.append({"type": "image_url", "image_url": {"url": encode_crop_to_base64(crop)}})
    
#     for attempt in range(max_retries + 1):
#         try:
#             response = client.chat.completions.create(
#                 model="pixtral-12b-2409",
#                 response_format={"type": "json_object"},
#                 temperature=0.0,
#                 max_tokens=max_tokens,
#                 messages=[{"role": "user", "content": content}]
#             )
            
#             data = json.loads(response.choices[0].message.content or "{}")
#             results = data.get("results", [])
            
#             normalized = []
#             for item in results:
#                 normalized.append({
#                     "title": (item.get("title") or "").strip(),
#                     "author": (item.get("author") or "").strip(),
#                     "publisher": (item.get("publisher") or "").strip(),
#                     "lines": [l.strip() for l in item.get("lines", []) if l]
#                 })
            
#             while len(normalized) < num_crops:
#                 normalized.append({"title": "", "author": "", "publisher": "", "lines": []})
            
#             return normalized[:num_crops]
        
#         except Exception as e:
#             if "429" in str(e).lower() and attempt < max_retries:
#                 time.sleep(2.0 * (2 ** attempt))
#                 continue
#             st.warning(f"Erreur OCR (tentative {attempt+1}/{max_retries+1}) : {str(e)}")
#             break
    
#     return [{"title": "", "author": "", "publisher": "", "lines": []} for _ in crops]


# # ================== INTERFACE PRINCIPALE ==================
# def main():
#     """Interface principale Streamlit."""
    
#     # Header
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         st.title("📚 Book Detector")
#         st.markdown("*Détection automatique de livres assistée par IA*")
    
#     # Sidebar
#     with st.sidebar:
#         st.header("⚙️ Configuration")
        
#         st.subheader("🎯 Détection YOLO")
#         st.info(f"📁 Modèle : `{MODEL_PATH}`")
        
#         # Vérifier l'existence du modèle
#         if not os.path.exists(MODEL_PATH):
#             st.error(f"❌ Fichier modèle introuvable !\n\nChemin attendu : `{os.path.abspath(MODEL_PATH)}`")
#             st.info("💡 Assurez-vous que le fichier `best.pt` est dans le même dossier que ce script.")
        
#         conf_threshold = st.slider("Seuil de confiance", 0.1, 0.95, 0.50, 0.05)
#         iou_threshold = st.slider("Seuil IoU (NMS)", 0.1, 0.95, 0.50, 0.05)
        
#         st.divider()
        
#         st.subheader("📝 OCR Scaleway")
#         ocr_enabled = st.checkbox("Activer l'OCR", value=True)
        
#         if ocr_enabled:
#             st.success("🔑 Clé API configurée")
#             batch_size = st.slider("Taille batches", 1, 10, 6)
#             max_tokens = st.slider("Tokens max", 500, 3000, 1800, 100)
#         else:
#             batch_size = 6
#             max_tokens = 1800
        
#         st.divider()
        
#         st.subheader("🔧 Options")
#         sort_left_to_right = st.radio("Ordre de tri", [True, False], format_func=lambda x: "Gauche → Droite" if x else "Droite → Gauche", index=0)
#         pad_ratio = st.slider("Padding (%)", 0, 15, 3) / 100
    
#     # Upload
#     st.header("📤 Upload d'image")
#     uploaded_file = st.file_uploader("Choisissez une photo de votre étagère", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         # Chargement
#         with st.spinner("⏳ Chargement de l'image..."):
#             file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#             image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
#             if image_bgr is None:
#                 st.error("❌ Impossible de charger l'image")
#                 return
            
#             H, W = image_bgr.shape[:2]
#             st.success(f"✅ Image chargée : {W}x{H}px")
        
#         # Modèle
#         with st.spinner("🤖 Chargement du modèle YOLO..."):
#             model, error = load_yolo_model(MODEL_PATH)
            
#             if model is None:
#                 st.error(f"❌ Erreur de chargement du modèle")
#                 st.error(error)
#                 st.info(f"Vérifiez que le fichier `{MODEL_PATH}` existe et est un modèle YOLO valide.")
#                 return
            
#             st.success("✅ Modèle chargé avec succès")
        
#         # Détection
#         progress_bar = st.progress(0)
        
#         with st.spinner("🔍 Détection des livres en cours..."):
#             progress_bar.progress(20)
            
#             try:
#                 results = model.predict(image_bgr, conf=conf_threshold, iou=iou_threshold, verbose=False)
#                 detections = sv.Detections.from_ultralytics(results[0])
#             except Exception as e:
#                 st.error(f"❌ Erreur lors de la détection : {str(e)}")
#                 progress_bar.empty()
#                 return
            
#             progress_bar.progress(40)
        
#         if len(detections) == 0:
#             st.warning("⚠️ Aucun livre détecté. Essayez d'ajuster les seuils de confiance.")
#             progress_bar.empty()
#             return
        
#         # Annotation
#         box_annotator = sv.OrientedBoxAnnotator(thickness=5)
#         image_annotated = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        
#         label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=5)
#         labels = [f"{conf:.2f}" for conf in detections.confidence]
#         image_annotated = label_annotator.annotate(scene=image_annotated, detections=detections, labels=labels)
        
#         # Extraction crops
#         progress_bar.progress(60)
        
#         if "obb" in detections.data:
#             data_type = "obb"
#             data_array = np.asarray(detections.data["obb"], dtype=float)
#         elif "xyxyxyxy" in detections.data:
#             data_type = "xyxyxyxy"
#             data_array = np.asarray(detections.data["xyxyxyxy"], dtype=float)
#         else:
#             st.error("❌ Format de détection inconnu")
#             progress_bar.empty()
#             return
        
#         conf_array = np.asarray(detections.confidence, dtype=float)
#         keep_mask = conf_array >= conf_threshold
#         indices_keep = np.arange(len(conf_array))[keep_mask]
        
#         crops_raw = []
        
#         for j, original_idx in enumerate(indices_keep):
#             if data_type == "obb":
#                 cx, cy, w, h, angle = data_array[original_idx]
#                 if max(abs(cx), abs(cy), abs(w), abs(h)) <= 1.2:
#                     cx *= W; cy *= H; w *= W; h *= H
#                 quad = obb_to_corners(cx, cy, w, h, angle)
#             else:
#                 pts = data_array[original_idx]
#                 if pts.ndim == 1 and len(pts) == 8:
#                     pts = pts.reshape(4, 2)
#                 if np.max(np.abs(pts)) <= 1.2:
#                     pts[:, 0] *= W; pts[:, 1] *= H
#                 quad = order_quad_points(pts)
            
#             quad[:, 0] = np.clip(quad[:, 0], 0, W - 1)
#             quad[:, 1] = np.clip(quad[:, 1], 0, H - 1)
            
#             crop = extract_crop(image_bgr, quad, pad_ratio=pad_ratio)
            
#             if crop is None:
#                 continue
            
#             center_x = float(quad[:, 0].mean())
            
#             crops_raw.append({
#                 "crop": crop,
#                 "quad": quad,
#                 "center_x": center_x,
#                 "confidence": conf_array[original_idx]
#             })
        
#         # Tri
#         crops_raw.sort(key=lambda x: x["center_x"], reverse=not sort_left_to_right)
        
#         # Numéros
#         for idx, item in enumerate(crops_raw, start=1):
#             quad = item["quad"]
#             center = quad.mean(axis=0).astype(int)
            
#             cv2.circle(image_annotated, tuple(center), 30, (0, 255, 0), -1)
#             cv2.circle(image_annotated, tuple(center), 30, (0, 0, 0), 3)
            
#             text = str(idx)
#             font = cv2.FONT_HERSHEY_SIMPLEX
            
#             text_size = cv2.getTextSize(text, font, 1.5, 4)[0]
#             text_x = center[0] - text_size[0] // 2
#             text_y = center[1] + text_size[1] // 2
            
#             cv2.putText(image_annotated, text, (text_x, text_y), font, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
        
#         progress_bar.progress(80)
        
#         # OCR
#         ocr_results = []
        
#         if ocr_enabled and SCW_API_KEY:
#             with st.spinner("🔤 Extraction du texte avec OCR..."):
#                 ocr_client = create_ocr_client(SCW_API_KEY, SCW_BASE_URL)
                
#                 if ocr_client:
#                     crops = [item["crop"] for item in crops_raw]
                    
#                     for i in range(0, len(crops), batch_size):
#                         batch = crops[i:i+batch_size]
#                         results = process_ocr_batch(batch, ocr_client, max_tokens=max_tokens)
#                         ocr_results.extend(results)
                        
#                         if i + batch_size < len(crops):
#                             time.sleep(2.5)
#                 else:
#                     st.warning("⚠️ Impossible de créer le client OCR")
#                     ocr_results = [{"title": "", "author": "", "publisher": "", "lines": []} for _ in crops_raw]
#         else:
#             ocr_results = [{"title": "", "author": "", "publisher": "", "lines": []} for _ in crops_raw]
        
#         progress_bar.progress(100)
#         time.sleep(0.3)
#         progress_bar.empty()
        
#         # Résultats
#         st.success(f"🎉 {len(crops_raw)} livre(s) détecté(s) !")
        
#         # Métriques
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("📚 Livres", len(crops_raw))
#         with col2:
#             st.metric("📖 Titres", sum(1 for r in ocr_results if r.get("title")))
#         with col3:
#             st.metric("✍️ Auteurs", sum(1 for r in ocr_results if r.get("author")))
#         with col4:
#             st.metric("🏢 Éditeurs", sum(1 for r in ocr_results if r.get("publisher")))
        
#         st.divider()
        
#         # Tabs
#         tab1, tab2, tab3, tab4 = st.tabs(["📸 Image annotée", "📚 Liste détaillée", "📊 Tableau", "💾 Export"])
        
#         with tab1:
#             st.subheader("Image avec détections")
#             st.image(cv2.cvtColor(image_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        
#         with tab2:
#             st.subheader("Liste des livres détectés")
            
#             for idx, (item, ocr_data) in enumerate(zip(crops_raw, ocr_results), start=1):
#                 with st.expander(f"📖 Livre #{idx} - {ocr_data.get('title') or '(inconnu)'}"):
#                     col_img, col_info = st.columns([1, 2])
                    
#                     with col_img:
#                         st.image(cv2.cvtColor(item["crop"], cv2.COLOR_BGR2RGB))
#                         st.caption(f"Confiance : {item['confidence']:.2%}")
                    
#                     with col_info:
#                         st.markdown(f"**Titre:** {ocr_data.get('title') or '*(inconnu)*'}")
#                         st.markdown(f"**Auteur:** {ocr_data.get('author') or '*(inconnu)*'}")
#                         st.markdown(f"**Éditeur:** {ocr_data.get('publisher') or '*(inconnu)*'}")
                        
#                         if ocr_data.get('lines'):
#                             st.markdown("**Lignes détectées:**")
#                             for line in ocr_data['lines'][:10]:
#                                 st.text(f"  • {line}")
        
#         with tab3:
#             st.subheader("Vue tableau")
#             df_data = []
#             for idx, (item, ocr_data) in enumerate(zip(crops_raw, ocr_results), start=1):
#                 df_data.append({
#                     "#": idx,
#                     "Titre": ocr_data.get('title') or '(inconnu)',
#                     "Auteur": ocr_data.get('author') or '(inconnu)',
#                     "Éditeur": ocr_data.get('publisher') or '(inconnu)',
#                     "Confiance": f"{item['confidence']:.2%}"
#                 })
            
#             df = pd.DataFrame(df_data)
#             st.dataframe(df, use_container_width=True, hide_index=True)
        
#         with tab4:
#             st.subheader("Exporter les résultats")
#             col_csv, col_json = st.columns(2)
            
#             with col_csv:
#                 csv = df.to_csv(index=False).encode('utf-8')
#                 st.download_button("📥 Télécharger CSV", csv, "livres.csv", "text/csv", key="csv_download")
            
#             with col_json:
#                 json_data = json.dumps([
#                     {
#                         "index": idx,
#                         "titre": ocr_data.get('title'),
#                         "auteur": ocr_data.get('author'),
#                         "editeur": ocr_data.get('publisher'),
#                         "confiance": float(item['confidence'])
#                     }
#                     for idx, (item, ocr_data) in enumerate(zip(crops_raw, ocr_results), start=1)
#                 ], ensure_ascii=False, indent=2)
                
#                 st.download_button("📥 Télécharger JSON", json_data.encode('utf-8'), "livres.json", "application/json", key="json_download")


# if __name__ == "__main__":
#     main()
    
#     # Footer
#     st.divider()
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("📚 **Book Detector v1.0**")
#     with col2:
#         st.markdown("🚀 *Powered by YOLO + Scaleway OCR*")

# -*- coding: utf-8 -*-
"""
================================================================================
📚 BOOK DETECTOR - Application Streamlit
================================================================================
Détection automatique de livres sur étagère avec OCR intelligent.
"""

import os

# ================== FIX PYTORCH 2.6+ ==================
# Correction pour le chargement des modèles YOLO avec PyTorch 2.6+
try:
    import torch
    # Monkey patch pour forcer weights_only=False
    _original_torch_load = torch.load
    
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    
    torch.load = _patched_torch_load
except Exception:
    pass

import streamlit as st
import math
import json
import base64
import time
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
import pandas as pd
import supervision as sv
from ultralytics import YOLO
import httpx
import certifi
from openai import OpenAI
from fuzzywuzzy import fuzz


# ================== CONFIGURATION API ==================
# Clé API Scaleway configurée directement (ne pas partager publiquement)
os.environ["SCW_SECRET_KEY"] = "ae1d12c9-9fd6-434b-aeba-286e9b9702fa"
SCW_API_KEY = "ae1d12c9-9fd6-434b-aeba-286e9b9702fa"
SCW_BASE_URL = "https://api.scaleway.ai/v1"

# Chemin du modèle YOLO
MODEL_PATH = "models/best.pt"

# ================== AJOUT : Fichier stock ==================
STOCK_FILE = "Stock_20251030.xls"


# ================== CONFIGURATION PAGE ==================
st.set_page_config(
    page_title="📚 Book Detector",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .match-card {
        padding: 10px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .match-found { background: #d4edda; border-left: 4px solid #28a745; }
    .match-uncertain { background: #fff3cd; border-left: 4px solid #ffc107; }
    .match-notfound { background: #f8d7da; border-left: 4px solid #dc3545; }
</style>
""", unsafe_allow_html=True)


# ================== CACHE MODÈLE ==================
@st.cache_resource
def load_yolo_model(model_path: str):
    """Charge le modèle YOLO (mis en cache)."""
    try:
        if not os.path.exists(model_path):
            return None, f"Fichier modèle introuvable : {model_path}"
        
        # Configuration pour PyTorch 2.6+ : autoriser le chargement de modèles personnalisés
        import torch
        
        # Ajouter les classes Ultralytics aux globaux sûrs
        try:
            from ultralytics.nn.tasks import OBBModel, DetectionModel
            torch.serialization.add_safe_globals([OBBModel, DetectionModel])
        except:
            pass
        
        # Charger le modèle avec weights_only=False pour les modèles personnalisés
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


# ================== AJOUT : Cache stock ==================
@st.cache_data
def load_stock(file_path: str) -> Optional[pd.DataFrame]:
    """Charge le fichier stock Excel."""
    try:
        if not os.path.exists(file_path):
            return None
        df = pd.read_excel(file_path, engine='xlrd')
        # Normaliser les colonnes
        df['Titre'] = df['Titre'].astype(str).str.strip().str.upper()
        df['Auteur'] = df['Auteur'].astype(str).str.strip().str.upper()
        df['Editeur'] = df['Editeur'].astype(str).str.strip().str.upper()
        return df
    except Exception as e:
        st.error(f"Erreur chargement stock : {e}")
        return None


# ================== AJOUT : Matching fonctions ==================
def normalize_text(text: str) -> str:
    """Normalise le texte pour le matching."""
    if not text or text == "nan":
        return ""
    text = str(text).upper().strip()
    # Supprimer accents basiques
    replacements = {'À':'A','Á':'A','Â':'A','Ã':'A','Ä':'A','È':'E','É':'E','Ê':'E','Ë':'E',
                   'Ì':'I','Í':'I','Î':'I','Ï':'I','Ò':'O','Ó':'O','Ô':'O','Õ':'O','Ö':'O',
                   'Ù':'U','Ú':'U','Û':'U','Ü':'U','Ç':'C','Œ':'OE','Æ':'AE'}
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Garder seulement alphanum et espaces
    text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    return ' '.join(text.split())


def match_book(ocr_title: str, ocr_author: str, ocr_publisher: str, 
               stock_df: pd.DataFrame, threshold: int = 60):
    """Trouve la meilleure correspondance dans le stock."""
    if stock_df is None or stock_df.empty:
        return None, 0
    
    ocr_title = normalize_text(ocr_title)
    ocr_author = normalize_text(ocr_author)
    ocr_publisher = normalize_text(ocr_publisher)
    
    if not ocr_title and not ocr_author:
        return None, 0
    
    best_score = 0
    best_match = None
    
    for _, row in stock_df.iterrows():
        stock_title = normalize_text(row['Titre'])
        stock_author = normalize_text(row['Auteur'])
        stock_publisher = normalize_text(row['Editeur'])
        
        # Score pondéré : Titre 60%, Auteur 30%, Éditeur 10%
        title_score = 0
        if ocr_title and stock_title:
            title_score = max(
                fuzz.ratio(ocr_title, stock_title),
                fuzz.partial_ratio(ocr_title, stock_title),
                fuzz.token_sort_ratio(ocr_title, stock_title)
            ) * 0.6
        
        author_score = 0
        if ocr_author and stock_author:
            author_score = max(
                fuzz.ratio(ocr_author, stock_author),
                fuzz.token_set_ratio(ocr_author, stock_author)
            ) * 0.3
        
        publisher_score = 0
        if ocr_publisher and stock_publisher:
            publisher_score = fuzz.partial_ratio(ocr_publisher, stock_publisher) * 0.1
        
        score = title_score + author_score + publisher_score
        
        if score > best_score:
            best_score = score
            best_match = row
    
    if best_score >= threshold:
        return best_match, best_score
    return None, best_score


# ================== FONCTIONS GÉOMÉTRIE ==================
def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Ordonne 4 points en [TL, TR, BR, BL]."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    
    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    
    v1 = quad[1] - quad[0]
    v2 = quad[2] - quad[1]
    if np.cross(v1, v2) < 0:
        quad = np.array([tl, bl, br, tr], dtype=np.float32)
    
    return quad


def obb_to_corners(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    """Convertit OBB en 4 coins."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    dx, dy = w / 2.0, h / 2.0
    
    local_corners = np.array([
        [-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]
    ], dtype=np.float32)
    
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    rotated = local_corners @ rotation_matrix.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    
    return order_quad_points(rotated)


def extract_crop(image_bgr: np.ndarray, quad: np.ndarray, pad_ratio: float = 0.03, min_side: int = 20):
    """Extrait un crop depuis un quadrilatère."""
    tl, tr, br, bl = quad
    
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    out_width = int(max(1, round(max(width_top, width_bottom))))
    
    height_right = np.linalg.norm(tr - br)
    height_left = np.linalg.norm(tl - bl)
    out_height = int(max(1, round(max(height_right, height_left))))
    
    if out_width < min_side or out_height < min_side:
        return None
    
    pad_w = int(round(out_width * pad_ratio))
    pad_h = int(round(out_height * pad_ratio))
    out_width_padded = out_width + 2 * pad_w
    out_height_padded = out_height + 2 * pad_h
    
    dst_points = np.array([
        [pad_w, pad_h],
        [pad_w + out_width - 1, pad_h],
        [pad_w + out_width - 1, pad_h + out_height - 1],
        [pad_w, pad_h + out_height - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst_points)
    
    crop = cv2.warpPerspective(
        image_bgr, M,
        (out_width_padded, out_height_padded),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    return crop


# ================== OCR ==================
def create_ocr_client(api_key: str, base_url: str):
    """Crée client OCR Scaleway."""
    if not api_key:
        return None
    
    try:
        http_client = httpx.Client(
            timeout=60.0,
            verify=certifi.where(),
            http2=False,
            limits=httpx.Limits(max_connections=5)
        )
        
        return OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
    except Exception as e:
        st.error(f"Erreur création client OCR : {e}")
        return None


def encode_crop_to_base64(crop: np.ndarray, quality: int = 85) -> str:
    """Encode crop en base64."""
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    if len(crop.shape) == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    
    h, w = crop.shape[:2]
    if max(h, w) > 1000:
        scale = 1000 / max(h, w)
        crop = cv2.resize(crop, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality, int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]
    success, buffer = cv2.imencode(".jpg", crop, encode_params)
    
    if not success:
        raise RuntimeError("Échec encodage JPEG")
    
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


PROMPT_OCR = """Tu es un expert en OCR de tranches de livres français.

CONTEXTE: Chaque image montre UNE tranche de livre (texte vertical ou horizontal).

INSTRUCTIONS: Pour CHAQUE image, extrais:
1. **lines** (array): TOUTES les lignes visibles, de HAUT en BAS
2. **title** (string): Le titre principal
3. **author** (string): L'auteur complet (Prénom NOM)
4. **publisher** (string): L'éditeur si visible

EXEMPLES:

Exemple 1: "LA TABLE | DES LOUPS | Adam Rapp | RIVAGES"
→ {{"lines": ["LA TABLE", "DES LOUPS", "Adam Rapp", "RIVAGES"], "title": "La Table des Loups", "author": "Adam Rapp", "publisher": "RIVAGES"}}

Exemple 2: Image illisible
→ {{"lines": [], "title": "", "author": "", "publisher": ""}}

RÈGLES: N'invente RIEN. Garde les accents. Si incertain → ""

FORMAT: {{"results": [{{"lines": [...], "title": "...", "author": "...", "publisher": "..."}}, ...]}}

Traite les {num_images} images."""


def process_ocr_batch(crops: List[np.ndarray], client, max_tokens: int = 1800, max_retries: int = 4):
    """Traite un batch OCR."""
    num_crops = len(crops)
    
    content = [{"type": "text", "text": PROMPT_OCR.format(num_images=num_crops)}]
    
    for crop in crops:
        content.append({"type": "image_url", "image_url": {"url": encode_crop_to_base64(crop)}})
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="pixtral-12b-2409",
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": content}]
            )
            
            data = json.loads(response.choices[0].message.content or "{}")
            results = data.get("results", [])
            
            normalized = []
            for item in results:
                normalized.append({
                    "title": (item.get("title") or "").strip(),
                    "author": (item.get("author") or "").strip(),
                    "publisher": (item.get("publisher") or "").strip(),
                    "lines": [l.strip() for l in item.get("lines", []) if l]
                })
            
            while len(normalized) < num_crops:
                normalized.append({"title": "", "author": "", "publisher": "", "lines": []})
            
            return normalized[:num_crops]
        
        except Exception as e:
            if "429" in str(e).lower() and attempt < max_retries:
                time.sleep(2.0 * (2 ** attempt))
                continue
            st.warning(f"Erreur OCR (tentative {attempt+1}/{max_retries+1}) : {str(e)}")
            break
    
    return [{"title": "", "author": "", "publisher": "", "lines": []} for _ in crops]


# ================== INTERFACE PRINCIPALE ==================
def main():
    """Interface principale Streamlit."""
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("📚 Book Detector")
        st.markdown("*Détection automatique de livres assistée par IA*")
    
    # ================== AJOUT : Chargement stock ==================
    stock_df = load_stock(STOCK_FILE)
    if stock_df is not None:
        st.success(f"✅ Stock chargé : {len(stock_df):,} livres")
    else:
        st.warning("⚠️ Stock non disponible - Matching désactivé")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("🎯 Détection YOLO")
        st.info(f"📁 Modèle : `{MODEL_PATH}`")
        
        # Vérifier l'existence du modèle
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Fichier modèle introuvable !\n\nChemin attendu : `{os.path.abspath(MODEL_PATH)}`")
            st.info("💡 Assurez-vous que le fichier `best.pt` est dans le même dossier que ce script.")
        
        conf_threshold = st.slider("Seuil de confiance", 0.1, 0.95, 0.50, 0.05)
        iou_threshold = st.slider("Seuil IoU (NMS)", 0.1, 0.95, 0.50, 0.05)
        
        st.divider()
        
        st.subheader("📝 OCR Scaleway")
        ocr_enabled = st.checkbox("Activer l'OCR", value=True)
        
        if ocr_enabled:
            st.success("🔑 Clé API configurée")
            batch_size = st.slider("Taille batches", 1, 10, 6)
            max_tokens = st.slider("Tokens max", 500, 3000, 1800, 100)
        else:
            batch_size = 6
            max_tokens = 1800
        
        st.divider()
        
        # ================== AJOUT : Paramètres matching ==================
        if stock_df is not None:
            st.subheader("🔎 Matching Stock")
            match_threshold = st.slider("Seuil matching (%)", 40, 90, 60, 5)
            st.caption(f"✅ ≥{match_threshold}% : Trouvé")
            st.caption(f"⚠️ 40-{match_threshold-1}% : Incertain")
            st.caption(f"❌ <40% : Non trouvé")
            st.divider()
        else:
            match_threshold = 60
        
        st.subheader("🔧 Options")
        sort_left_to_right = st.radio("Ordre de tri", [True, False], format_func=lambda x: "Gauche → Droite" if x else "Droite → Gauche", index=0)
        pad_ratio = st.slider("Padding (%)", 0, 15, 3) / 100
    
    # Upload
    st.header("📤 Upload d'image")
    uploaded_file = st.file_uploader("Choisissez une photo de votre étagère", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Chargement
        with st.spinner("⏳ Chargement de l'image..."):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image_bgr is None:
                st.error("❌ Impossible de charger l'image")
                return
            
            H, W = image_bgr.shape[:2]
            st.success(f"✅ Image chargée : {W}x{H}px")
        
        # Modèle
        with st.spinner("🤖 Chargement du modèle YOLO..."):
            model, error = load_yolo_model(MODEL_PATH)
            
            if model is None:
                st.error(f"❌ Erreur de chargement du modèle")
                st.error(error)
                st.info(f"Vérifiez que le fichier `{MODEL_PATH}` existe et est un modèle YOLO valide.")
                return
            
            st.success("✅ Modèle chargé avec succès")
        
        # Détection
        progress_bar = st.progress(0)
        
        with st.spinner("🔍 Détection des livres en cours..."):
            progress_bar.progress(20)
            
            try:
                results = model.predict(image_bgr, conf=conf_threshold, iou=iou_threshold, verbose=False)
                detections = sv.Detections.from_ultralytics(results[0])
            except Exception as e:
                st.error(f"❌ Erreur lors de la détection : {str(e)}")
                progress_bar.empty()
                return
            
            progress_bar.progress(40)
        
        if len(detections) == 0:
            st.warning("⚠️ Aucun livre détecté. Essayez d'ajuster les seuils de confiance.")
            progress_bar.empty()
            return
        
        # Annotation
        box_annotator = sv.OrientedBoxAnnotator(thickness=5)
        image_annotated = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        
        label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=5)
        labels = [f"{conf:.2f}" for conf in detections.confidence]
        image_annotated = label_annotator.annotate(scene=image_annotated, detections=detections, labels=labels)
        
        # Extraction crops
        progress_bar.progress(60)
        
        if "obb" in detections.data:
            data_type = "obb"
            data_array = np.asarray(detections.data["obb"], dtype=float)
        elif "xyxyxyxy" in detections.data:
            data_type = "xyxyxyxy"
            data_array = np.asarray(detections.data["xyxyxyxy"], dtype=float)
        else:
            st.error("❌ Format de détection inconnu")
            progress_bar.empty()
            return
        
        conf_array = np.asarray(detections.confidence, dtype=float)
        keep_mask = conf_array >= conf_threshold
        indices_keep = np.arange(len(conf_array))[keep_mask]
        
        crops_raw = []
        
        for j, original_idx in enumerate(indices_keep):
            if data_type == "obb":
                cx, cy, w, h, angle = data_array[original_idx]
                if max(abs(cx), abs(cy), abs(w), abs(h)) <= 1.2:
                    cx *= W; cy *= H; w *= W; h *= H
                quad = obb_to_corners(cx, cy, w, h, angle)
            else:
                pts = data_array[original_idx]
                if pts.ndim == 1 and len(pts) == 8:
                    pts = pts.reshape(4, 2)
                if np.max(np.abs(pts)) <= 1.2:
                    pts[:, 0] *= W; pts[:, 1] *= H
                quad = order_quad_points(pts)
            
            quad[:, 0] = np.clip(quad[:, 0], 0, W - 1)
            quad[:, 1] = np.clip(quad[:, 1], 0, H - 1)
            
            crop = extract_crop(image_bgr, quad, pad_ratio=pad_ratio)
            
            if crop is None:
                continue
            
            center_x = float(quad[:, 0].mean())
            
            crops_raw.append({
                "crop": crop,
                "quad": quad,
                "center_x": center_x,
                "confidence": conf_array[original_idx]
            })
        
        # Tri
        crops_raw.sort(key=lambda x: x["center_x"], reverse=not sort_left_to_right)
        
        # Numéros
        for idx, item in enumerate(crops_raw, start=1):
            quad = item["quad"]
            center = quad.mean(axis=0).astype(int)
            
            cv2.circle(image_annotated, tuple(center), 30, (0, 255, 0), -1)
            cv2.circle(image_annotated, tuple(center), 30, (0, 0, 0), 3)
            
            text = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            text_size = cv2.getTextSize(text, font, 1.5, 4)[0]
            text_x = center[0] - text_size[0] // 2
            text_y = center[1] + text_size[1] // 2
            
            cv2.putText(image_annotated, text, (text_x, text_y), font, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
        
        progress_bar.progress(80)
        
        # OCR
        ocr_results = []
        
        if ocr_enabled and SCW_API_KEY:
            with st.spinner("🔤 Extraction du texte avec OCR..."):
                ocr_client = create_ocr_client(SCW_API_KEY, SCW_BASE_URL)
                
                if ocr_client:
                    crops = [item["crop"] for item in crops_raw]
                    
                    for i in range(0, len(crops), batch_size):
                        batch = crops[i:i+batch_size]
                        results = process_ocr_batch(batch, ocr_client, max_tokens=max_tokens)
                        ocr_results.extend(results)
                        
                        if i + batch_size < len(crops):
                            time.sleep(2.5)
                else:
                    st.warning("⚠️ Impossible de créer le client OCR")
                    ocr_results = [{"title": "", "author": "", "publisher": "", "lines": []} for _ in crops_raw]
        else:
            ocr_results = [{"title": "", "author": "", "publisher": "", "lines": []} for _ in crops_raw]
        
        progress_bar.progress(90)
        
        # ================== AJOUT : Matching avec le stock ==================
        matches = []
        if stock_df is not None and ocr_enabled:
            with st.spinner("🔎 Matching avec le stock..."):
                for ocr_data in ocr_results:
                    match_row, score = match_book(
                        ocr_data.get('title', ''),
                        ocr_data.get('author', ''),
                        ocr_data.get('publisher', ''),
                        stock_df,
                        match_threshold
                    )
                    
                    if match_row is not None:
                        status = 'found'
                    elif score >= 40:
                        status = 'uncertain'
                    else:
                        status = 'notfound'
                    
                    matches.append({
                        'match': match_row,
                        'score': score,
                        'status': status
                    })
        else:
            matches = [{'match': None, 'score': 0, 'status': 'notfound'} for _ in ocr_results]
        
        progress_bar.progress(100)
        time.sleep(0.3)
        progress_bar.empty()
        
        # Résultats
        st.success(f"🎉 {len(crops_raw)} livre(s) détecté(s) !")
        
        # ================== AJOUT : Métriques avec matching ==================
        found_count = sum(1 for m in matches if m['status'] == 'found')
        uncertain_count = sum(1 for m in matches if m['status'] == 'uncertain')
        notfound_count = sum(1 for m in matches if m['status'] == 'notfound')
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📚 Détectés", len(crops_raw))
        with col2:
            st.metric("✅ Trouvés", found_count)
        with col3:
            st.metric("⚠️ Incertains", uncertain_count)
        with col4:
            st.metric("❌ Introuvables", notfound_count)
        with col5:
            st.metric("📖 Titres OCR", sum(1 for r in ocr_results if r.get("title")))
        
        st.divider()
        
        # ================== AJOUT : Tabs avec matching ==================
        if stock_df is not None:
            tabs = st.tabs(["📸 Image", "✅ Trouvés", "⚠️ Incertains", "❌ Introuvables", "📚 Liste OCR", "💾 Export"])
            
            with tabs[0]:
                st.subheader("Image avec détections")
                st.image(cv2.cvtColor(image_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with tabs[1]:
                st.subheader(f"✅ Livres trouvés dans le stock ({found_count})")
                for idx, (item, ocr_data, match_data) in enumerate(zip(crops_raw, ocr_results, matches), start=1):
                    if match_data['status'] != 'found':
                        continue
                    match = match_data['match']
                    with st.expander(f"📖 #{idx} - {ocr_data.get('title') or '(inconnu)'} - Score: {match_data['score']:.0f}%"):
                        col_img, col_info = st.columns([1, 3])
                        with col_img:
                            st.image(cv2.cvtColor(item["crop"], cv2.COLOR_BGR2RGB))
                        with col_info:
                            st.markdown(f"**Code:** `{match['Code article']}`")
                            st.markdown(f"**Titre:** {match['Titre']}")
                            st.markdown(f"**Auteur:** {match['Auteur']}")
                            st.markdown(f"**Éditeur:** {match['Editeur']}")
                            qty = match['Qté']
                            if qty > 5:
                                st.success(f"✅ En stock ({qty})")
                            elif qty > 0:
                                st.warning(f"⚠️ Stock faible ({qty})")
                            else:
                                st.error("❌ Rupture")
            
            with tabs[2]:
                st.subheader(f"⚠️ Correspondances incertaines ({uncertain_count})")
                for idx, (item, ocr_data, match_data) in enumerate(zip(crops_raw, ocr_results, matches), start=1):
                    if match_data['status'] != 'uncertain':
                        continue
                    with st.expander(f"📖 #{idx} - Score: {match_data['score']:.0f}%"):
                        col_img, col_info = st.columns([1, 2])
                        with col_img:
                            st.image(cv2.cvtColor(item["crop"], cv2.COLOR_BGR2RGB))
                        with col_info:
                            st.markdown(f"**OCR:** {ocr_data.get('title')} - {ocr_data.get('author')}")
                            st.caption("Vérification manuelle recommandée")
            
            with tabs[3]:
                st.subheader(f"❌ Livres non trouvés ({notfound_count})")
                for idx, (item, ocr_data, match_data) in enumerate(zip(crops_raw, ocr_results, matches), start=1):
                    if match_data['status'] != 'notfound':
                        continue
                    with st.expander(f"📖 #{idx} - {ocr_data.get('title') or '(inconnu)'}"):
                        col_img, col_info = st.columns([1, 2])
                        with col_img:
                            st.image(cv2.cvtColor(item["crop"], cv2.COLOR_BGR2RGB))
                        with col_info:
                            st.markdown(f"**Titre:** {ocr_data.get('title') or '(vide)'}")
                            st.markdown(f"**Auteur:** {ocr_data.get('author') or '(vide)'}")
            
            with tabs[4]:
                st.subheader("Liste OCR complète")
                for idx, (item, ocr_data) in enumerate(zip(crops_raw, ocr_results), start=1):
                    with st.expander(f"📖 #{idx} - {ocr_data.get('title') or '(inconnu)'}"):
                        col_img, col_info = st.columns([1, 2])
                        with col_img:
                            st.image(cv2.cvtColor(item["crop"], cv2.COLOR_BGR2RGB))
                        with col_info:
                            st.markdown(f"**Titre:** {ocr_data.get('title') or '*(inconnu)*'}")
                            st.markdown(f"**Auteur:** {ocr_data.get('author') or '*(inconnu)*'}")
                            st.markdown(f"**Éditeur:** {ocr_data.get('publisher') or '*(inconnu)*'}")
            
            with tabs[5]:
                st.subheader("Export avec matching")
                df_data = []
                for idx, (item, ocr_data, match_data) in enumerate(zip(crops_raw, ocr_results, matches), start=1):
                    row = {
                        "#": idx,
                        "OCR_Titre": ocr_data.get('title', ''),
                        "OCR_Auteur": ocr_data.get('author', ''),
                        "Status": match_data['status'],
                        "Score": f"{match_data['score']:.1f}%",
                    }
                    if match_data['match'] is not None:
                        m = match_data['match']
                        row.update({
                            "Stock_Code": m['Code article'],
                            "Stock_Titre": m['Titre'],
                            "Stock_Auteur": m['Auteur'],
                            "Stock_Qté": m['Qté']
                        })
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 CSV", csv, "livres_stock.csv", "text/csv")
        else:
            # Version sans matching (originale)
            tabs = st.tabs(["📸 Image", "📚 Liste", "📊 Tableau", "💾 Export"])
            
            with tabs[0]:
                st.subheader("Image avec détections")
                st.image(cv2.cvtColor(image_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with tabs[1]:
                st.subheader("Liste des livres détectés")
                for idx, (item, ocr_data) in enumerate(zip(crops_raw, ocr_results), start=1):
                    with st.expander(f"📖 Livre #{idx} - {ocr_data.get('title') or '(inconnu)'}"):
                        col_img, col_info = st.columns([1, 2])
                        with col_img:
                            st.image(cv2.cvtColor(item["crop"], cv2.COLOR_BGR2RGB))
                        with col_info:
                            st.markdown(f"**Titre:** {ocr_data.get('title') or '*(inconnu)*'}")
                            st.markdown(f"**Auteur:** {ocr_data.get('author') or '*(inconnu)*'}")
                            st.markdown(f"**Éditeur:** {ocr_data.get('publisher') or '*(inconnu)*'}")
            
            with tabs[2]:
                df_data = []
                for idx, (item, ocr_data) in enumerate(zip(crops_raw, ocr_results), start=1):
                    df_data.append({
                        "#": idx,
                        "Titre": ocr_data.get('title') or '(inconnu)',
                        "Auteur": ocr_data.get('author') or '(inconnu)',
                        "Éditeur": ocr_data.get('publisher') or '(inconnu)',
                    })
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
            
            with tabs[3]:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 CSV", csv, "livres.csv", "text/csv")


if __name__ == "__main__":
    main()
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("📚 **Book Detector v1.0**")
    with col2:
        st.markdown("🚀 *Powered by YOLO + Scaleway OCR*")