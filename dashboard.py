import streamlit as st
# from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import rasterio
import plotly.express as px
from rasterio.enums import Resampling
import rasterio.plot
import geopandas as gpd
import matplotlib.pyplot as plt

# -----------------------
# Streamlit Dashboard App
# -----------------------

st.set_page_config(page_title="Road Defect Detection Dashboard", layout="wide")

st.title("🛣️ UAV / Aerial Road Defect Detection Dashboard")
st.markdown(
    """
    This dashboard allows you to visualize the results of the YOLOv8 model on UAV / Arial images.
    You can select an image from a folder and see the detected defects. 
    """
)

# Sidebar controls
st.sidebar.header("🔧 Settings")
image_folder = st.sidebar.text_input("Image Folder", "stitched/")
model_path = st.sidebar.text_input("YOLO Model Path", "runs/detect/train6/weights/best.pt")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
fiter = st.sidebar.slider("PCI Filter", 0.0, 100.0, 1.0, 1.0)


st.header("Aerial Data")

# Load class names
class_file = Path("class_names.txt")
classes = [line.strip() for line in class_file.read_text().splitlines() if line.strip()]
selected_classes = st.sidebar.multiselect("Defect Types", classes, default=classes)

# @st.cache_resource
# def load_yolo_model(path):
#     return YOLO(path)

# model = load_yolo_model(model_path)

# List images
image_dir = Path(image_folder)
image_paths = sorted(image_dir.glob("*.*"))
image_names = [p.name for p in image_paths]
selected_image = st.selectbox("Select Image", image_names)

if selected_image:
    img_path = image_dir / selected_image
    # Run YOLO inference
    # results = model(img_path, conf=confidence)[0]
    # Read and prepare image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Draw detections
    # for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
    #     cls_name = classes[int(cls_id)]
    #     if cls_name in selected_classes:
    #         x1, y1, x2, y2 = map(int, box.cpu().numpy())
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #         cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, max(y1 - 10, 0)),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display image
    st.image(img, use_container_width=True)

# create 5 equal-width columns
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

# put Image Width in the second column
with col2:
    st.metric(label="PCI", value="55")

with col4:
    st.metric(label="Image Width", value=w)

# put Image Height in the fourth column
with col6:
    st.metric(label="Image Height", value=h)


st.markdown("---")
st.header("Satellite Data")
tiff_path = st.sidebar.text_input("TIFF File Path", "pothole_satellite.tif")
if tiff_path:
    tif_file = Path(tiff_path)
    if tif_file.exists():
        # Read GeoTIFF
        with rasterio.open(tiff_path) as src:
            # choose max output dimension
            max_dim = 1024
            scale = min(max_dim / src.width, max_dim / src.height, 1.0)
            out_w = int(src.width * scale)
            out_h = int(src.height * scale)
            # read and resample
            data = src.read(
                out_shape=(src.count, out_h, out_w),
                resampling=Resampling.bilinear
            )
            # build RGB (or single band) image
            if data.shape[0] >= 3:
                img = np.dstack([data[i] for i in range(3)])
            else:
                img = data[0]

    # show interactively
    fig = px.imshow(img, origin="upper")
    fig.update_layout(dragmode="pan", margin=dict(l=0,r=0,t=0,b=0))
    fig.update_layout(
        dragmode="pan",
        margin=dict(l=0, r=0, t=0, b=0),
        height=1000,       # set plot height in pixels
        # width=1200,     # optionally set width (otherwise container width)
    )

    # 2) Then render with Streamlit, optionally specifying a height:
    st.plotly_chart(
        fig,
        use_container_width=True,
        height=1000     # match the fig height or adjust as you like
    )


# tiff_path = st.sidebar.text_input("GeoTIFF Path", "pothole_satellite.tif")
# shapefile_path = st.sidebar.text_input("Shapefile Path", "potholes_shapefile.shp")

# if not Path(tiff_path).exists():
#     st.error(f"GeoTIFF not found: {tiff_path}")
# elif not Path(shapefile_path).exists():
#     st.error(f"Shapefile not found: {shapefile_path}")
# else:
#     # Read & downsample
#     with rasterio.open(tiff_path) as src:
#         max_dim = 1024
#         scale = min(max_dim/src.width, max_dim/src.height, 1.0)
#         out_shape = (src.count,
#                      int(src.height * scale),
#                      int(src.width * scale))
#         data = src.read(
#             out_shape=out_shape,
#             resampling=rasterio.enums.Resampling.bilinear
#         )
#         transform = src.transform
#         # Prepare image array
#         if data.shape[0] >= 3:
#             # move bands to last axis
#             img = np.moveaxis(data[:3], 0, -1)
#         else:
#             img = data[0]
#         crs = src.crs

#     # Load shapefile
#     shp = gpd.read_file(shapefile_path).to_crs(crs)

#     # Plot using matplotlib directly
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.imshow(img, origin="upper")
#     shp.boundary.plot(ax=ax, edgecolor="red", linewidth=2)
#     ax.set_axis_off()

#     st.pyplot(fig)


st.markdown("---")
st.header("Information")
st.markdown("**PCI:** Pavement Condition Index")


# Footer
st.markdown("---")
st.markdown("Powered by [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and Streamlit")
