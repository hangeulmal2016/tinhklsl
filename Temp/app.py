import streamlit as st
import numpy as np
import pandas as pd

# 🛠️ CẤU HÌNH BẮT BUỘC ĐỂ KHÔNG BỊ LỖI ĐỒ HỌA TRÊN STREAMLIT COMMUNITY CLOUD
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import ezdxf
import io

# Thiết lập giao diện hiển thị cho Web App tối ưu cho Mobile
st.set_page_config(page_title="Darvis TIN Mobile App", layout="centered")
st.title("🗺️ Ứng Dụng Darvis TIN (Giao Diện Di Động)")
st.markdown("Cấu trúc xếp chồng **Khung tổng thể** và **Khung cục bộ** giúp xem trực quan, rõ ràng trên màn hình điện thoại.")

# Khởi tạo các biến nhớ lưu trạng thái tương tác của người dùng (Session State)
if 'manual_boundary' not in st.session_state:
    st.session_state.manual_boundary = []
if 'ignored_triangles' not in st.session_state:
    st.session_state.ignored_triangles = set()
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = ""

def parse_txt_data(text_content):
    """Đọc file văn bản TXT và tự động lọc bỏ các điểm trùng lặp tọa độ X, Y"""
    points = []
    for line in text_content.strip().split('\n'):
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                x = float(parts)
                y = float(parts)
                z = float(parts) if len(parts) > 2 else 0.0
                points.append((x, y, z))
            except ValueError:
                continue
    df_pts = pd.DataFrame(points, columns=['x', 'y', 'z']).drop_duplicates(subset=['x', 'y'])
    return df_pts.to_numpy()

def parse_dxf_data(file_bytes):
    """Đọc file bản vẽ DXF và lọc bỏ các điểm có tọa độ trùng nhau"""
    points = []
    try:
        stream = io.StringIO(file_bytes.decode('utf-8', errors='ignore'))
        doc = ezdxf.read(stream)
        msp = doc.modelspace()
        for entity in msp.query('POINT'):
            p = entity.dxf.location
            points.append((p.x, p.y, p.z))
        if not points:
            for entity in msp.query('TEXT'):
                p = entity.dxf.insert
                try: z = float(entity.dxf.text)
                except ValueError: z = p.z
                points.append((p.x, p.y, z))
    except Exception as e:
        st.error(f"Lỗi đọc định dạng cấu trúc DXF: {e}")
    df_pts = pd.DataFrame(points, columns=['x', 'y', 'z']).drop_duplicates(subset=['x', 'y'])
    return df_pts.to_numpy()
def get_polygon_properties(coords):
    """Tính chu vi và diện tích phẳng hình học của polygon đường bao"""
    x = coords[:, 0]
    y = coords[:, 1]
    dx = np.diff(x, append=x)
    dy = np.diff(y, append=y)
    perimeter = np.sum(np.sqrt(dx**2 + dy**2))
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return perimeter, area

def generate_cad_dxf(points, boundary_indices):
    """Xuất file DXF CAD chuẩn với Layer phân biệt: BOUNDARY và CAODO"""
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    doc.layers.add(name='CAODO', color=3)     # Lớp chữ nhãn cao độ - Màu xanh lá
    doc.layers.add(name='BOUNDARY', color=1)  # Lớp nét polyline biên ranh giới - Màu đỏ
    
    # ĐÃ ĐIỀU CHỈNH X2: Tăng chiều cao nhãn chữ cao độ trong CAD từ 0.4 lên 0.8
    for p in points:
        msp.add_text(
            text=f"{p:.3f}",
            dxfattribs={
                'layer': 'CAODO',
                'height': 0.8,
                'insert': (p, p, p),
                'style': 'Standard'
            }
        )
        
    if len(boundary_indices) > 2:
        poly_points = [(points[idx], points[idx]) for idx in boundary_indices]
        msp.add_lwpolyline(points=poly_points, dxfattribs={'layer': 'BOUNDARY', 'flags': 1})
        
    out_stream = io.StringIO()
    doc.write(out_stream)
    return out_stream.getvalue()
# --- SIDEBAR: KHU VỰC NẠP FILE VÀ KIỂM TRA ĐỔI FILE ---
st.sidebar.header("⚙️ Nạp Dữ Liệu Đầu Vào")
upload_mode = st.sidebar.selectbox("Định dạng file:", ["File tọa độ văn bản (.TXT)", "File bản vẽ CAD (.DXF)"])
uploaded_file = st.sidebar.file_uploader("Tải tệp hình học của bạn lên đây để tính toán", type=["txt", "dxf"])

if uploaded_file is not None and uploaded_file.name != st.session_state.current_file_name:
    st.session_state.current_file_name = uploaded_file.name
    st.session_state.manual_boundary = []
    st.session_state.ignored_triangles = set()

if uploaded_file is None:
    st.info("👋 Vui lòng tải lên file dữ liệu (.TXT hoặc .DXF) ở thanh công cụ bên trái (Sidebar) để bắt đầu hiển thị mô hình.")
else:
    file_bytes = uploaded_file.read()
    if "DXF" in upload_mode:
        pts = parse_dxf_data(file_bytes)
    else:
        pts = parse_txt_data(file_bytes.decode("utf-8"))

    if len(pts) >= 3:
        pts_2d = pts[:, :2]
        tri_mesh = Delaunay(pts_2d)
        x_min_idx = int(np.argmin(pts_2d[:, 0]))

        if not st.session_state.manual_boundary:
            st.session_state.manual_boundary = [x_min_idx]

        curr_pivot = st.session_state.manual_boundary[-1]

        adjacent_tri_indices = []
        adjacent_nodes = set()
        for idx, simplex in enumerate(tri_mesh.simplices):
            if idx in st.session_state.ignored_triangles:
                continue
            if curr_pivot in simplex:
                adjacent_tri_indices.append(idx)
                for node in simplex:
                    if node != curr_pivot:
                        adjacent_nodes.add(int(node))

        # --- BẢNG ĐIỀU KHIỂN CHIỀU DỌC MOBILE ---
        st.subheader("🕹️ Bảng Điều Khiển Ranh Giới")
        st.markdown(f"📍 **Điểm chủ hiện tại:** Đỉnh `#{curr_pivot}` (X: {pts[curr_pivot]:.2f}, Y: {pts[curr_pivot]:.2f})")
        
        available_options = sorted(list(adjacent_nodes))
        next_chosen_node = st.selectbox("Chọn đỉnh tiếp theo nằm trên Boundary:", available_options)
        
        if st.button("👉 Xác nhận đỉnh này thuộc Boundary", use_container_width=True):
            if next_chosen_node == x_min_idx and len(st.session_state.manual_boundary) > 2:
                st.session_state.manual_boundary.append(int(next_chosen_node))
                st.success("🎉 Đường bao Darvis TIN đã khép mạch thành công!")
            elif next_chosen_node not in st.session_state.manual_boundary:
                st.session_state.manual_boundary.append(int(next_chosen_node))
                st.rerun()
            else:
                st.warning("Đỉnh này đã tồn tại trong chuỗi ranh giới trước đó.")
        if adjacent_tri_indices:
            tri_to_ignore = st.selectbox("Chọn Tam giác muốn ẩn để định hướng Boundary:", adjacent_tri_indices, format_func=lambda i: f"Tam giác số #{i}")
            if st.button("❌ Ẩn tam giác này để lọc tập T", use_container_width=True):
                st.session_state.ignored_triangles.add(tri_to_ignore)
                st.rerun()
        else:
            st.write("Không tìm thấy tam giác nào quanh đỉnh hiện tại.")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("⏪ Quay lại (Undo)", use_container_width=True):
                if len(st.session_state.manual_boundary) > 1:
                    st.session_state.manual_boundary.pop()
                    st.rerun()
        with col_btn2:
            if st.button("🔄 Làm lại (Reset)", use_container_width=True):
                st.session_state.manual_boundary = [x_min_idx]
                st.session_state.ignored_triangles = set()
                st.rerun()

        if len(st.session_state.manual_boundary) >= 3:
            p_len, p_area = get_polygon_properties(pts_2d[st.session_state.manual_boundary])
            st.markdown(f"📊 **Thông số phẳng:** Chu vi: `{p_len:.2f} m` | Diện tích: `{p_area:.2f} m²`")

        st.markdown("---")
        
        # ----------------------------------------------------
        # KHUNG ĐỒ HỌA 1 (PHÍA TRÊN): BẢN ĐỒ TỔNG THỂ LƯỚI TIN
        # ----------------------------------------------------
        st.subheader("🌐 1. Khung Đồ Họa Tổng Thể Lưới TIN")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        
        for idx, simplex in enumerate(tri_mesh.simplices):
            if idx in st.session_state.ignored_triangles:
                ax1.plot(pts_2d[simplex, 0], pts_2d[simplex, 1], color='#E0E0E0', linewidth=0.6, linestyle=':', alpha=0.3, zorder=1)
            else:
                ax1.plot(pts_2d[simplex, 0], pts_2d[simplex, 1], color='#B0B0B0', linewidth=0.9, linestyle='-', zorder=1)
            
        ax1.scatter(pts_2d[:, 0], pts_2d[:, 1], color='#0275d8', s=25, zorder=2)
        
        if len(st.session_state.manual_boundary) > 0:
            active_idx = st.session_state.manual_boundary
            ax1.plot(pts_2d[active_idx, 0], pts_2d[active_idx, 1], color='#d9534f', linewidth=2.5, marker='o', zorder=3)
            ax1.scatter(pts_2d[x_min_idx, 0], pts_2d[x_min_idx, 1], color='#f0ad4e', s=140, marker='*', edgecolors='black', zorder=4)
            ax1.scatter(pts_2d[curr_pivot, 0], pts_2d[curr_pivot, 1], color='#5cb85c', s=90, edgecolors='black', zorder=4)
            
        ax1.grid(True, linestyle=':', alpha=0.4)
        ax1.set_aspect('equal', adjustable='datalim')
        st.pyplot(fig1)

        st.markdown("---")

        # ----------------------------------------------------
        # KHUNG ĐỒ HỌA 2 (PHÍA DƯỚI): PHÓNG TO VÙNG CỤC BỘ (CHỮ ĐIỂM X2)
        # ----------------------------------------------------
        st.subheader("🔍 2. Khung Cận Cảnh Điểm Đang Xử Lý")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        local_pts_idx = set([curr_pivot])
        for idx in adjacent_tri_indices:
            simplex = tri_mesh.simplices[idx]
            polygon = plt.Polygon(pts_2d[simplex], facecolor='#FFF2CC', edgecolor='#D6B656', linewidth=1.5, alpha=0.9, zorder=1)
            ax2.add_patch(polygon)
            for node in simplex:
                local_pts_idx.add(node)
            
            centroid_x = np.mean(pts_2d[simplex, 0])
            centroid_y = np.mean(pts_2d[simplex, 1])
            ax2.text(centroid_x, centroid_y, f"#{idx}", fontsize=26, color='#CC0000', fontweight='bold',
                    ha='center', va='center', zorder=2,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFFF', edgecolor='#CC0000', alpha=0.9))
        
        local_pts_idx = list(local_pts_idx)
        if local_pts_idx:
            ax2.scatter(pts_2d[local_pts_idx, 0], pts_2d[local_pts_idx, 1], color='#0275d8', s=40, zorder=3)
            # ĐÃ ĐIỀU CHỈNH X2: Tăng cỡ chữ hiển thị thông tin điểm trên biểu đồ từ 8 lên 16
            for idx in local_pts_idx:
                p = pts[idx]
                ax2.text(p + 0.1, p + 0.1, f"Z:{p:.2f}\n(#{idx})", fontsize=16, color='#111111', fontweight='bold', zorder=4)
        
        ax2.scatter(pts_2d[curr_pivot, 0], pts_2d[curr_pivot, 1], color='#5cb85c', s=130, edgecolors='black', zorder=5)
        
        ax2.grid(True, linestyle=':', alpha=0.4)
        ax2.set_aspect('equal', adjustable='datalim')
        st.pyplot(fig2)

        # --- KẾT XUẤT CAD ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("💾 Tải Bản Vẽ Kiểm Tra")
        dxf_output_raw = generate_cad_dxf(pts, st.session_state.manual_boundary)
        st.sidebar.download_button(
            label="📥 Tải file DXF (Darvis TIN)",
            data=dxf_output_raw,
            file_name="darvis_tin_interactive.dxf",
            mime="application/dxf",
            use_container_width=True
        )
    else:
        st.error("❌ Tệp dữ liệu khảo sát không chứa đủ 3 điểm tọa độ.")
