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

# Thiết lập giao diện hiển thị cho Web App
st.set_page_config(page_title="Darvis TIN Interactive App", layout="wide")
st.title("🗺️ Ứng Dụng Tương Tác Đường Bao - Darvis TIN (Từng Bước)")
st.markdown("Hệ thống hỗ trợ nạp dữ liệu, loại bỏ điểm trùng lặp, dựng lưới TIN và **cho phép bạn phê duyệt/chọn từng đỉnh ranh giới**.")

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
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2]) if len(parts) > 2 else 0.0
                points.append((x, y, z))
            except ValueError:
                continue
    # Loại bỏ điểm thừa trùng lặp (Duplicate Elimination) bằng Pandas
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
    
    # Tạo lớp (Layer) bản vẽ kỹ thuật kèm mã màu chuẩn
    doc.layers.add(name='CAODO', color=3)     # Lớp chữ nhãn cao độ - Màu xanh lá
    doc.layers.add(name='BOUNDARY', color=1)  # Lớp nét polyline biên ranh giới - Màu đỏ
    
    # Ghi nhãn chữ text cao độ lên bản vẽ
    for p in points:
        msp.add_text(
            text=f"{p[2]:.3f}",
            dxfattribs={
                'layer': 'CAODO',
                'height': 0.4,
                'insert': (p[0], p[1], p[2]),
                'style': 'Standard'
            }
        )
        
    # Tạo đường đa tuyến khép kín (Closed Polyline) nối chu vi ranh giới Darvis TIN
    if len(boundary_indices) > 2:
        poly_points = [(points[idx][0], points[idx][1]) for idx in boundary_indices]
        # Thêm flag=1 để định dạng đường Polyline dạng Closed khép kín hoàn toàn trong CAD
        msp.add_lwpolyline(points=poly_points, dxfattribs={'layer': 'BOUNDARY', 'flags': 1})
        
    out_stream = io.StringIO()
    doc.write(out_stream)
    return out_stream.getvalue()
# --- SIDEBAR: KHU VỰC NẠP FILE VÀ KIỂM TRA ĐỔI FILE ---
st.sidebar.header("⚙️ Nạp Dữ Liệu Đầu Vào")
upload_mode = st.sidebar.selectbox("Định dạng file:", ["File tọa độ văn bản (.TXT)", "File bản vẽ CAD (.DXF)"])
uploaded_file = st.sidebar.file_uploader("Tải tệp hình học của bạn lên đây để tính toán", type=["txt", "dxf"])

# Nếu người dùng thay đổi tệp dữ liệu khác, tự động xóa bộ nhớ đệm tương tác cũ
if uploaded_file is not None and uploaded_file.name != st.session_state.current_file_name:
    st.session_state.current_file_name = uploaded_file.name
    st.session_state.manual_boundary = []
    st.session_state.ignored_triangles = set()

if uploaded_file is None:
    st.info("👋 Chào mừng bạn đến với công cụ Darvis TIN! Vui lòng tải lên file dữ liệu (.TXT hoặc .DXF) ở thanh công cụ bên trái (Sidebar) để bắt đầu tự động xác định đường bao chu vi.")
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

        # Đặt điểm cực tây Xmin làm điểm xuất phát đầu tiên của ranh giới
        if not st.session_state.manual_boundary:
            st.session_state.manual_boundary = [x_min_idx]

        curr_pivot = st.session_state.manual_boundary[-1]

        # Khóa dữ liệu và trích xuất tập T (các đỉnh tam giác kề trực tiếp với điểm chủ hiện tại)
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

        # --- GIAO DIỆN PHÊ DUYỆT TỪNG BƯỚC ---
        st.subheader("🕹️ Bảng Điều Khiển Từng Đỉnh Ranh Giới (Darvis TIN)")
        col_ui1, col_ui2, col_ui3 = st.columns(3)
        
        with col_ui1:
            st.markdown(f"**Điểm chủ hiện thời:** Đỉnh `#{curr_pivot}` (X: {pts[curr_pivot][0]:.2f}, Y: {pts[curr_pivot][1]:.2f})")
            available_options = sorted(list(adjacent_nodes))
            next_chosen_node = st.selectbox("Chọn đỉnh tiếp theo nằm trên Boundary:", available_options)
            
            if st.button("👉 Xác nhận đỉnh này thuộc Boundary", use_container_width=True):
                if next_chosen_node == x_min_idx and len(st.session_state.manual_boundary) > 2:
                    st.session_state.manual_boundary.append(int(next_chosen_node))
                    st.success("🎉 Đường bao Darvis TIN đã khép mạch khép kín thành công!")
                elif next_chosen_node not in st.session_state.manual_boundary:
                    st.session_state.manual_boundary.append(int(next_chosen_node))
                    st.rerun()
                else:
                    st.warning("Đỉnh này đã tồn tại trong chuỗi ranh giới trước đó.")
        with col_ui2:
            st.markdown("**Quản lý tam giác lỗi / ngoài ranh:**")
            if adjacent_tri_indices:
                tri_to_ignore = st.selectbox("Chọn tam giác TIN muốn loại bỏ:", adjacent_tri_indices, format_func=lambda i: f"Tam giác #{i} {tri_mesh.simplices[i]}")
                if st.button("❌ Loại bỏ tam giác này khỏi lưới", use_container_width=True):
                    st.session_state.ignored_triangles.add(tri_to_ignore)
                    st.rerun()
            else:
                st.write("Không tìm thấy tam giác nào quanh đỉnh hiện tại.")

        with col_ui3:
            st.markdown("**Thao tác chỉnh sửa luồng:**")
            if st.button("⏪ Quay lại đỉnh trước đó (Undo)", use_container_width=True):
                if len(st.session_state.manual_boundary) > 1:
                    st.session_state.manual_boundary.pop()
                    st.rerun()
            if st.button("🔄 Khởi động lại từ đầu (Reset)", use_container_width=True):
                st.session_state.manual_boundary = [x_min_idx]
                st.session_state.ignored_triangles = set()
                st.rerun()

        # Tính toán thông số ranh giới theo thời gian thực (Real-time Metric)
        if len(st.session_state.manual_boundary) >= 3:
            p_len, p_area = get_polygon_properties(pts_2d[st.session_state.manual_boundary])
            st.markdown(f"📊 **Thông số hiện hành:** Chiều dài chu vi: `{p_len:.2f} m` | Diện tích ranh giới: `{p_area:.2f} m²`")

        # --- DỰNG BIỂU ĐỒ HÌNH HỌC TRỰC QUAN ---
        fig, ax = plt.subplots(figsize=(10, 6.5))
        
        # Vẽ các tam giác nền, tô màu vàng nhạt các tam giác kề (Tập T) để duyệt ranh giới
        for idx, simplex in enumerate(tri_mesh.simplices):
            if idx in st.session_state.ignored_triangles:
                continue
            if idx in adjacent_tri_indices:
                polygon = plt.Polygon(pts_2d[simplex], facecolor='#FFF2CC', edgecolor='#D6B656', linewidth=1, alpha=0.7, label='Tam giác kề (Tập T)' if 'Tam giác kề (Tập T)' not in ax.get_legend_handles_labels() else "")
                ax.add_patch(polygon)
            else:
                ax.plot(pts_2d[simplex, 0], pts_2d[simplex, 1], color='#E5E5E5', linewidth=0.7, linestyle='--')

        # Biểu diễn các chấm tọa độ phẳng
        ax.scatter(pts_2d[:, 0], pts_2d[:, 1], color='#0275d8', s=25, zorder=3)
        # Viết nhãn Text nhãn cao độ thực tế và ký hiệu số thứ tự đỉnh lên màn hình
        for idx, p in enumerate(pts):
            ax.text(p[0] + 0.2, p[1] + 0.2, f"Z:{p[2]:.2f}\n(#{idx})", fontsize=7, color='#4A4A4A', zorder=4)

        # Vẽ đường ranh giới đang xác định từng bước
        if len(st.session_state.manual_boundary) > 0:
            active_idx = st.session_state.manual_boundary
            ax.plot(pts_2d[active_idx, 0], pts_2d[active_idx, 1], color='#d9534f', linewidth=2.5, marker='o', zorder=5, label='Đường bao Darvis TIN')
            # Ngôi sao màu vàng đánh dấu điểm xuất phát Xmin
            ax.scatter(pts_2d[x_min_idx, 0], pts_2d[x_min_idx, 1], color='#f0ad4e', s=130, marker='*', edgecolors='black', zorder=6, label='Khởi phát Xmin')
            # Điểm tròn màu xanh lá đánh dấu điểm chủ hiện tại đang chọn
            ax.scatter(pts_2d[curr_pivot, 0], pts_2d[curr_pivot, 1], color='#5cb85c', s=80, edgecolors='black', zorder=6, label='Điểm chủ hiện tại')

        ax.set_title("Mô hình hình học tương tác từng bước (Darvis TIN)", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(loc="lower right")
        
        # Đồng bộ tỉ lệ trục bản vẽ CAD, triệt tiêu lỗi chính tả datalim cũ
        ax.set_aspect('equal', adjustable='datalim')
        st.pyplot(fig)

        # --- KẾT XUẤT VÀ TẢI FILE DXF TRỰC TIẾP ---
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
        st.error("❌ Tệp dữ liệu khảo sát không hợp lệ hoặc số lượng điểm tọa độ ít hơn 3.")
