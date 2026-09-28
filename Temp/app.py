import streamlit as st
import ezdxf
import pandas as pd
import numpy as np
import io
import time
from scipy.spatial import Delaunay, KDTree

st.set_page_config(page_title="Sửa Lỗi Đường Bao & Xuất Tọa Độ", layout="wide", page_icon="📐")

st.title("📐 Sửa Lỗi Đường Bao Hoàn Chỉnh & Xuất Dữ Liệu Tọa Độ")
st.markdown("Hệ thống sửa lỗi hình học góc quét và bổ sung chức năng xuất file bảng tính CSV tọa độ đường bao ngoài.")

if 'survey_df' not in st.session_state:
    st.session_state.survey_df = None

st.sidebar.header("📁 Tải tệp số liệu trắc địa")
uploaded_file = st.sidebar.file_uploader("Chọn file dữ liệu gốc (TXT hoặc DXF)", type=["txt", "dxf"])
num_points_test = st.sidebar.slider("Hoặc chạy thử với số điểm giả lập", 100, 5000, 1000, step=100)
generate_test = st.sidebar.button("🔄 Sinh dữ liệu phức tạp chạy thử")

def parse_txt_large(file_bytes):
    data = io.BytesIO(file_bytes)
    df = pd.read_csv(data, sep=r'[,\s\t]+', engine='python', header=None)
    if df.shape[1] >= 4:
        df = df.iloc[:, 1:4]
    elif df.shape[1] == 2:
        df[2] = 0.0
    df.columns = ['X', 'Y', 'Z']
    df['X'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
    df['Z'] = pd.to_numeric(df['Z'], errors='coerce').fillna(0.0)
    return df.dropna(subset=['X', 'Y'])

def is_inside_triangle(p, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def compute_correct_boundary(points_2d):
    """Thuật toán quét tam giác rỗng có định hướng góc để sửa lỗi đường thẳng"""
    n_points = len(points_2d)
    xmin_idx = np.argmin(points_2d[:, 0])
    xmax_idx = np.argmax(points_2d[:, 0])
    
    pt_Xmin = points_2d[xmin_idx]
    pt_Xmax = points_2d[xmax_idx]
    max_search_radius = float(np.linalg.norm(pt_Xmax - pt_Xmin))
    tree = KDTree(points_2d)
    
    boundary_indices = [xmin_idx]
    current_idx = xmin_idx
    
    # Hướng vector tham chiếu ban đầu đi thẳng đứng lên trên
    ref_vector = np.array([0.0, 1.0])
    visited = set()
    
    for step in range(n_points):
        current_pt = points_2d[current_idx]
        neighbor_indices = tree.query_ball_point(current_pt, max_search_radius)
        
        best_next_idx = -1
        min_angle = float('inf')
        
        for i in neighbor_indices:
            if i == current_idx: continue
            if i == xmin_idx and len(boundary_indices) >= 3:
                best_next_idx = xmin_idx
                break
            if i in visited: continue
                
            test_pt = points_2d[i]
            test_vector = test_pt - current_pt
            
            norms = np.linalg.norm(ref_vector) * np.linalg.norm(test_vector)
            if norms == 0: continue
            
            # Sử dụng atan2 để xác định chính xác góc định hướng 360 độ chống lỗi trùng đường thẳng
            angle_ref = np.arctan2(ref_vector[1], ref_vector[0])
            angle_test = np.arctan2(test_vector[1], test_vector[0])
            diff_angle = (angle_test - angle_ref) % (2 * np.pi)
            
            is_empty_triangle = True
            for j in neighbor_indices:
                if j == current_idx or j == i: continue
                # Sử dụng điểm cuối cùng trong chuỗi để tạo tam giác kiểm tra tính rỗng
                ref_pt = current_pt + ref_vector
                if is_inside_triangle(points_2d[j], current_pt, test_pt, ref_pt):
                    is_empty_triangle = False
                    break
                    
            if is_empty_triangle and diff_angle < min_angle:
                min_angle = diff_angle
                best_next_idx = i
                
        if best_next_idx == -1 or best_next_idx == xmin_idx:
            boundary_indices.append(xmin_idx)
            break
            
        boundary_indices.append(best_next_idx)
        visited.add(best_next_idx)
        ref_vector = points_2d[best_next_idx] - current_pt  # Cập nhật hướng đi tiếp theo
        current_idx = best_next_idx
        
    return boundary_indices, max_search_radius

# Kích hoạt luồng dữ liệu đầu vào
if generate_test:
    t = np.linspace(0, 2*np.pi, num_points_test)
    # Sinh mô hình sao khuyết góc phức tạp để kiểm tra tính đúng đắn của đường ranh giới cụ thể
    r = 100 + 35 * np.sin(5*t) 
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = np.random.uniform(5, 30, num_points_test)
    st.session_state.survey_df = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
elif uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_ext = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_ext == 'txt':
            st.session_state.survey_df = parse_txt_large(file_bytes)
        elif file_ext == 'dxf':
            stream = io.StringIO(file_bytes.decode('utf-8', errors='ignore'))
            doc = ezdxf.read(stream)
            points_list = [p.dxf.location for p in doc.modelspace().query('POINT')]
            if points_list:
                st.session_state.survey_df = pd.DataFrame(points_list, columns=['X', 'Y', 'Z'])
    except Exception as e:
        st.sidebar.error(f"Lỗi: {e}")

# Xử lý kết quả hình học đầu ra
if st.session_state.survey_df is not None:
    df = st.session_state.survey_df
    pts_2d = df[['X', 'Y']].to_numpy()
    pts_3d = df[['X', 'Y', 'Z']].to_numpy()
    
    tri = Delaunay(pts_2d)
    boundary_indices, max_r = compute_correct_boundary(pts_2d)
    
    df_boundary = df.iloc[boundary_indices].copy()
    # Đánh số thứ tự các điểm thuộc đường chu vi
    df_boundary.insert(0, 'STT_RanhGioi', range(1, len(df_boundary) + 1))
    
    st.success(f"✅ Đã phân tích đường bao hoàn chỉnh. Tổng số điểm biên: {len(df_boundary)-1} điểm.")
    
    # Render Đồ họa Preview
    st.subheader("📈 Bản đồ kiểm tra đường ranh giới khép kín")
    st.line_chart(df_boundary, x='X', y='Y')
    
    # Khu vực xuất tệp tin kép (CSV + DXF)
    st.markdown("---")
    st.subheader("📥 Tải về kết quả xử lý")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 1. Tệp bảng tính CSV")
        st.write("Chứa danh sách tọa độ sắp xếp tuần tự theo đường chu vi cụ thể:")
        csv_buffer = io.StringIO()
        df_boundary.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Tải xuống CSV tọa độ đường bao",
            data=csv_buffer.getvalue(),
            file_name="danh_sach_toa_do_ranh_gioi.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.dataframe(df_boundary.head(10), height=180)
        
    with col2:
        st.markdown("### 📐 2. Bản vẽ AutoCAD DXF sửa lỗi")
        st.write("Bản vẽ hoàn chỉnh chứa đường bao khép kín liên tục cùng mạng lưới lớp:")
        
        doc_out = ezdxf.new("R2010")
        msp_out = doc_out.modelspace()
        doc_out.layers.new(name="TRAC_DIA_DIEM", dxfattribs={"color": 2})
        doc_out.layers.new(name="TRAC_DIA_LUOI_TIN", dxfattribs={"color": 8})
        doc_out.layers.new(name="TRAC_DIA_RANH_GIOI", dxfattribs={"color": 1})
        
        # Ghi lớp Điểm
        for p in pts_3d:
            msp_out.add_point((p[0], p[1], p[2]), dxfattribs={"layer": "TRAC_DIA_DIEM"})
        # Ghi lớp mạng lưới TIN
        for simplex in tri.simplices:
            p1, p2, p3 = pts_3d[simplex[0]], pts_3d[simplex[1]], pts_3d[simplex[2]]
            msp_out.add_line(p1, p2, dxfattribs={"layer": "TRAC_DIA_LUOI_TIN"})
            msp_out.add_line(p2, p3, dxfattribs={"layer": "TRAC_DIA_LUOI_TIN"})
            msp_out.add_line(p3, p1, dxfattribs={"layer": "TRAC_DIA_LUOI_TIN"})
        # Ghi lớp Đường bao hoàn chỉnh (Sử dụng danh sách chỉ mục topo đã sửa hướng quét)
        msp_out.add_polyline3d([pts_3d[i] for i in boundary_indices], dxfattribs={"closed": True, "layer": "TRAC_DIA_RANH_GIOI"})
        
        dxf_buffer = io.StringIO()
        doc_out.write(dxf_buffer)
        st.download_button(
            label="📥 Tải xuống DXF đường bao hoàn chỉnh",
            data=dxf_buffer.getvalue(),
            file_name="ban_ve_ranh_gioi_hoan_chinh.dxf",
            mime="application/dxf",
            use_container_width=True
        )
