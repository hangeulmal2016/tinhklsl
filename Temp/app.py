import streamlit as st
import ezdxf
import pandas as pd
import numpy as np
import io
import time
from scipy.spatial import Delaunay, KDTree

st.set_page_config(page_title="TIN & Boundary Trắc Địa App", layout="wide", page_icon="📐")

st.title("📐 Web App Tạo Lưới TIN & Tìm Đường Chu Vi Trắc Địa")
st.markdown("Xử lý tệp dữ liệu lớn (>10.000 dòng). Hệ thống bỏ qua bước lọc trùng theo cấu hình tiền xử lý đầu vào.")

if 'survey_df' not in st.session_state:
    st.session_state.survey_df = None

st.sidebar.header("📁 Tải tệp số liệu trắc địa")
uploaded_file = st.sidebar.file_uploader("Chọn file dữ liệu gốc (TXT hoặc DXF)", type=["txt", "dxf"])
num_points_test = st.sidebar.slider("Hoặc chạy thử với số điểm giả lập", 100, 10000, 10000, step=100)
generate_test = st.sidebar.button("🔄 Sinh dữ liệu chạy thử")

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

def compute_exact_boundary_dynamic_radius(points_2d):
    n_points = len(points_2d)
    xmin_idx = np.argmin(points_2d[:, 0])
    xmax_idx = np.argmax(points_2d[:, 0])
    
    pt_Xmin = points_2d[xmin_idx]
    pt_Xmax = points_2d[xmax_idx]
    
    # Thiết lập bán kính dò tìm lớn nhất bằng khoảng cách Xmin - Xmax
    max_search_radius = float(np.linalg.norm(pt_Xmax - pt_Xmin))
    tree = KDTree(points_2d)
    
    boundary_indices = [xmin_idx]
    current_idx = xmin_idx
    pt_N = np.array([pt_Xmin[0] - (max_search_radius * 0.05), pt_Xmin[1]])
    visited = set()
    initial_radius = max_search_radius * 0.1
    
    for step in range(n_points):
        current_pt = points_2d[current_idx]
        neighbor_indices = tree.query_ball_point(current_pt, initial_radius)
        
        if len(neighbor_indices) < 5:
            neighbor_indices = tree.query_ball_point(current_pt, max_search_radius)
            
        best_next_idx = -1
        min_angle = float('inf')
        
        if len(boundary_indices) == 1:
            ref_vector = pt_N - current_pt
        else:
            ref_vector = points_2d[boundary_indices[-2]] - current_pt
            
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
            angle = np.arccos(np.clip(np.dot(ref_vector, test_vector) / norms, -1.0, 1.0))
            
            is_empty_triangle = True
            for j in neighbor_indices:
                if j == current_idx or j == i: continue
                if is_inside_triangle(points_2d[j], current_pt, test_pt, pt_N if len(boundary_indices)==1 else points_2d[boundary_indices[-2]]):
                    is_empty_triangle = False
                    break
                    
            if is_empty_triangle and angle < min_angle:
                min_angle = angle
                best_next_idx = i
                
        if best_next_idx == -1 or best_next_idx == xmin_idx:
            boundary_indices.append(xmin_idx)
            break
            
        boundary_indices.append(best_next_idx)
        visited.add(best_next_idx)
        current_idx = best_next_idx
        
    return boundary_indices, max_search_radius

# Xử lý sinh dữ liệu thử nghiệm hoặc nhận file tải lên
if generate_test:
    t = np.linspace(0, 2*np.pi, num_points_test)
    r = 100 + 25 * np.sin(4*t) + np.random.normal(0, 1.5, num_points_test)
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = np.random.uniform(5, 45, num_points_test)
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
        st.sidebar.error(f"Lỗi đọc file: {e}")

# Render đồ họa và tạo nút download DXF
if st.session_state.survey_df is not None:
    df = st.session_state.survey_df
    pts_2d = df[['X', 'Y']].to_numpy()
    pts_3d = df[['X', 'Y', 'Z']].to_numpy()
    
    start_time = time.time()
    tri = Delaunay(pts_2d)
    boundary_indices, max_r = compute_exact_boundary_dynamic_radius(pts_2d)
    execution_time = time.time() - start_time
    
    st.success(f"✅ Đã xử lý {len(df):,} điểm.")
    st.metric("⏱️ Thời gian xử lý", f"{execution_time:.4f} giây")
    st.metric("📏 Bán kính dò tìm tối đa (Xmin - Xmax)", f"{max_r:.2f} đơn vị")
    
    df_b = df.iloc[boundary_indices]
    st.subheader("📈 Bản đồ Preview ranh giới cụ thể")
    st.line_chart(df_b, x='X', y='Y')
    
    # Khởi tạo xuất file DXF chuyên dụng 3 Layer
    doc_out = ezdxf.new("R2010")
    msp_out = doc_out.modelspace()
    doc_out.layers.new(name="TRAC_DIA_DIEM", dxfattribs={"color": 2})
    doc_out.layers.new(name="TRAC_DIA_LUOI_TIN", dxfattribs={"color": 8})
    doc_out.layers.new(name="TRAC_DIA_RANH_GIOI", dxfattribs={"color": 1})
    
    for p in pts_3d:
        msp_out.add_point((p[0], p[1], p[2]), dxfattribs={"layer": "TRAC_DIA_DIEM"})
        msp_out.add_text(f"{p[2]:.2f}", dxfattribs={"layer": "TRAC_DIA_DIEM", "height": 0.25}).set_placement((p[0]+0.2, p[1]+0.2, p[2]))
        
    for simplex in tri.simplices:
        p1, p2, p3 = pts_3d[simplex[0]], pts_3d[simplex[1]], pts_3d[simplex[2]]
        msp_out.add_line(p1, p2, dxfattribs={"layer": "TRAC_DIA_LUOI_TIN"})
        msp_out.add_line(p2, p3, dxfattribs={"layer": "TRAC_DIA_LUOI_TIN"})
        msp_out.add_line(p3, p1, dxfattribs={"layer": "TRAC_DIA_LUOI_TIN"})
        
    msp_out.add_polyline3d([pts_3d[i] for i in boundary_indices], dxfattribs={"closed": True, "layer": "TRAC_DIA_RANH_GIOI"})
    
    out_stream = io.StringIO()
    doc_out.write(out_stream)
    
    st.download_button(
        label="📥 Tải xuống bản vẽ DXF (3 Layer)",
        data=out_stream.getvalue(),
        file_name="tin_boundary_exact.dxf",
        mime="application/dxf",
        use_container_width=True
    )
