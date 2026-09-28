import streamlit as st
import ezdxf
import pandas as pd
import numpy as np
import io
from scipy.spatial import Delaunay

# Cấu hình giao diện Streamlit Cloud
st.set_page_config(page_title="TIN & Boundary Trắc Địa App", layout="wide", page_icon="📐")

st.title("📐 Web App Tạo Lưới TIN & Tìm Đường Chu Vi Trắc Địa")
st.markdown("Xử lý tệp dữ liệu lớn (>10.000 dòng). Tự động phân tích cấu trúc cấu phần địa vật và xuất file DXF.")

# Khởi tạo bộ nhớ dữ liệu tạm trong session state
if 'survey_df' not in st.session_state:
    st.session_state.survey_df = None

st.sidebar.header("📁 Tải tệp số liệu trắc địa")
uploaded_file = st.sidebar.file_uploader("Chọn file dữ liệu gốc (TXT hoặc DXF)", type=["txt", "dxf"])

def parse_txt_large(file_bytes):
    """Đọc dữ liệu file TXT trắc địa định dạng X, Y, Z hiệu năng cao bằng Pandas"""
    data = io.BytesIO(file_bytes)
    df = pd.read_csv(data, sep=r'[,\s\t]+', engine='python', header=None)
    
    # Chuẩn hóa nếu có thêm cột tên điểm ở đầu
    if df.shape >= 4:
        df = df.iloc[:, 1:4]
    elif df.shape == 2:
        df = 0.0  # Bổ sung cao độ mặc định nếu chỉ có X, Y
        
    df.columns = ['X', 'Y', 'Z']
    df['X'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
    df['Z'] = pd.to_numeric(df['Z'], errors='coerce').fillna(0.0)
    return df.dropna(subset=['X', 'Y'])

def compute_tin_and_boundary(df):
    """Tính toán lưới tam giác TIN (Delaunay) và thuật toán dò đường chu vi Boundary"""
    points_2d = df[['X', 'Y']].to_numpy()
    points_3d = df[['X', 'Y', 'Z']].to_numpy()
    
    # 1. Tạo lưới tam giác TIN
    tri = Delaunay(points_2d)
    
    # 2. Thuật toán lọc các cạnh ranh giới ngoài cùng (cạnh chỉ thuộc 1 tam giác)
    edges = {}
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted((simplex[i], simplex[(i+1)%3])))
            edges[edge] = edges.get(edge, 0) + 1
            
    boundary_edges = [edge for edge, count in edges.items() if count == 1]
    
    # Xây dựng bản đồ kề (adjacency list) phục vụ dò đường bao liên tục
    adjacency = {}
    for u, v in boundary_edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
        
    # 3. Dò chuỗi điểm liên tục bắt đầu từ Xmin
    xmin_idx = np.argmin(points_2d[:, 0])
    
    boundary_indices = [xmin_idx]
    current = xmin_idx
    # Lấy điểm đầu tiên trong danh sách kết nối kề
    prev = adjacency[current][0]
    
    # Lặp tuần tự qua topo lưới cho tới khi khép kín vòng tròn tại Xmin
    max_iter = len(boundary_edges)
    iterations = 0
    
    while prev != xmin_idx and iterations < max_iter:
        boundary_indices.append(prev)
        next_nodes = adjacency[prev]
        # Tìm điểm kế tiếp chưa trùng với hướng đi ngược lại vừa qua
        next_node = next_nodes[0] if next_nodes[0] != current else next_nodes[1]
        
        current = prev
        prev = next_node
        iterations += 1
        
    boundary_indices.append(xmin_idx) # Ép khép kín
    
    boundary_points = points_3d[boundary_indices]
    
    return tri.simplices, boundary_points

# Xử lý tệp dữ liệu đầu vào
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    with st.spinner("🔄 Hệ thống đang xử lý cấu trúc lưới hình học..."):
        try:
            if file_ext == 'txt':
                st.session_state.survey_df = parse_txt_large(file_bytes)
            elif file_ext == 'dxf':
                stream = io.StringIO(file_bytes.decode('utf-8', errors='ignore'))
                doc = ezdxf.read(stream)
                points_list = [p.dxf.location for p in doc.modelspace().query('POINT')]
                if points_list:
                    st.session_state.survey_df = pd.DataFrame(points_list, columns=['X', 'Y', 'Z'])
                else:
                    st.sidebar.error("Không tìm thấy điểm POINT nào trong file DXF gốc.")
        except Exception as e:
            st.sidebar.error(f"Lỗi phân tích tệp: {e}")

# Tiến hành render và xuất bản vẽ hình học
if st.session_state.survey_df is not None:
    df = st.session_state.survey_df
    points_3d = df[['X', 'Y', 'Z']].to_numpy()
    
    st.success(f"✅ Đã tải và xử lý thành công {len(df):,} điểm dữ liệu trắc địa!")
    
    # Tính toán lưới hình học chuyên sâu
    simplices, boundary_points = compute_tin_and_boundary(df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Bản đồ hiển thị tổng quan ranh giới")
        # Bản đồ Preview nhanh các điểm bao ngoài
        df_boundary = pd.DataFrame(boundary_points, columns=['X', 'Y', 'Z'])
        st.line_chart(df_boundary, x='X', y='Y')
        
    with col2:
        st.subheader("📋 Bảng thống kê thông số lớp hình học")
        st.metric("Tổng số điểm (POINT & TEXT)", f"{len(df):,}")
        st.metric("Số tam giác mặt lưới TIN", f"{len(simplices):,}")
        st.metric("Số điểm thuộc đường chu vi (BOUNDARY)", f"{len(boundary_points)-1:,}")

    st.markdown("---")
    st.subheader("📥 Trích xuất Bản vẽ AutoCAD kỹ thuật (.DXF)")
    
    # Khởi tạo tệp DXF mới phiên bản chuẩn AutoCAD
    doc_out = ezdxf.new("R2010")
    msp_out = doc_out.modelspace()
    
    # 🗂️ 1. Khởi tạo và gán mã màu cho các Layers theo quy chuẩn trắc địa
    doc_out.layers.new(name="TRAC_DIA_DIEM", dxfattribs={"color": 2})   # Màu Vàng (Yellow)
    doc_out.layers.new(name="TRAC_DIA_LUOI_TIN", dxfattribs={"color": 8})# Màu Xám tối (Dark Gray)
    doc_out.layers.new(name="TRAC_DIA_RANH_GIOI", dxfattribs={"color": 1})# Màu Đỏ (Red - Nổi bật)
    
    # 🖊️ 2. Xuất cấu phần: LAYER ĐIỂM (Gồm POINT + TEXT Cao độ Z)
    for p in points_3d:
        # Xuất thực thể POINT 3D
        msp_out.add_point((p[0], p[1], p[2]), dxfattribs={"layer": "TRAC_DIA_DIEM"})
        # Xuất thực thể TEXT hiển thị trị số cao độ Z lên bản vẽ bên cạnh điểm (offset nhẹ sang phải)
        elevation_str = f"{p[2]:.2f}"
        msp_out.add_text(
            elevation_str, 
            dxfattribs={"layer": "TRAC_DIA_DIEM", "height": 0.25}
        ).set_placement((p[0] + 0.2, p[1] + 0.2, p[2]))
        
    # 🖊️ 3. Xuất cấu phần: LAYER LƯỚI TIN (Các cạnh tam giác không giao nhau)
    for simplex in simplices:
        p1, p2, p3 = points_3d[simplex[0]], points_3d[simplex[1]], points_3d[simplex[2]]
        # Vẽ chu vi tam giác rỗng khép kín bằng các đoạn thẳng LINE 3D
        msp_out.add_line((p1[0], p1[1], p1[2]), (p2[0], p2[1], p2[2]), dxfattribs={"layer": "TRAC_DIA_LUOI_TIN"})
        msp_out.add_line((p2[0], p2[1], p2[2]), (p3[0], p3[1], p3[2]), dxfattribs={"layer": "TRAC_DIA_LUOI_TIN"})
        msp_out.add_line((p3[0], p3[1], p3[2]), (p1[0], p1[1], p1[2]), dxfattribs={"layer": "TRAC_DIA_LUOI_TIN"})
        
    # 🖊️ 4. Xuất cấu phần: LAYER ĐƯỜNG BOUNDARY (Đường đa tuyến 3D khép kín)
    # Sử dụng add_polyline3d để bao toàn vẹn cao độ trắc địa thực tế của địa hình thửa
    msp_out.add_polyline3d(
        [(pt[0], pt[1], pt[2]) for pt in boundary_points], 
        dxfattribs={"closed": True, "layer": "TRAC_DIA_RANH_GIOI"}
    )
    
    # Chuyển đổi luồng byte tải về trực tiếp không lưu ổ cứng trung gian
    out_stream = io.StringIO()
    doc_out.write(out_stream)
    dxf_string = out_stream.getvalue()
    
    st.info("Bản vẽ xuất ra đã phân tách hoàn chỉnh 3 Layer: `TRAC_DIA_DIEM`, `TRAC_DIA_LUOI_TIN`, và `TRAC_DIA_RANH_GIOI`.")
    st.download_button(
        label="⚡ Tải xuống file DXF (TIN & Boundary 3D)",
        data=dxf_string,
        file_name="tin_boundary_output.dxf",
        mime="application/dxf",
        use_container_width=True
    )
else:
    st.info("💡 Vui lòng tải file dữ liệu ở thanh bên trái để hệ thống tự động sinh lưới TIN và dò đường bao.")
