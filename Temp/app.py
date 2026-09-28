import streamlit as st
import numpy as np
import pandas as pd

# 🛠️ CẤU HÌNH BẮT BUỘC ĐỂ TRÁNH LỖI GRAPHIC TRÊN WEB STREAMLIT CLOUD
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import ezdxf
import io

# Cấu hình giao diện tổng quan trang Web App
st.set_page_config(page_title="Darvis TIN WebApp", layout="wide")
st.title("🗺️ Ứng Dụng Xác Định Đường Bao Chu Vi - Darvis TIN")
st.markdown("Xác định ranh giới ngoài cùng của tập hợp điểm dựa trên giải thuật **Darvis TIN**.")

def parse_txt_data(text_content):
    """Đọc dữ liệu từ file văn bản tọa độ X Y Z thô"""
    points = []
    for line in text_content.strip().split('\n'):
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2]) if len(parts) > 2 else 0.0
                points.append([x, y, z])
            except ValueError:
                continue
    return np.array(points)

def parse_dxf_data(file_bytes):
    """Đọc dữ liệu hình học từ cấu trúc tệp tin CAD (.DXF)"""
    points = []
    try:
        stream = io.StringIO(file_bytes.decode('utf-8', errors='ignore'))
        doc = ezdxf.read(stream)
        msp = doc.modelspace()
        # Ưu tiên quét các đối tượng POINT
        for entity in msp.query('POINT'):
            p = entity.dxf.location
            points.append([p.x, p.y, p.z])
        # Nếu file không có đối tượng POINT, tìm các nhãn TEXT cao độ
        if not points:
            for entity in msp.query('TEXT'):
                p = entity.dxf.insert
                try:
                    z = float(entity.dxf.text)
                except ValueError:
                    z = p.z
                points.append([p.x, p.y, z])
    except Exception as e:
        st.error(f"Lỗi đọc định dạng file DXF: {e}")
    return np.array(points)
def run_darvis_tin_boundary(pts_2d):
    """
    Hiện thực hóa giải thuật Darvis TIN:
    Dựng mạng lưới Delaunay để khóa tập T cục bộ, duyệt tuần tự qua các cạnh ranh giới cô đơn.
    """
    tri = Delaunay(pts_2d)
    edge_counts = {}
    
    # Đếm số lần xuất hiện của từng cạnh đơn trong mạng lưới tam giác
    for simplex in tri.simplices:
        for i in range(3):
            u, v = simplex[i], simplex[(i + 1) % 3]
            edge = tuple(sorted((u, v)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            
    # Lọc ra tập hợp các cạnh ranh giới ngoài cùng (chỉ tham gia vào đúng 1 tam giác)
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    
    # Xây dựng ma trận đường nối kề (Graph Adjacency)
    adj_graph = {}
    for u, v in boundary_edges:
        if u not in adj_graph: adj_graph[u] = []
        if v not in adj_graph: adj_graph[v] = []
        adj_graph[u].append(v)
        adj_graph[v].append(u)
        
    if not adj_graph:
        return []
        
    # Bắt đầu xuất phát từ điểm Xmin có giá trị hoành độ nhỏ nhất
    x_min_idx = np.argmin(pts_2d[:, 0])
    
    boundary_indices = []
    current_node = x_min_idx
    prev_node = None
    
    # Đi tuần tự men theo các cạnh kề cho đến khi đóng vòng về lại Xmin
    while True:
        boundary_indices.append(current_node)
        neighbors = adj_graph.get(current_node, [])
        
        if len(boundary_indices) > 1 and current_node == x_min_idx:
            break
            
        next_node = None
        for n in neighbors:
            if n != prev_node:
                if n == x_min_idx and len(boundary_indices) > 2:
                    next_node = n
                    break
                if n not in boundary_indices:
                    next_node = n
                    break
                    
        if next_node is None:
            if x_min_idx in neighbors and len(boundary_indices) > 2:
                boundary_indices.append(x_min_idx)
            break
            
        prev_node = current_node
        current_node = next_node
        
    return boundary_indices

def get_polygon_properties(coords):
    """Tính chu vi và diện tích phẳng hình học"""
    x = coords[:, 0]
    y = coords[:, 1]
    dx = np.diff(x, append=x)
    dy = np.diff(y, append=y)
    perimeter = np.sum(np.sqrt(dx**2 + dy**2))
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return perimeter, area
def generate_cad_dxf(points, boundary_indices):
    """Đóng gói dữ liệu xuất thành file DXF nhị phân hoàn chỉnh"""
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Thiết lập Layer đặc trưng theo đúng chuẩn kỹ thuật
    doc.layers.new(name='CAODO', color=3)     # Nhãn Text chữ màu Xanh Lá
    doc.layers.new(name='BOUNDARY', color=1)  # Đường Polyline biên màu Đỏ
    
    # Ghi nhãn chữ cao độ Z của điểm lên bản vẽ
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
        
    # Tạo đường đa tuyến khép kín đại diện cho ranh giới Darvis TIN
    if len(boundary_indices) > 2:
        poly_points = [(points[idx][0], points[idx][1]) for idx in boundary_indices]
        # Sử dụng thuộc tính flags=1 trong ezdxf để thông báo đây là một đường khép kín
        msp.add_lwpolyline(points=poly_points, dxfattribs={'layer': 'BOUNDARY', 'flags': 1})
        
    out_stream = io.StringIO()
    doc.write(out_stream)
    return out_stream.getvalue()
# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("⚙️ Nguồn Dữ Liệu Hình Học")
upload_mode = st.sidebar.selectbox("Lựa chọn định dạng file:", ["File tọa độ văn bản (.TXT)", "File bản vẽ CAD (.DXF)"])
uploaded_file = st.sidebar.file_uploader("Tải tệp hình học của bạn lên đây", type=["txt", "dxf"])

# Chuỗi 71 điểm dữ liệu chuẩn của bạn để làm baseline dữ liệu mẫu nền tảng
BASELINE_DATA = """558154.823	1153935.014	0.679
558153.646	1153940.001	0.54
558153.714	1153945.897	0.943
558162.959	1153946.233	0.438
558164.454	1153941.043	0.364
558164.967	1153936.894	0.407
558175.643	1153937.452	0.469
558175.997	1153942.73	0.359
558175.84	1153947.323	0.416
558186.623	1153948.234	0.4
558187.765	1153943.912	0.267
558187.828	1153938.039	0.489
558198.218	1153939.243	0.464
558198.79	1153944.351	0.291
558198.898	1153949.351	0.303
558210.47	1153950.172	0.249
558212.052	1153945.455	-0.065
558212.777	1153940.2	-0.014
558224.025	1153941.162	-0.076
558224.745	1153945.329	-0.234
558225.592	1153951.072	-0.021
558226.25	1153945.936	-0.167
558226.103	1153941.25	-0.118
558235.144	1153941.756	-0.165
558235.336	1153946.902	-0.259
558234.698	1153951.394	-0.184
558242.966	1153952.174	-0.19
558244.191	1153947.242	-0.526
558248.471	1153947.009	-0.378
558247.93	1153942.549	-0.393
558243.668	1153942.149	-0.252
558243.427	1153945.508	-0.653
558249.775	1153942.704	-0.285
558256.373	1153943.142	-0.194
558262.873	1153943.46	-0.25
558272.791	1153943.854	-0.204
558280.225	1153944.5	-0.2
558291.524	1153944.777	-0.091
558300.373	1153945.795	-0.214
558309.886	1153946.296	-0.24
558322.085	1153947.188	0.187
558332.397	1153947.898	0.291
558339.661	1153948.105	0.287
558345.956	1153948.479	0.318
558351.404	1153948.658	0.393
558359.918	1153949.178	0.522
558368.312	1153949.641	0.5
558366.008	1153954.552	0.296
558367.884	1153954.914	0.458
558366.703	1153961.629	0.593
558366.362	1153961.093	0.502
558356.717	1153960.56	0.472
558356.171	1153954.785	0.412
558347.614	1153954.007	0.314
558345.77	1153959.554	0.423
558334.772	1153959.108	0.354
558333.764	1153953.33	0.167
558323.538	1153952.769	0.195
558320.871	1153958.467	0.255
558313.658	1153953.285	0.084
558311.947	1153957.88	0.17
558305.356	1153957.432	-0.128
558298.227	1153957.019	-0.047
558288.745	1153956.246	-0.109
558277.91	1153955.484	-0.073
558266.434	1153954.31	0.017
558250.001	1153952.777	-0.035
558241.464	1153952.073	-0.034
558229.066	1153950.927	0.001
558215.796	1153950.39	0.100"""

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    if "DXF" in upload_mode:
        pts = parse_dxf_data(file_bytes)
    else:
        pts = parse_txt_data(file_bytes.decode("utf-8"))
else:
    st.info("ℹ️ Đang hiển thị dữ liệu trắc địa mẫu. Hãy đăng tải file mới tại Sidebar để thử nghiệm với các cấu trúc hình học khác.")
    pts = parse_txt_data(BASELINE_DATA)

# Thực thi dựng hình và tính toán ranh giới
if len(pts) >= 3:
    pts_2d = pts[:, :2]
    boundary_idx = run_darvis_tin_boundary(pts_2d)
    
    # Tính toán thông số kỹ thuật khu đất
    p_len, p_area = get_polygon_properties(pts_2d[boundary_idx])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📌 Số điểm dữ liệu", len(pts))
    col2.metric("📏 Tổng chiều dài chu vi", f"{p_len:.2f} m")
    col3.metric("📐 Diện tích ranh giới", f"{p_area:.2f} m²")
    
    # Dựng biểu đồ phẳng bằng Matplotlib Backend độc lập
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    # Vẽ các liên kết mạng tam giác mờ làm nền
    tri_mesh = Delaunay(pts_2d)
    ax.triplot(pts_2d[:, 0], pts_2d[:, 1], tri_mesh.simplices, color='#E5E5E5', linewidth=0.7, linestyle='--')
    
    # Điểm dữ liệu trắc địa
    ax.scatter(pts_2d[:, 0], pts_2d[:, 1], color='#0275d8', s=22, zorder=3, label='Điểm khảo sát')
    
    # Ghi text cao độ Z thực tế của điểm
    for p in pts:
        ax.text(p[0] + 0.3, p[1] + 0.3, f"{p[2]:.2f}", fontsize=7, color='#4A4A4A')
        
    # Vẽ ranh giới ngoài cùng khép kín
    if len(boundary_idx) > 0:
        loop_idx = boundary_idx + [boundary_idx[0]]
        ax.plot(pts_2d[loop_idx, 0], pts_2d[loop_idx, 1], color='#d9534f', linewidth=2.2, zorder=4, label='Đường bao Darvis TIN')
        # Điểm bắt đầu Xmin hình ngôi sao vàng
        ax.scatter(pts_2d[boundary_idx[0], 0], pts_2d[boundary_idx[0], 1], color='#f0ad4e', s=120, marker='*', edgecolors='black', zorder=5, label='Điểm chủ khởi phát (Xmin)')
        
    ax.set_title("Mô phỏng đồ họa giải thuật Darvis TIN", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc="lower right")
    ax.set_aspect('equal', 'dataclim')
    
    st.pyplot(fig)
    
    # Xuất file dữ liệu DXF CAD dạng tải xuống trực tiếp
    dxf_output_raw = generate_cad_dxf(pts, boundary_idx)
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Tải Xuất Dữ Liệu")
    st.sidebar.download_button(
        label="📥 Tải file DXF (Darvis TIN)",
        data=dxf_output_raw,
        file_name="darvis_tin_boundary.dxf",
        mime="application/dxf",
        use_container_width=True
    )
else:
    st.error("❌ Số lượng điểm hình học hợp lệ trong file nhỏ hơn 3. Vui lòng kiểm tra lại dữ liệu đầu vào.")
