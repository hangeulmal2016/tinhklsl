import streamlit as st
import numpy as np
import pandas as pd

# 🛠️ CẤU HÌNH ĐỂ TRÁNH LỖI GRAPHIC TRÊN WEB STREAMLIT CLOUD
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

# Yêu cầu bắt buộc tải file lên
uploaded_file = st.sidebar.file_uploader("Tải tệp hình học của bạn lên đây để tính toán", type=["txt", "dxf"])

# Trạng thái ban đầu khi chưa có file
if uploaded_file is None:
    st.info("👋 Chào mừng bạn đến với công cụ Darvis TIN! Vui lòng tải lên file dữ liệu (.TXT hoặc .DXF) ở thanh công cụ bên trái (Sidebar) để bắt đầu tự động xác định đường bao chu vi.")
else:
    # Chỉ xử lý dữ liệu KHI VÀ CHỈ KHI có file được tải lên
    file_bytes = uploaded_file.read()
    if "DXF" in upload_mode:
        pts = parse_dxf_data(file_bytes)
    else:
        pts = parse_txt_data(file_bytes.decode("utf-8"))

    # Thực thi dựng hình và tính toán ranh giới Darvis TIN
    if len(pts) >= 3:
        pts_2d = pts[:, :2]
        boundary_idx = run_darvis_tin_boundary(pts_2d)
        
        # Tính toán thông số kỹ thuật khu đất
        p_len, p_area = get_polygon_properties(pts_2d[boundary_idx])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📌 Số điểm dữ liệu nhận diện", len(pts))
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
        
        # ĐỒNG BỘ CHUẨN THUỘC TÍNH ĐÃ FIX LỖI TẠI ĐÂY
        ax.set_aspect('equal', adjustable='datalim')
        
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
        st.error("❌ Tệp dữ liệu không chứa đủ cấu trúc hình học hợp lệ (Yêu cầu tối thiểu 3 điểm tọa độ).")
