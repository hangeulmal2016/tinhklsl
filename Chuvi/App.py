import streamlit as st
import ezdxf
import pandas as pd
import numpy as np
import io

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Trắc Địa Web App", layout="wide", page_icon="🗺️")

st.title("🗺️ Web App Xử Lý Ranh Giới & Xuất File DXF")
st.markdown("Hỗ trợ tệp dữ liệu lớn (>10.000 dòng). Xử lý tọa độ và cao độ 3D.")

# Khởi tạo session state để lưu trữ dữ liệu giữa các lần render
if 'points_data' not in st.session_state:
    st.session_state.points_data = None
if 'df_display' not in st.session_state:
    st.session_state.df_display = None

# Thanh bên (Sidebar) để tải file
st.sidebar.header("📁 Tải tệp dữ liệu")
uploaded_file = st.sidebar.file_uploader("Chọn file TXT hoặc DXF", type=["txt", "dxf"])

def parse_txt_large(file_bytes):
    """Hàm đọc file TXT dung lượng lớn tối ưu bằng Pandas"""
    # Đọc file dạng chuỗi byte để tránh nghẽn bộ nhớ
    data = io.BytesIO(file_bytes)
    # Tự động nhận diện dấu phân cách (Dấu phẩy, Khoảng trắng, Tab)
    df = pd.read_csv(data, sep=r'[,\s\t]+', engine='python', header=None, names=['X', 'Y', 'Z'])
    
    # Nếu file có 4 cột (ví dụ: Tên_điểm, X, Y, Z), lấy 3 cột cuối
    if df.shape[1] >= 4:
        df = pd.read_csv(data, sep=r'[,\s\t]+', engine='python', header=None)
        df = df.iloc[:, 1:4]
        df.columns = ['X', 'Y', 'Z']
        
    # Ép kiểu dữ liệu số và xóa bỏ các dòng lỗi (NaN)
    df['X'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
    df['Z'] = pd.to_numeric(df['Z'], errors='coerce').fillna(0.0) # Nếu không có Z thì mặc định bằng 0
    df = df.dropna(subset=['X', 'Y'])
    
    return df.to_numpy(), df

# Xử lý khi người dùng tải file lên
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    with st.spinner("🔄 Đang xử lý dữ liệu lớn..."):
        if file_extension == 'txt':
            points, df_display = parse_txt_large(file_bytes)
            st.session_state.points_data = points
            st.session_state.df_display = df_display
        elif file_extension == 'dxf':
            # Đọc file DXF sử dụng ezdxf
            try:
                stream = io.StringIO(file_bytes.decode('utf-8', errors='ignore'))
                doc = ezdxf.read(stream)
                msp = doc.modelspace()
                points_list = []
                # Trích xuất các thực thể POINT từ file DXF đầu vào
                for point in msp.query('POINT'):
                    points_list.append(point.dxf.location)
                if points_list:
                    st.session_state.points_data = np.array(points_list)
                    st.session_state.df_display = pd.DataFrame(points_list, columns=['X', 'Y', 'Z'])
                else:
                    st.sidebar.error("Không tìm thấy thực thể POINT nào trong file DXF.")
            except Exception as e:
                st.sidebar.error(f"Lỗi đọc file DXF: {e}")

# Hiển thị và xử lý đồ họa
if st.session_state.points_data is not None:
    pts = st.session_state.points_data
    df = st.session_state.df_display
    
    st.success(f"✅ Đã tải thành công {len(pts):,} điểm dữ liệu trắc địa!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📍 Biểu diễn đường ranh giới (2D/3D Preview)")
        # Tạo bản đồ phân tán (Scatter Plot) đại diện cho các điểm và đường bao
        # Tự động khép kín đường bao bằng cách nối điểm cuối về điểm đầu
        closed_pts = np.vstack([pts, pts[0]])
        df_closed = pd.DataFrame(closed_pts, columns=['X', 'Y', 'Z'])
        
        # Sử dụng biểu đồ đường của Streamlit (tối ưu hóa hiển thị nhanh)
        st.line_chart(df_closed, x='X', y='Y')
        
    with col2:
        st.subheader("📊 Bảng tọa độ chi tiết")
        st.dataframe(df.head(100), height=300) # Chỉ hiển thị preview 100 dòng đầu để mượt giao diện
        st.caption(f"Hiển thị 100 / {len(pts):,} dòng.")

    # 🛠️ Tạo và xuất file DXF
    st.markdown("---")
    st.subheader("📥 Xuất kết quả đồ họa trắc địa")
    
    # Tạo nút bấm sinh file DXF bằng ezdxf
    doc_out = ezdxf.new("R2010")
    msp_out = doc_out.modelspace()
    
    # 1. Ghi 10.000 điểm trắc địa vào Layer "DIEM_TRAC_DIA"
    for p in pts:
        msp_out.add_point((p[0], p[1], p[2]), dxfattribs={"layer": "DIEM_TRAC_DIA"})
        
    # 2. Tạo đường chu vi ranh giới khép kín vào Layer "RANH_GIOI_THUA_DAT"
    # Dùng add_polyline3d để giữ nguyên cao độ Z cho trắc địa 3D
    msp_out.add_polyline3d([(p[0], p[1], p[2]) for p in pts], dxfattribs={"closed": True, "layer": "RANH_GIOI_THUA_DAT"})
    
    # Chuyển file DXF thành chuỗi byte để tải về không tốn tài nguyên ổ cứng server
    out_stream = io.StringIO()
    doc_out.write(out_stream)
    dxf_string = out_stream.getvalue()
    
    st.download_button(
        label="⚡ Tải xuống file DXF kết quả",
        data=dxf_string,
        file_name="ranh_gioi_output.dxf",
        mime="application/dxf",
        use_container_width=True
    )
else:
    st.info("💡 Vui lòng chọn và tải file dữ liệu ở thanh bên (TXT/DXF) để bắt đầu vẽ đường ranh giới.")
