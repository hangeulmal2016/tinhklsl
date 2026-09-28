import streamlit as st
import ezdxf
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Trắc Địa Cực Trị App", layout="wide", page_icon="📐")

st.title("📐 Ứng Dụng Dựng Hình Cực Trị & Đoạn Thẳng AB")
st.markdown("Xử lý dữ liệu trắc địa lớn, tìm tọa độ giới hạn ($X_{min}, X_{max}$) và xuất file DXF.")

# Khởi tạo bộ nhớ tạm
if 'survey_data' not in st.session_state:
    st.session_state.survey_data = None

st.sidebar.header("📁 Tải tệp dữ liệu")
uploaded_file = st.sidebar.file_uploader("Chọn file TXT hoặc DXF", type=["txt", "dxf"])

def parse_txt_large(file_bytes):
    data = io.BytesIO(file_bytes)
    df = pd.read_csv(data, sep=r'[,\s\t]+', engine='python', header=None)
    # Xử lý linh hoạt số lượng cột dữ liệu đầu vào
    if df.shape[1] >= 4:
        df = df.iloc[:, 1:4]
    elif df.shape[1] == 2:
        df[2] = 0.0 # Bổ sung cột Z nếu file chỉ có X, Y
    df.columns = ['X', 'Y', 'Z']
    df['X'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
    df['Z'] = pd.to_numeric(df['Z'], errors='coerce').fillna(0.0)
    return df.dropna(subset=['X', 'Y'])

# Đọc file dữ liệu đầu vào
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    with st.spinner("🔄 Đang quét dữ liệu hình học..."):
        try:
            if file_ext == 'txt':
                st.session_state.survey_data = parse_txt_large(file_bytes)
            elif file_ext == 'dxf':
                stream = io.StringIO(file_bytes.decode('utf-8', errors='ignore'))
                doc = ezdxf.read(stream)
                points_list = [p.dxf.location for p in doc.modelspace().query('POINT')]
                if points_list:
                    st.session_state.survey_data = pd.DataFrame(points_list, columns=['X', 'Y', 'Z'])
                else:
                    st.sidebar.error("Không tìm thấy điểm POINT nào trong file DXF.")
        except Exception as e:
            st.sidebar.error(f"Lỗi cấu trúc tệp tin: {e}")

# Xử lý tính toán hình học theo phương án mới
if st.session_state.survey_data is not None:
    df = st.session_state.survey_data
    
    # 1. Tìm các điểm cực trị Xmin, Xmax
    idx_min_x = df['X'].idxmin()
    idx_max_x = df['X'].idxmax()
    
    pt_X1 = df.loc[idx_min_x].to_dict() # Điểm X1 (Xmin)
    pt_X2 = df.loc[idx_max_x].to_dict() # Điểm X2 (Xmax)
    
    # 2. Dựng tọa độ điểm A và điểm B theo quy tắc toán học yêu cầu
    pt_A = {'X': pt_X2['X'], 'Y': pt_X1['Y'], 'Z': 0.0}
    pt_B = {'X': pt_X1['X'], 'Y': pt_X2['Y'], 'Z': 0.0}
    
    st.success(f"📊 Phân tích thành công {len(df):,} hàng dữ liệu!")
    
    # Hiển thị thông số tọa độ tính toán
    st.subheader("📍 Tọa độ các điểm mục tiêu xác định")
    geo_data = {
        "Điểm": ["X1 (Xmin)", "X2 (Xmax)", "Điểm A (Xmax, Y_X1)", "Điểm B (Xmin, Y_X2)"],
        "Tọa độ X": [pt_X1['X'], pt_X2['X'], pt_A['X'], pt_B['X']],
        "Tọa độ Y": [pt_X1['Y'], pt_X2['Y'], pt_A['Y'], pt_B['Y']],
        "Cao độ Z": [pt_X1['Z'], pt_X2['Z'], pt_A['Z'], pt_B['Z']]
    }
    st.table(pd.DataFrame(geo_data))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Bản đồ biểu diễn đồ họa trực quan")
        # Chuẩn bị dữ liệu hiển thị hình học trực quan
        chart_df = pd.DataFrame([
            {"Tên": "X1", "X": pt_X1['X'], "Y": pt_X1['Y'], "Loại": "Điểm Cực Trị"},
            {"Tên": "X2", "X": pt_X2['X'], "Y": pt_X2['Y'], "Loại": "Điểm Cực Trị"},
            {"Tên": "Đoạn AB - Đầu A", "X": pt_A['X'], "Y": pt_A['Y'], "Loại": "Đoạn Thẳng AB"},
            {"Tên": "Đoạn AB - Đầu B", "X": pt_B['X'], "Y": pt_B['Y'], "Loại": "Đoạn Thẳng AB"}
        ])
        st.scatter_chart(chart_df, x='X', y='Y', color='Loại', size=200)
        
    with col2:
        st.subheader("📥 Cấu hình xuất tệp DXF")
        
        # Thiết lập bản vẽ CAD mới thông qua ezdxf
        doc_out = ezdxf.new("R2010")
        msp_out = doc_out.modelspace()
        
        # Khởi tạo các Layer phân tách màu sắc trong AutoCAD
        doc_out.layers.new(name="DIEM_CUC_TRI", dxfattribs={"color": 1}) # Màu đỏ
        doc_out.layers.new(name="DOAN_THANG_AB", dxfattribs={"color": 3}) # Màu xanh lá
        
        # Thực hiện vẽ đối tượng hình học vào file DXF
        # 1. Vẽ điểm X1 và X2 dạng TEXT/POINT định vị
        msp_out.add_point((pt_X1['X'], pt_X1['Y'], pt_X1['Z']), dxfattribs={"layer": "DIEM_CUC_TRI"})
        msp_out.add_text("X1", dxfattribs={"layer": "DIEM_CUC_TRI"}).set_placement((pt_X1['X'], pt_X1['Y'], pt_X1['Z']))
        
        msp_out.add_point((pt_X2['X'], pt_X2['Y'], pt_X2['Z']), dxfattribs={"layer": "DIEM_CUC_TRI"})
        msp_out.add_text("X2", dxfattribs={"layer": "DIEM_CUC_TRI"}).set_placement((pt_X2['X'], pt_X2['Y'], pt_X2['Z']))
        
        # 2. Vẽ đoạn thẳng AB nối liền hình học
        msp_out.add_line(
            (pt_A['X'], pt_A['Y'], pt_A['Z']), 
            (pt_B['X'], pt_B['Y'], pt_B['Z']), 
            dxfattribs={"layer": "DOAN_THANG_AB"}
        )
        msp_out.add_text("A", dxfattribs={"layer": "DOAN_THANG_AB"}).set_placement((pt_A['X'], pt_A['Y'], pt_A['Z']))
        msp_out.add_text("B", dxfattribs={"layer": "DOAN_THANG_AB"}).set_placement((pt_B['X'], pt_B['Y'], pt_B['Z']))
        
        # Chuyển dữ liệu CAD thành bộ nhớ đệm luồng tải về máy
        out_stream = io.StringIO()
        doc_out.write(out_stream)
        dxf_string = out_stream.getvalue()
        
        st.write("Bản vẽ DXF sẽ bao gồm:")
        st.markdown("- **Layer DIEM_CUC_TRI**: Điểm X1, X2 kèm text nhãn.")
        st.markdown("- **Layer DOAN_THANG_AB**: Đường nối thẳng từ điểm A đến B.")
        
        st.download_button(
            label="⚡ Tải xuống file DXF hình học",
            data=dxf_string,
            file_name="hinh_hoc_cuc_tri.dxf",
            mime="application/dxf",
            use_container_width=True
        )
else:
    st.info("💡 Vui lòng tải tệp trắc địa (.TXT/.DXF) ở thanh menu trái để bắt đầu thuật toán tìm điểm cực trị và dựng đoạn thẳng AB.")
