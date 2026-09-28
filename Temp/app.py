import streamlit as pd_st
import streamlit as st
import pandas as pd
import ezdxf
import io

st.set_page_config(page_title="Xử lý dữ liệu Hình học", layout="wide")

st.title("📐 Ứng dụng Xử lý Dữ liệu Hình học (DXF / TXT)")
st.write("Tải lên file dữ liệu hình học, tùy chỉnh các trục tọa độ và loại bỏ các điểm trùng lặp tự động.")

# Hàm xử lý đọc file TXT/CSV thông thường
def parse_txt(file):
    # Đọc thử một vài dòng để đoán ký tự phân tách (Delimiter)
    sample = file.read(2048).decode("utf-8")
    file.seek(0)
    
    delimiter = ","
    if "\t" in sample:
        delimiter = "\t"
    elif ";" in sample:
        delimiter = ";"
    elif " " in sample and "," not in sample:
        delimiter = r"\s+" # Phân tách bằng khoảng trắng
        
    try:
        df = pd.read_csv(file, sep=delimiter, engine='python')
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file TXT: {e}")
        return None

# Hàm trích xuất dữ liệu tọa độ từ các thực thể POINT hoặc LWPOLYLINE trong file DXF
def parse_dxf(file):
    try:
        # Đọc dữ liệu binary từ file upload
        bytes_data = file.read()
        doc = ezdxf.readstr(bytes_data.decode('utf-8', errors='ignore'))
        msp = doc.modelspace()
        
        points_list = []
        
        # Lấy tất cả các đối tượng POINT
        for entity in msp.query('POINT'):
            p = entity.dxf.location
            points_list.append({'X': p.x, 'Y': p.y, 'Z': p.z})
            
        # Lấy thêm tọa độ đỉnh của LWPOLYLINE (nếu có)
        for entity in msp.query('LWPOLYLINE'):
            for vertex in entity.vertices():
                # LWPOLYLINE mặc định chỉ có X, Y, cao độ Z nằm ở thuộc tính elevation
                z_coord = entity.dxf.elevation if entity.dxf.hasattr('elevation') else 0.0
                points_list.append({'X': vertex[0], 'Y': vertex[1], 'Z': z_coord})
                
        if not points_list:
            st.warning("Không tìm thấy thực thể POINT hoặc LWPOLYLINE nào trong file DXF.")
            return None
            
        return pd.DataFrame(points_list)
    except Exception as e:
        st.error(f"Lỗi khi đọc file DXF: {e}")
        return None

# ----------------- GIAO DIỆN TẢI FILE -----------------
uploaded_file = st.file_uploader("Tải lên file dữ liệu (Định dạng .dxf hoặc .txt)", type=["dxf", "txt"])

if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    df_raw = None
    
    with st.spinner("Đang đọc tệp tin..."):
        if file_ext == 'dxf':
            df_raw = parse_dxf(uploaded_file)
        elif file_ext == 'txt':
            df_raw = parse_txt(uploaded_file)
            
    if df_raw is not None:
        st.success(f"Tải file thành công! Tìm thấy {len(df_raw)} hàng dữ liệu gốc.")
        
        # Hiển thị dữ liệu xem trước ban đầu
        st.subheader("👀 Xem trước dữ liệu thô")
        st.dataframe(df_raw.head(10))
        
        # ----------------- CẤU HÌNH TRỤC TỌA ĐỘ -----------------
        st.subheader("⚙️ Bản đồ hóa các cột dữ liệu")
        columns = list(df_raw.columns)
        
        col1, col2 = st.columns(2)
        with col1:
            # Tự động gợi ý cột dựa trên ký tự định danh phổ biến
            def_x = columns.index('X') if 'X' in columns else (columns.index('x') if 'x' in columns else 0)
            def_y = columns.index('Y') if 'Y' in columns else (columns.index('y') if 'y' in columns else min(1, len(columns)-1))
            def_z = columns.index('Z') if 'Z' in columns else (columns.index('z') if 'z' in columns else min(2, len(columns)-1))
            
            col_x = st.selectbox("Chọn cột cho trục X (Hoành độ):", columns, index=def_x)
            col_y = st.selectbox("Chọn cột cho trục Y (Tung độ):", columns, index=def_y)
            col_z = st.selectbox("Chọn cột cho trục Cao độ (Z):", columns, index=def_z)
            
        with col2:
            # Trục Số thứ tự cho phép bỏ qua
            options_stt = ["-- Bỏ qua (Tự động đánh số) --"] + columns
            col_stt = st.selectbox("Chọn cột cho Số thứ tự (STT):", options_stt, index=0)

        # ----------------- XỬ LÝ DỮ LIỆU CHÍNH -----------------
        if col_x == col_y:
            st.error("⚠️ Cột trục X và trục Y không được trùng nhau!")
        else:
            # Tạo DataFrame mới chỉ giữ lại các cột cần thiết được chuẩn hóa tên
            df_processed = pd.DataFrame()
            
            # Xử lý cột STT
            if col_stt == "-- Bỏ qua (Tự động đánh số) --":
                df_processed['STT'] = range(1, len(df_raw) + 1)
            else:
                df_processed['STT'] = df_raw[col_stt]
                
            df_processed['X'] = pd.to_numeric(df_raw[col_x], errors='coerce')
            df_processed['Y'] = pd.to_numeric(df_raw[col_y], errors='coerce')
            df_processed['Z'] = pd.to_numeric(df_raw[col_z], errors='coerce')
            
            # Loại bỏ các hàng bị lỗi ép kiểu số (NaN) ở các trục tọa độ chính
            df_processed = df_processed.dropna(subset=['X', 'Y', 'Z'])
            
            # 1. Đếm số lượng điểm trước khi lọc trùng
            total_before = len(df_processed)
            
            # 2. Tự động loại bỏ các điểm trùng tọa độ X và Y
            df_clean = df_processed.drop_duplicates(subset=['X', 'Y'], keep='first')
            total_after = len(df_clean)
            duplicated_count = total_before - total_after
            
            # Nếu bỏ qua STT ngay từ đầu, ta đánh lại STT liên tục sau khi lọc trùng để danh sách đẹp hơn
            if col_stt == "-- Bỏ qua (Tự động đánh số) --":
                df_clean['STT'] = range(1, len(df_clean) + 1)

            # ----------------- HIỂN THỊ KẾT QUẢ LỌC -----------------
            st.subheader("✨ Kết quả sau khi làm sạch dữ liệu")
            
            if duplicated_count > 0:
                st.info(f"🔄 Hệ thống đã tự động phát hiện và loại bỏ **{duplicated_count}** điểm có tọa độ (X, Y) trùng lặp.")
            else:
                st.success("✅ Tuyệt vời! Không tìm thấy tọa độ trùng lặp nào trong file dữ liệu của bạn.")
                
            st.write(f"Tổng số điểm hiện tại: **{total_after}** điểm.")
            st.dataframe(df_clean)
            
            # Nút tải xuống file kết quả (Định dạng CSV sạch)
            csv_buffer = io.StringIO()
            df_clean.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Tải xuống dữ liệu sạch (.CSV)",
                data=csv_data,
                file_name=f"cleaned_{uploaded_file.name.split('.')[0]}.csv",
                mime="text/csv"
            )
