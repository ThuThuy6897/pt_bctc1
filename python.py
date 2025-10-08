import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from google import genai
from google.genai.errors import APIError
from typing import Dict, Any

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="Phân Tích Hiệu Quả Dự Án Đầu Tư (NPV/IRR)",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Hiệu Quả Dự Án Đầu Tư 📊")
st.markdown("Sử dụng Gemini AI để trích xuất dữ liệu tài chính từ văn bản, tính toán các chỉ số quan trọng (NPV, IRR, PP, DPP) và phân tích chuyên sâu.")
st.divider()

# --- Định nghĩa Schema JSON cho việc Trích xuất dữ liệu (Task 1) ---
# Sử dụng Integer/Number thay vì String để đảm bảo tính toán chính xác
FINANCIAL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "initial_investment": {"type": "NUMBER", "description": "Tổng vốn đầu tư ban đầu (tỷ VNĐ hoặc VNĐ)."},
        "project_life": {"type": "INTEGER", "description": "Vòng đời dự án (số năm)."},
        "annual_revenue": {"type": "NUMBER", "description": "Doanh thu hàng năm (tỷ VNĐ hoặc VNĐ)."},
        "annual_cost": {"type": "NUMBER", "description": "Chi phí hoạt động hàng năm (tỷ VNĐ hoặc VNĐ)."},
        "wacc": {"type": "NUMBER", "description": "Chi phí vốn bình quân gia quyền (WACC) dưới dạng thập phân (ví dụ: 0.13 cho 13%)."},
        "tax_rate": {"type": "NUMBER", "description": "Thuế suất thuế TNDN dưới dạng thập phân (ví dụ: 0.2 cho 20%)."}
    },
    "required": ["initial_investment", "project_life", "annual_revenue", "annual_cost", "wacc", "tax_rate"]
}

# --- Hàm gọi API Gemini với Structured Output (Task 1) ---
def extract_financial_data(document_content: str, api_key: str) -> Dict[str, Any] | None:
    """Sử dụng Gemini để trích xuất dữ liệu tài chính theo cấu trúc JSON."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính. Hãy trích xuất các tham số tài chính sau từ nội dung văn bản dưới đây. 
        Đảm bảo các giá trị được đưa ra dưới dạng số (NUMBER hoặc INTEGER) và các tỷ lệ (WACC, Thuế) phải là số thập phân (ví dụ: 0.13).
        
        Nội dung văn bản:
        ---
        {document_content}
        ---
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FINANCIAL_SCHEMA,
            )
        )
        
        # Xử lý response.text là chuỗi JSON
        return json.loads(response.text)

    except APIError as e:
        st.error(f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Lỗi: AI không trả về dữ liệu đúng định dạng JSON. Vui lòng thử lại hoặc chỉnh sửa nội dung đầu vào.")
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi không xác định: {e}")
        return None

# --- Hàm tính toán Dòng tiền và Chỉ số (Task 2 & 3) ---
# Sử dụng Caching để Tối ưu hiệu suất
@st.cache_data
def calculate_project_metrics(params: Dict[str, Any]):
    """Xây dựng dòng tiền và tính toán NPV, IRR, PP, DPP."""
    
    # 1. Trích xuất tham số
    I0 = params['initial_investment']
    N = params['project_life']
    R = params['annual_revenue']
    C = params['annual_cost']
    WACC = params['wacc']
    T = params['tax_rate']
    
    # 2. Tính toán Dòng tiền hoạt động hàng năm (ACF) - Giả định không có Khấu hao
    EBIT = R - C
    Tax = EBIT * T
    EAT = EBIT - Tax # Lợi nhuận sau thuế
    ACF = EAT # Dòng tiền thuần hàng năm (ròng)
    
    if ACF <= 0:
        st.error("Lợi nhuận sau thuế (ACF) không dương. Không thể thực hiện tính toán hiệu quả dự án.")
        return None, None
        
    # 3. Xây dựng Bảng Dòng tiền
    years = np.arange(0, N + 1)
    
    # Dòng tiền không chiết khấu (Net Cash Flow - NCF)
    NCF = np.full(N + 1, ACF)
    NCF[0] = -I0 # Vốn đầu tư ban đầu
    
    # Hệ số chiết khấu
    discount_factors = 1 / (1 + WACC)**years
    
    # Dòng tiền chiết khấu (Discounted Cash Flow - DCF)
    DCF = NCF * discount_factors
    
    # Dòng tiền lũy kế (Cumulative)
    Cumulative_NCF = np.cumsum(NCF)
    Cumulative_DCF = np.cumsum(DCF)
    
    # Tạo DataFrame cho Dòng tiền (Task 2)
    df_cf = pd.DataFrame({
        'Năm': years,
        'Dòng tiền thuần (NCF)': NCF,
        'Hệ số chiết khấu': discount_factors,
        'Dòng tiền chiết khấu (DCF)': DCF,
        'NCF Lũy kế': Cumulative_NCF,
        'DCF Lũy kế': Cumulative_DCF,
    })
    
    # 4. Tính toán Chỉ số Hiệu quả (Task 3)
    
    # NPV (Net Present Value)
    NPV = np.sum(DCF)
    
    # IRR (Internal Rate of Return)
    try:
        IRR = np.irr(NCF)
    except Exception:
        IRR = np.nan # Không thể tính IRR
        
    # PP (Payback Period - Thời gian hoàn vốn không chiết khấu)
    PP = I0 / ACF
    
    # DPP (Discounted Payback Period - Thời gian hoàn vốn có chiết khấu)
    # Tìm năm đầu tiên mà DCF Lũy kế > 0
    dpp_year = np.where(Cumulative_DCF >= 0)[0]
    if len(dpp_year) > 0:
        year_after = dpp_year[0]
        year_before = year_after - 1
        
        if year_before < 0:
            DPP = 0
        else:
            # Công thức nội suy: DPP = Y_{trước} + |CFD_{trước}| / CFD_{sau}
            CFD_before = Cumulative_DCF[year_before] # Giá trị âm
            CFD_after = DCF[year_after] # Giá trị dương
            DPP = year_before + (abs(CFD_before) / CFD_after)
    else:
        DPP = np.nan # Không hoàn vốn trong vòng đời dự án

    metrics = {
        'NPV': NPV,
        'IRR': IRR,
        'PP': PP,
        'DPP': DPP,
        'I0': I0,
        'N': N,
        'WACC': WACC,
    }
    
    return df_cf, metrics

# --- Hàm gọi AI Phân tích Chỉ số (Task 4) ---
def get_ai_analysis_metrics(metrics: Dict[str, Any], df_cf: pd.DataFrame, api_key: str):
    """Gửi các chỉ số hiệu quả dự án đến Gemini API để nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        # Chuyển đổi dữ liệu sang định dạng dễ đọc cho AI
        metrics_text = "\n".join([f"- {k}: {v:,.2f}" for k, v in metrics.items() if k not in ['I0', 'N', 'WACC']])
        
        prompt = f"""
        Bạn là một chuyên gia phân tích đầu tư và tài chính. Dựa trên các chỉ số hiệu quả dự án sau, hãy đưa ra một đánh giá chuyên sâu và khách quan (khoảng 3-4 đoạn) về tính khả thi và rủi ro của dự án. 
        Đánh giá tập trung vào: 
        1. **NPV**: Dự án có tạo ra giá trị thặng dư không?
        2. **IRR** so với **WACC ({metrics['WACC']*100:.2f}%)**: Dự án có hấp dẫn hơn chi phí vốn không?
        3. **Thời gian hoàn vốn (PP/DPP)** so với **Vòng đời dự án ({metrics['N']} năm)**: Mức độ rủi ro thanh khoản.

        Các chỉ số tài chính cơ bản:
        - Vốn đầu tư ban đầu (I0): {metrics['I0']:,.0f}
        - WACC: {metrics['WACC']*100:.2f}%
        
        Các chỉ số hiệu quả được tính toán (Đơn vị VNĐ):
        {metrics_text}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Giao diện và Logic chính của Ứng dụng ---

# 1. Đầu vào (Mô phỏng nội dung file Word)
st.subheader("1. Tải và Trích xuất Tham số Dự án (Sử dụng AI)")
document_content = st.text_area(
    "Vui lòng dán nội dung tóm tắt của dự án (mô phỏng nội dung file Word) vào đây:",
    """
    Dự án đầu tư 1 dây chuyền sản xuất bánh mì với vốn đầu tư 30.000.000.000 VNĐ. 
    Dự án có vòng đời trong 10 năm, bắt đầu có dòng tiền từ cuối năm thứ 1 của dự án.
    Mỗi năm tạo ra 3.500.000.000 VNĐ doanh thu và chi phí mỗi năm là 2.000.000.000 VNĐ.
    Thuế suất 20%, WACC của doanh nghiệp là 13%.
    """,
    height=250
)

# Nút Trích xuất
col_extract, _ = st.columns([1, 4])
if col_extract.button("1. Trích xuất Dữ liệu Tài chính (AI)", type="primary") or 'extracted_params' in st.session_state:
    api_key = st.secrets.get("GEMINI_API_KEY")
    
    if not api_key:
        st.error("Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng cấu hình Khóa API trong Streamlit Secrets.")
    elif document_content.strip():
        with st.spinner('Đang gửi văn bản và chờ AI trích xuất dữ liệu...'):
            extracted_params = extract_financial_data(document_content, api_key)
            st.session_state['extracted_params'] = extracted_params

if 'extracted_params' in st.session_state and st.session_state['extracted_params'] is not None:
    params = st.session_state['extracted_params']
    st.success("Trích xuất dữ liệu thành công!")

    # Hiển thị tham số đã trích xuất
    st.markdown("#### Tham số đã trích xuất:")
    params_df = pd.DataFrame({
        'Chỉ tiêu': ['Vốn đầu tư ban đầu (I0)', 'Vòng đời dự án (Năm)', 'Doanh thu hàng năm', 'Chi phí hàng năm', 'WACC', 'Thuế suất'],
        'Giá trị': [
            f"{params['initial_investment']:,.0f} VNĐ",
            f"{params['project_life']} năm",
            f"{params['annual_revenue']:,.0f} VNĐ",
            f"{params['annual_cost']:,.0f} VNĐ",
            f"{params['wacc']*100:.2f}%",
            f"{params['tax_rate']*100:.0f}%"
        ]
    })
    st.dataframe(params_df, hide_index=True)
    st.divider()

    # 2 & 3. Tính toán và hiển thị
    try:
        df_cf, metrics = calculate_project_metrics(params)
        
        if df_cf is not None and metrics is not None:
            st.session_state['df_cf'] = df_cf
            st.session_state['metrics'] = metrics
            
            # Hiển thị Chỉ số Hiệu quả Dự án (Task 3)
            st.subheader("3. Các Chỉ số Đánh giá Hiệu quả Dự án")
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("NPV (Giá trị hiện tại ròng)", f"{metrics['NPV']:,.0f} VNĐ", delta="> 0 (Tạo ra giá trị thặng dư)" if metrics['NPV'] > 0 else "< 0 (Phá hủy giá trị)")
            
            # Hiển thị IRR so với WACC
            irr_delta = f"{metrics['IRR']*100:.2f}% so với WACC ({metrics['WACC']*100:.2f}%)"
            irr_color = "inverse" if metrics['IRR'] < metrics['WACC'] else "normal"
            col2.metric("IRR (Tỷ suất sinh lợi nội bộ)", f"{metrics['IRR']*100:.2f}%", delta=irr_delta, delta_color=irr_color)
            
            col3.metric("PP (Thời gian hoàn vốn)", f"{metrics['PP']:.2f} năm")
            col4.metric("DPP (Hoàn vốn có chiết khấu)", f"{metrics['DPP']:.2f} năm")

            st.divider()
            
            # Hiển thị Bảng Dòng tiền (Task 2)
            st.subheader("2. Bảng Dòng tiền của Dự án")
            
            st.dataframe(
                df_cf.style.format({
                    'Dòng tiền thuần (NCF)': '{:,.0f} VNĐ',
                    'Hệ số chiết khấu': '{:.4f}',
                    'Dòng tiền chiết khấu (DCF)': '{:,.0f} VNĐ',
                    'NCF Lũy kế': '{:,.0f} VNĐ',
                    'DCF Lũy kế': '{:,.0f} VNĐ',
                }), 
                use_container_width=True,
                column_config={
                    "Năm": st.column_config.NumberColumn("Năm", format="%d", help="Năm của dự án")
                }
            )
            st.divider()

            # 4. Phân tích AI
            st.subheader("4. Phân tích Chuyên sâu Chỉ số Hiệu quả (AI)")
            if st.button("4. Yêu cầu AI Phân tích Hiệu quả Dự án", key="analyze_metrics"):
                with st.spinner('Đang gửi các chỉ số và chờ Gemini AI phân tích...'):
                    ai_analysis = get_ai_analysis_metrics(metrics, df_cf, api_key)
                    st.session_state['ai_analysis_result'] = ai_analysis
            
            if 'ai_analysis_result' in st.session_state:
                st.markdown("---")
                st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                st.info(st.session_state['ai_analysis_result'])

    except Exception as e:
        st.error(f"Lỗi tính toán: {e}. Vui lòng kiểm tra lại các tham số đầu vào.")

else:
    st.info("Vui lòng dán nội dung dự án và nhấn nút **'1. Trích xuất Dữ liệu Tài chính (AI)'** để bắt đầu.")
