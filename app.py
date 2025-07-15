import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd
from scipy.optimize import curve_fit
import io
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="별 분석 도구",
    page_icon="⭐",
    layout="wide"
)

# 타이틀
st.title("⭐ 별 분석 도구 - FITS 파일 분석기")
st.markdown("---")

# 사이드바 설정
st.sidebar.title("🔧 설정")
st.sidebar.markdown("### 파일 업로드")

# 파일 업로드
uploaded_file = st.sidebar.file_uploader(
    "FITS 파일을 선택하세요",
    type=['fits', 'fit'],
    help="별의 스펙트럼 데이터가 포함된 FITS 파일을 업로드하세요"
)

# 메인 함수들
def analyze_fits_file(file):
    """FITS 파일을 분석하여 기본 정보 추출"""
    try:
        # FITS 파일 읽기
        hdul = fits.open(file)
        
        # 헤더 정보 추출
        header = hdul[0].header
        data = hdul[0].data
        
        # 기본 정보 추출
        info = {}
        
        # 일반적인 FITS 헤더 키들
        common_keys = {
            'OBJECT': '천체명',
            'RA': '적경',
            'DEC': '적위',
            'EXPTIME': '노출시간',
            'DATE-OBS': '관측일시',
            'TELESCOPE': '망원경',
            'INSTRUME': '기기',
            'FILTER': '필터',
            'AIRMASS': '대기질량'
        }
        
        for key, description in common_keys.items():
            if key in header:
                info[description] = header[key]
        
        # 좌표 정보가 있으면 변환
        if 'RA' in header and 'DEC' in header:
            try:
                coord = SkyCoord(ra=header['RA']*u.degree, dec=header['DEC']*u.degree)
                info['좌표 (J2000)'] = coord.to_string('hmsdms')
            except:
                pass
        
        hdul.close()
        return info, data, header
    
    except Exception as e:
        st.error(f"파일 분석 중 오류 발생: {str(e)}")
        return None, None, None

def simulate_stellar_distance(magnitude, spectral_type='G'):
    """별의 등급과 분광형을 이용한 거리 추정 시뮬레이션"""
    # 분광형별 절대등급 (대략적 값)
    absolute_magnitudes = {
        'O': -5.0,
        'B': -2.0,
        'A': 1.0,
        'F': 3.0,
        'G': 5.0,
        'K': 7.0,
        'M': 10.0
    }
    
    abs_mag = absolute_magnitudes.get(spectral_type, 5.0)
    
    # 거리 모듈러스 공식: m - M = 5 * log10(d) - 5
    # d = 10^((m - M + 5)/5) 파섹
    distance_pc = 10**((magnitude - abs_mag + 5) / 5)
    distance_ly = distance_pc * 3.26  # 파섹을 광년으로 변환
    
    return distance_pc, distance_ly

def simulate_doppler_effect(wavelength_data, velocity_km_s):
    """도플러 효과 시뮬레이션"""
    c = 299792.458  # 광속 (km/s)
    
    # 도플러 공식: λ_observed = λ_rest * (1 + v/c)
    # v > 0: 적색편이 (멀어짐), v < 0: 청색편이 (다가옴)
    doppler_factor = 1 + velocity_km_s / c
    
    return wavelength_data * doppler_factor

def analyze_spectrum(data):
    """스펙트럼 데이터 분석"""
    if data is None:
        return None
    
    # 1차원 데이터로 변환
    if len(data.shape) > 1:
        spectrum = np.mean(data, axis=0)
    else:
        spectrum = data
    
    # 기본 통계
    stats = {
        '최대값': np.max(spectrum),
        '최소값': np.min(spectrum),
        '평균값': np.mean(spectrum),
        '표준편차': np.std(spectrum),
        '데이터 포인트 수': len(spectrum)
    }
    
    return spectrum, stats

# 메인 앱
if uploaded_file is not None:
    # 파일 분석
    with st.spinner("FITS 파일을 분석하는 중..."):
        info, data, header = analyze_fits_file(uploaded_file)
    
    if info is not None:
        # 두 개의 컬럼으로 나누기
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 기본 정보")
            
            # 기본 정보 표시
            for key, value in info.items():
                st.write(f"**{key}:** {value}")
            
            # 사용자 입력 섹션
            st.subheader("🔍 추가 분석 설정")
            
            # 별의 겉보기 등급 입력
            apparent_magnitude = st.slider(
                "별의 겉보기 등급",
                min_value=-2.0,
                max_value=15.0,
                value=5.0,
                step=0.1,
                help="별의 겉보기 등급을 입력하세요"
            )
            
            # 분광형 선택
            spectral_type = st.selectbox(
                "분광형",
                options=['O', 'B', 'A', 'F', 'G', 'K', 'M'],
                index=4,
                help="별의 분광형을 선택하세요"
            )
            
            # 시선속도 입력
            radial_velocity = st.slider(
                "시선속도 (km/s)",
                min_value=-200.0,
                max_value=200.0,
                value=0.0,
                step=1.0,
                help="양수: 멀어짐 (적색편이), 음수: 다가옴 (청색편이)"
            )
            
            # 거리 계산
            distance_pc, distance_ly = simulate_stellar_distance(apparent_magnitude, spectral_type)
            
            st.subheader("📏 거리 정보")
            st.write(f"**거리 (파섹):** {distance_pc:.2f} pc")
            st.write(f"**거리 (광년):** {distance_ly:.2f} ly")
            
            # 도플러 효과 정보
            st.subheader("🌊 도플러 효과")
            if radial_velocity > 0:
                st.write(f"**적색편이:** {radial_velocity:.1f} km/s")
                st.write("⬆️ 별이 우리로부터 멀어지고 있습니다")
            elif radial_velocity < 0:
                st.write(f"**청색편이:** {abs(radial_velocity):.1f} km/s")
                st.write("⬇️ 별이 우리에게 다가오고 있습니다")
            else:
                st.write("**시선속도:** 0 km/s")
                st.write("↔️ 별의 시선방향 움직임이 없습니다")
        
        with col2:
            st.subheader("📈 스펙트럼 분석")
            
            # 스펙트럼 데이터 분석
            spectrum_result = analyze_spectrum(data)
            
            if spectrum_result is not None:
                spectrum, stats = spectrum_result
                
                # 스펙트럼 플롯
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # 원본 스펙트럼
                wavelength = np.linspace(400, 700, len(spectrum))  # 가시광선 범위 (nm)
                ax1.plot(wavelength, spectrum, 'b-', linewidth=1)
                ax1.set_title('원본 스펙트럼')
                ax1.set_xlabel('파장 (nm)')
                ax1.set_ylabel('강도')
                ax1.grid(True, alpha=0.3)
                
                # 도플러 효과 적용된 스펙트럼
                doppler_wavelength = simulate_doppler_effect(wavelength, radial_velocity)
                ax2.plot(doppler_wavelength, spectrum, 'r-', linewidth=1, label='도플러 효과 적용')
                ax2.plot(wavelength, spectrum, 'b--', alpha=0.5, label='원본')
                ax2.set_title('도플러 효과가 적용된 스펙트럼')
                ax2.set_xlabel('파장 (nm)')
                ax2.set_ylabel('강도')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 스펙트럼 통계 정보
                st.subheader("📊 스펙트럼 통계")
                stats_df = pd.DataFrame(list(stats.items()), columns=['항목', '값'])
                st.dataframe(stats_df, use_container_width=True)
                
                # 도플러 편이 계산
                c = 299792.458  # km/s
                doppler_shift = radial_velocity / c
                
                st.subheader("🔬 도플러 편이 분석")
                st.write(f"**도플러 편이 (Δλ/λ):** {doppler_shift:.6f}")
                
                if radial_velocity != 0:
                    # 특정 파장에서의 편이량 계산 (예: 550nm - 녹색광)
                    reference_wavelength = 550  # nm
                    shifted_wavelength = reference_wavelength * (1 + doppler_shift)
                    wavelength_change = shifted_wavelength - reference_wavelength
                    
                    st.write(f"**기준 파장 (550nm)에서의 편이량:** {wavelength_change:.4f} nm")
                    
                    if radial_velocity > 0:
                        st.write("🔴 적색편이 - 파장이 길어짐")
                    else:
                        st.write("🔵 청색편이 - 파장이 짧아짐")
            
            else:
                st.warning("스펙트럼 데이터를 분석할 수 없습니다.")
    
    else:
        st.error("FITS 파일을 분석할 수 없습니다. 파일이 올바른 형식인지 확인해주세요.")

else:
    # 홈페이지 설명
    st.markdown("""
    ## 🌟 별 분석 도구 사용법
    
    이 앱은 별의 FITS 파일을 업로드하여 다음과 같은 분석을 수행합니다:
    
    ### 📋 주요 기능
    
    1. **기본 정보 추출**
       - 천체명, 좌표, 관측 정보 등
       - FITS 헤더에서 자동 추출
    
    2. **거리 계산**
       - 겉보기 등급과 분광형을 이용한 거리 추정
       - 파섹과 광년 단위로 표시
    
    3. **도플러 효과 분석**
       - 시선속도에 따른 스펙트럼 변화
       - 적색편이/청색편이 시각화
       - 파장 편이량 계산
    
    4. **스펙트럼 분석**
       - 원본 스펙트럼 표시
       - 도플러 효과 적용된 스펙트럼 비교
       - 통계 정보 제공
    
    ### 🚀 시작하기
    
    1. **왼쪽 사이드바**에서 FITS 파일을 업로드하세요
    2. **별의 정보**를 입력하세요 (겉보기 등급, 분광형, 시선속도)
    3. **자동 분석** 결과를 확인하세요
    
    ### 📖 참고 사항
    
    - **분광형**: O, B, A, F, G, K, M 순으로 온도가 낮아집니다
    - **시선속도**: 양수는 멀어짐(적색편이), 음수는 다가옴(청색편이)
    - **거리 계산**: 거리 모듈러스 공식을 사용합니다
    - **도플러 효과**: 상대론적 효과는 고려하지 않습니다
    
    ### 🎯 교육 목적
    
    이 도구는 지구과학 교육을 위해 설계되었으며, 실제 천체 관측 데이터 분석의 기본 원리를 학습할 수 있습니다.
    """)
    
    # 샘플 데이터 정보
    st.markdown("---")
    st.subheader("💡 팁")
    st.info("""
    FITS 파일이 없다면, 다음 사이트에서 샘플 데이터를 다운로드할 수 있습니다:
    - NASA의 MAST Archive
    - ESO Science Archive
    - 각 천문대의 공개 데이터 아카이브
    """)

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    ⭐ 별 분석 도구 | 지구과학2 프로젝트 | Made with Streamlit
</div>
""", unsafe_allow_html=True)
