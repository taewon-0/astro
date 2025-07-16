import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u
import pandas as pd
from scipy.optimize import curve_fit
import io
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="별 보러 갈래?",
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
    type=['fits', 'fit', 'fz', 'fits.fz'],
    help="별의 스펙트럼 데이터가 포함된 FITS 파일을 업로드하세요 (.fits, .fit, .fz, .fits.fz 지원)"
)

# 메인 함수들
def analyze_fits_file(file):
    """FITS 파일을 분석하여 기본 정보 추출"""
    try:
        # FITS 파일 읽기 (압축된 파일도 자동 처리)
        # astropy는 .fits.fz 파일을 자동으로 압축해제합니다
        hdul = fits.open(file)
        
        # 여러 HDU가 있을 수 있으므로 데이터가 있는 HDU 찾기
        header = None
        data = None
        
        for i, hdu in enumerate(hdul):
            if hdu.header is not None:
                if header is None:  # 첫 번째 헤더 저장
                    header = hdu.header
                if hdu.data is not None and data is None:  # 첫 번째 데이터 저장
                    data = hdu.data
                    
        # 기본값 설정
        if header is None:
            header = hdul[0].header
        if data is None:
            data = hdul[0].data
        
        # 기본 정보 추출
        info = {}
        
        # 파일 정보 추가
        info['파일명'] = file.name if hasattr(file, 'name') else "업로드된 파일"
        info['파일 형식'] = "압축된 FITS" if file.name.endswith('.fz') else "FITS" if hasattr(file, 'name') else "FITS"
        
        # HDU 정보 추가
        info['HDU 개수'] = len(hdul)
        if data is not None:
            if len(data.shape) == 1:
                info['데이터 형태'] = f"1차원 ({data.shape[0]} 포인트)"
            elif len(data.shape) == 2:
                info['데이터 형태'] = f"2차원 ({data.shape[0]} × {data.shape[1]})"
            else:
                info['데이터 형태'] = f"{len(data.shape)}차원 {data.shape}"
        
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
            'AIRMASS': '대기질량',
            'OBSERVER': '관측자',
            'SITE': '관측지',
            'CRVAL1': '중심 파장',
            'CDELT1': '파장 간격',
            'NAXIS1': '스펙트럼 픽셀 수',
            'NAXIS2': '공간 픽셀 수'
        }
        
        for key, description in common_keys.items():
            if key in header:
                value = header[key]
                # 값 포맷팅
                if isinstance(value, float):
                    if key in ['RA', 'DEC']:
                        info[description] = f"{value:.6f}도"
                    elif key in ['EXPTIME']:
                        info[description] = f"{value:.1f}초"
                    elif key in ['CRVAL1', 'CDELT1']:
                        info[description] = f"{value:.4f}"
                    else:
                        info[description] = f"{value:.3f}"
                else:
                    info[description] = str(value)
        
        # 좌표 정보가 있으면 변환
        if 'RA' in header and 'DEC' in header:
            try:
                # RA, DEC이 도 단위인지 시간 단위인지 확인
                ra_val = header['RA']
                dec_val = header['DEC']
                
                # 일반적으로 RA는 0-360도 또는 0-24시간
                if ra_val > 24:  # 도 단위
                    coord = SkyCoord(ra=ra_val*u.degree, dec=dec_val*u.degree)
                else:  # 시간 단위일 가능성
                    coord = SkyCoord(ra=ra_val*u.hour, dec=dec_val*u.degree)
                    
                info['좌표 (J2000)'] = coord.to_string('hmsdms')
            except Exception as coord_error:
                info['좌표 변환 오류'] = str(coord_error)
        
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

def calculate_visibility(ra, dec, date_str=None):
    """서울에서의 별 관측 가능성 계산"""
    try:
        # 서울 위치 설정
        seoul = EarthLocation(lat=37.5665*u.deg, lon=126.9780*u.deg, height=38*u.m)
        
        # 날짜 설정 (기본값: 오늘)
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        # 날짜 객체 생성 (UTC+9 시간대 고려)
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # 별의 좌표
        if isinstance(ra, str) and isinstance(dec, str):
            # 문자열 형태의 좌표인 경우
            target = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
        else:
            # 도 단위 좌표인 경우
            target = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        
        # 하루 동안의 시간 배열 (30분 간격)
        times = []
        altitudes = []
        azimuths = []
        
        for hour in np.arange(0, 24, 0.5):
            # 한국 시간 (UTC+9) 고려
            time_point = date_obj + timedelta(hours=hour)
            astro_time = Time(time_point) + 9*u.hour  # UTC+9
            
            # 고도각과 방위각 계산
            altaz = target.transform_to(AltAz(obstime=astro_time, location=seoul))
            
            times.append(hour)
            altitudes.append(altaz.alt.deg)
            azimuths.append(altaz.az.deg)
        
        return times, altitudes, azimuths, target
    
    except Exception as e:
        return None, None, None, None

def get_visibility_info(altitudes, times):
    """관측 가능성 정보 추출"""
    altitudes = np.array(altitudes)
    times = np.array(times)
    
    # 지평선 위에 있는 시간 찾기
    visible_indices = altitudes > 0
    
    if not np.any(visible_indices):
        return "관측 불가능", "없음", "없음", 0, 0
    
    visible_times = times[visible_indices]
    visible_alts = altitudes[visible_indices]
    
    # 최고 고도와 시간
    max_alt_idx = np.argmax(visible_alts)
    best_time = visible_times[max_alt_idx]
    best_altitude = visible_alts[max_alt_idx]
    
    # 관측 시간 범위
    rise_time = visible_times[0]
    set_time = visible_times[-1]
    
    # 시간 포맷팅
    def format_time(hour):
        h = int(hour)
        m = int((hour - h) * 60)
        return f"{h:02d}:{m:02d}"
    
    if best_altitude > 60:
        quality = "매우 좋음"
    elif best_altitude > 30:
        quality = "좋음"
    elif best_altitude > 15:
        quality = "보통"
    else:
        quality = "어려움"
    
    time_range = f"{format_time(rise_time)} - {format_time(set_time)}"
    best_time_str = format_time(best_time)
    
    return quality, time_range, best_time_str, best_altitude, len(visible_times) * 0.5

def get_direction_from_azimuth(azimuth):
    """방위각을 방향으로 변환"""
    directions = ["북", "북동", "동", "남동", "남", "남서", "서", "북서"]
    idx = round(azimuth / 45) % 8
    return directions[idx]

def analyze_spectrum(data):
    """스펙트럼 데이터 분석"""
    if data is None:
        return None
    
    # 데이터 타입과 차원 확인
    original_shape = data.shape
    
    # 다차원 데이터 처리
    if len(data.shape) > 2:
        # 3차원 이상인 경우 첫 번째 슬라이스 사용
        data = data[0] if data.shape[0] < data.shape[-1] else data.reshape(-1, data.shape[-1])
    
    # 2차원 데이터 처리
    if len(data.shape) == 2:
        # 가로가 더 긴 경우 세로 평균, 세로가 더 긴 경우 가로 평균
        if data.shape[1] > data.shape[0]:
            spectrum = np.mean(data, axis=0)
        else:
            spectrum = np.mean(data, axis=1)
    else:
        spectrum = data
    
    # NaN이나 무한대 값 처리
    spectrum = spectrum[~np.isnan(spectrum)]
    spectrum = spectrum[~np.isinf(spectrum)]
    
    if len(spectrum) == 0:
        return None
    
    # 기본 통계
    stats = {
        '원본 데이터 형태': str(original_shape),
        '처리된 스펙트럼 길이': len(spectrum),
        '최대값': np.max(spectrum),
        '최소값': np.min(spectrum),
        '평균값': np.mean(spectrum),
        '중간값': np.median(spectrum),
        '표준편차': np.std(spectrum),
        '데이터 타입': str(spectrum.dtype)
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
            
            # 관측 날짜 선택
            st.subheader("🌃 관측 계획")
            observation_date = st.date_input(
                "관측 희망 날짜",
                value=datetime.now().date(),
                help="서울에서 관측하고 싶은 날짜를 선택하세요"
            )
            
            # 거리 계산
            distance_pc, distance_ly = simulate_stellar_distance(apparent_magnitude, spectral_type)
            
            st.subheader("📏 거리 정보")
            st.write(f"**거리 (파섹):** {distance_pc:.2f} pc")
            st.write(f"**거리 (광년):** {distance_ly:.2f} ly")
            
            # 관측 가능성 분석
            st.subheader("🌃 서울에서의 관측 가능성")
            
            # 좌표 정보 추출
            ra_for_calc = None
            dec_for_calc = None
            
            # info에서 좌표 정보 찾기
            if 'RA' in header and 'DEC' in header:
                ra_for_calc = header['RA']
                dec_for_calc = header['DEC']
            elif '적경' in info and '적위' in info:
                try:
                    ra_str = info['적경'].replace('도', '')
                    dec_str = info['적위'].replace('도', '')
                    ra_for_calc = float(ra_str)
                    dec_for_calc = float(dec_str)
                except:
                    pass
            
            if ra_for_calc is not None and dec_for_calc is not None:
                date_str = observation_date.strftime('%Y-%m-%d')
                times, altitudes, azimuths, target_coord = calculate_visibility(ra_for_calc, dec_for_calc, date_str)
                
                if times is not None:
                    quality, time_range, best_time, best_alt, duration = get_visibility_info(altitudes, times)
                    
                    # 최적 관측 시간의 방위각 찾기
                    best_time_float = float(best_time.split(':')[0]) + float(best_time.split(':')[1])/60
                    best_az_idx = np.argmin(np.abs(np.array(times) - best_time_float))
                    best_direction = get_direction_from_azimuth(azimuths[best_az_idx])
                    
                    # 관측 정보 표시
                    if quality != "관측 불가능":
                        st.success(f"**관측 품질:** {quality}")
                        st.write(f"**관측 가능 시간:** {time_range}")
                        st.write(f"**최적 관측 시간:** {best_time}")
                        st.write(f"**최고 고도:** {best_alt:.1f}도")
                        st.write(f"**최적 관측 방향:** {best_direction}")
                        st.write(f"**관측 지속 시간:** {duration:.1f}시간")
                        
                        # 고도각 그래프
                        fig_alt, ax_alt = plt.subplots(figsize=(10, 4))
                        ax_alt.plot(times, altitudes, 'b-', linewidth=2)
                        ax_alt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='지평선')
                        ax_alt.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='좋은 관측 고도')
                        ax_alt.set_xlabel('시간 (24시간)')
                        ax_alt.set_ylabel('고도각 (도)')
                        ax_alt.set_title(f'{observation_date} - 별의 고도각 변화')
                        ax_alt.grid(True, alpha=0.3)
                        ax_alt.legend()
                        ax_alt.set_xlim(0, 24)
                        ax_alt.set_xticks(range(0, 25, 3))
                        ax_alt.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)], rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig_alt)
                    else:
                        st.error("**관측 불가능:** 이 날짜에는 별이 지평선 위로 올라오지 않습니다.")
                else:
                    st.warning("좌표 정보를 처리할 수 없습니다.")
            else:
                st.info("FITS 파일에서 좌표 정보를 찾을 수 없습니다. 수동으로 입력해보세요.")
                
                # 수동 좌표 입력
                manual_ra = st.number_input("적경 (도)", value=180.0, min_value=0.0, max_value=360.0)
                manual_dec = st.number_input("적위 (도)", value=0.0, min_value=-90.0, max_value=90.0)
                
                if st.button("관측 가능성 계산"):
                    date_str = observation_date.strftime('%Y-%m-%d')
                    times, altitudes, azimuths, target_coord = calculate_visibility(manual_ra, manual_dec, date_str)
                    
                    if times is not None:
                        quality, time_range, best_time, best_alt, duration = get_visibility_info(altitudes, times)
                        
                        if quality != "관측 불가능":
                            st.success(f"**관측 품질:** {quality}")
                            st.write(f"**관측 가능 시간:** {time_range}")
                            st.write(f"**최적 관측 시간:** {best_time}")
                            st.write(f"**최고 고도:** {best_alt:.1f}도")
                        else:
                            st.error("**관측 불가능:** 이 날짜에는 별이 지평선 위로 올라오지 않습니다.")
        
        with col2:
            st.subheader("📈 스펙트럼 분석")
            
            # 스펙트럼 데이터 분석
            spectrum_result = analyze_spectrum(data)
            
            if spectrum_result is not None:
                spectrum, stats = spectrum_result
                
                # 파장 축 생성 (헤더에 정보가 있으면 사용, 없으면 기본값)
                if 'CRVAL1' in header and 'CDELT1' in header:
                    # WCS 정보가 있는 경우
                    start_wave = header['CRVAL1']
                    delta_wave = header['CDELT1']
                    wavelength = start_wave + np.arange(len(spectrum)) * delta_wave
                    wave_unit = "Å" if start_wave > 1000 else "nm"
                else:
                    # 기본 가시광선 범위
                    wavelength = np.linspace(400, 700, len(spectrum))
                    wave_unit = "nm"
                
                # 스펙트럼 플롯
                fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
                
                # 원본 스펙트럼
                ax1.plot(wavelength, spectrum, 'b-', linewidth=1)
                ax1.set_title(f'스펙트럼 - {info.get("천체명", "Unknown")}')
                ax1.set_xlabel(f'파장 ({wave_unit})')
                ax1.set_ylabel('강도')
                ax1.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 스펙트럼 통계 정보
                st.subheader("📊 스펙트럼 통계")
                stats_df = pd.DataFrame(list(stats.items()), columns=['항목', '값'])
                st.dataframe(stats_df, use_container_width=True)
                
                # 스펙트럼 특징 분석
                st.subheader("🔍 스펙트럼 특징")
                
                # 간단한 스펙트럼 분류
                max_intensity_idx = np.argmax(spectrum)
                peak_wavelength = wavelength[max_intensity_idx]
                
                if wave_unit == "Å":
                    peak_nm = peak_wavelength / 10
                else:
                    peak_nm = peak_wavelength
                
                # 색온도 추정 (간단한 방법)
                if peak_nm < 450:
                    color_temp = "> 10000K (매우 뜨거운 별)"
                    star_color = "청백색"
                elif peak_nm < 500:
                    color_temp = "7000-10000K (뜨거운 별)"
                    star_color = "청색"
                elif peak_nm < 550:
                    color_temp = "6000-7000K (중간 온도)"
                    star_color = "백색"
                elif peak_nm < 600:
                    color_temp = "5000-6000K (태양과 비슷)"
                    star_color = "황색"
                else:
                    color_temp = "< 5000K (차가운 별)"
                    star_color = "적색"
                
                st.write(f"**최대 강도 파장:** {peak_wavelength:.1f} {wave_unit}")
                st.write(f"**추정 색온도:** {color_temp}")
                st.write(f"**별의 색상:** {star_color}")
            
            else:
                st.warning("스펙트럼 데이터를 분석할 수 없습니다.")
    
    else:
        st.error("FITS 파일을 분석할 수 없습니다. 파일이 올바른 형식인지 확인해주세요.")

else:
    # 홈페이지 설명
    st.markdown("""
    ## 🌟 별 분석 도구 사용법
    
    이 앱은 별의 FITS 파일을 업로드하여 다음과 같은 분석을 수행합니다:
    """)
    
    # 용어 설명 섹션 추가
    st.subheader("📚 천문학 용어 설명")
    
    with st.expander("⭐ 겉보기 등급 (Apparent Magnitude)"):
        st.markdown("""
        **겉보기 등급**은 지구에서 보는 별의 밝기를 나타냅니다.
        
        - **숫자가 작을수록 더 밝습니다**
        - 1등급 차이 = 약 2.5배 밝기 차이
        - **예시:**
          - 태양: -26.7등급 (매우 밝음)
          - 보름달: -12.6등급
          - 시리우스: -1.5등급 (가장 밝은 별)
          - 북극성: 2.0등급
          - 육안 한계: 약 6등급
        """)
    
    with st.expander("🌈 분광형 (Spectral Type)"):
        st.markdown("""
        **분광형**은 별의 표면 온도와 색깔을 나타냅니다.
        
        **O → B → A → F → G → K → M** 순으로 온도가 낮아집니다.
        
        | 분광형 | 온도 | 색깔 | 예시 |
        |--------|------|------|------|
        | O | 30,000K+ | 청백색 | 민타카 |
        | B | 10,000-30,000K | 청색 | 리겔 |
        | A | 7,500-10,000K | 백색 | 시리우스 |
        | F | 6,000-7,500K | 황백색 | 프로키온 |
        | **G** | 5,200-6,000K | **황색** | **태양** |
        | K | 3,700-5,200K | 주황색 | 아르크투루스 |
        | M | 2,400-3,700K | 적색 | 베텔기우스 |
        """)
    
    with st.expander("📏 거리 단위"):
        st.markdown("""
        **천문학적 거리 단위**들을 알아보세요.
        
        - **파섹 (pc)**: 천문학에서 주로 사용하는 거리 단위
          - 1 파섹 = 3.26 광년
          - 1 파섹 = 약 31조 km
        
        - **광년 (ly)**: 빛이 1년 동안 가는 거리
          - 1 광년 = 약 9.5조 km
          - 가장 가까운 별(프록시마 센타우리): 4.2 광년
        
        - **천문단위 (AU)**: 지구-태양 거리
          - 1 AU = 약 1억 5천만 km
          - 주로 태양계 내 거리 측정에 사용
        """)
    
    with st.expander("🌃 관측 조건"):
        st.markdown("""
        **좋은 별 관측을 위한 조건**
        
        - **고도각**: 지평선으로부터의 각도
          - 30도 이상: 좋은 관측 조건
          - 60도 이상: 매우 좋은 관측 조건
        
        - **방향**: 8방위로 표시
          - 북쪽: 북극성 방향
          - 남쪽: 가장 높이 올라가는 방향 (한국 기준)
        
        - **시간**: 별마다 다름
          - 계절별로 보이는 별자리가 다름
          - 자정 전후가 가장 어두움
        """)
    
    # 기존 주요 기능 설명
    st.subheader("📋 주요 기능")
    
    st.markdown("""
    1. **FITS 파일 지원**
       - 일반 FITS 파일 (.fits, .fit)
       - 압축된 FITS 파일 (.fz, .fits.fz)
       - 다차원 데이터 자동 처리
    
    2. **기본 정보 추출**
       - 천체명, 좌표, 관측 정보 등
       - FITS 헤더에서 자동 추출
       - 파장 정보 (WCS) 자동 인식
    
    3. **거리 계산**
       - 겉보기 등급과 분광형을 이용한 거리 추정
       - 파섹과 광년 단위로 표시
    
    4. **서울에서의 관측 가능성**
       - 선택한 날짜의 별 관측 조건 분석
       - 최적 관측 시간과 방향 제시
       - 고도각 변화 그래프 제공
    
    5. **스펙트럼 분석**
       - 원본 스펙트럼 표시
       - 색온도와 별의 색상 추정
       - 통계 정보 제공
    
    ### 🚀 시작하기
    
    1. **왼쪽 사이드바**에서 FITS 파일을 업로드하세요
       - 일반 FITS 파일: `.fits`, `.fit`
       - 압축된 FITS 파일: `.fz`, `.fits.fz`
    2. **별의 정보**를 입력하세요 (겉보기 등급, 분광형, 관측 날짜)
    3. **자동 분석** 결과를 확인하세요
    
    ### 📖 참고 사항
    
    - **분광형**: O, B, A, F, G, K, M 순으로 온도가 낮아집니다
    - **거리 계산**: 거리 모듈러스 공식을 사용합니다
    - **관측 계획**: 서울 기준 최적 관측 시간을 계산합니다
    - **고도각**: 30도 이상이면 좋은 관측 조건입니다
    
    # 샘플 데이터 정보
    st.markdown("---")
    st.subheader("💡 팁")
    st.info("""
    **FITS 파일이 없다면:**
    - NASA의 MAST Archive에서 샘플 데이터 다운로드
    - ESO Science Archive 이용
    - 각 천문대의 공개 데이터 아카이브 활용
    
    **좋은 관측을 위해:**
    - 달이 없는 밤 선택
    - 도시 외곽의 어두운 곳
    - 맑은 날씨 확인
    """)

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
     별 분석 도구 | 지구과학2 프로젝트 | Made with Streamlit
</div>
""", unsafe_allow_html=True)
