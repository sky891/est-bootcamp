from statsmodels.tsa.stattools import adfuller
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# ✅ 데이터 저장 경로 정의
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)  # ✅ `data` 폴더가 없으면 생성
RESAMPLED_FILE = os.path.join(DATA_DIR, "data_resampled.csv")

def data_preprocessing():
    print('✅ 데이터 전처리 시작...')


    # 프로젝트 루트 디렉토리 기준 경로 설정
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    os.makedirs(DATA_DIR, exist_ok=True)  # ✅ data 폴더가 없으면 생성

    # ✅ 원본 데이터 파일 경로
    DATA_PATH = os.path.join(DATA_DIR, "Japan earthquakes 2001 - 2018.csv")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ 데이터 파일이 없습니다: {DATA_PATH}\n📌 `data/` 폴더에 CSV 파일을 추가해주세요.")

    # ✅ 데이터 불러오기
    data = pd.read_csv(DATA_PATH, encoding="utf-8")
    print(f"✅ 원본 데이터 로드 완료: {DATA_PATH}")
    
    # ✅ 컬럼명 확인 및 정리
    print(f"✅ 원본 데이터 컬럼명: {data.columns.tolist()}")
    data.columns = data.columns.str.strip()  # 공백 제거
    print(f"✅ 정리된 컬럼명: {data.columns.tolist()}")

    # 결측치 확인
    data.isnull().sum()
    data.isnull().mean()*100


    # 결측치 시각화
    plt.figure(figsize=(20,12))
    sns.heatmap(data.isnull(), cmap='viridis', cbar=False)
    plt.show()


    # 1️⃣ time을 datetime으로 변환
    data['time'] = pd.to_datetime(data['time'], errors='coerce')
    
    # 2️⃣ time을 인덱스로 설정 후 정렬
    data.set_index('time', inplace=True)
    data.sort_index(inplace=True)


    print(data)


    # 위도, 경도 결측치 확인
    null_lat = data['latitude'].isnull().sum()
    null_long = data['longitude'].isnull().sum()

    print(f'latitude 결측치 수 : {null_lat}')
    print(f'longitude 결측치 수 : {null_long}')


    # 위도, 경도 범위 확인
    min_lat = data['latitude'].min()
    max_lat = data['latitude'].max()

    min_long = data['longitude'].min()
    max_long = data['longitude'].max()

    print(f'위도 범위 : {min_lat:.2f} ~ {max_lat:.2f}')
    print(f'경도 범위 : {min_long:.2f} ~ {max_long:.2f}')


    # 지진이 일어난 위치를 지도 위에 표시하여 시각화

    # GeoDataFrame으로 변환
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['longitude'], data['latitude']))

    # 지도 설정
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(ax=ax, color='blue', markersize=1)  # 데이터프레임의 점 표시

    # 배경에 지도 타일 추가 (OpenStreetMap Mapnik 사용)
    ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik)

    # 축 범위 설정 (지진이 일어난 지역만 표시)
    ax.set_xlim(min_long, max_long)
    ax.set_ylim(min_lat, max_lat)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Earthquake points in Map')
    plt.show()


    # depth, mag, magType 결측치 확인

    null_dep = data['depth'].isnull().sum()
    null_mag = data['mag'].isnull().sum()
    null_magType = data['magType'].isnull().sum()

    print(f'depth 결측치 수 : {null_dep}')
    print(f'mag 결측치 수 : {null_mag}')
    print(f'magType 결측치 수 : {null_magType}')


    # depth 값의 범위 확인

    depth_min = data['depth'].min()
    depth_max = data['depth'].max()
    print(f'depth 범위 : {depth_min:.2f} ~ {depth_max:.2f}')
    print()

    # 시각화
    plt.figure(figsize=(8, 6))
    plt.hist(data['depth'], bins=70, color='skyblue', edgecolor='black')
    plt.xlabel('Depth')
    plt.ylabel('Frequency')
    plt.title('Distribution of Depth')
    plt.show()

    # 각 구간별 값의 개수 확인
    df_depth = pd.DataFrame(data, columns=['depth'])

    depth_bins = [0, 100, 200, 300, 400, 500, 600, 700]
    depth_labels = ['0 ~ 100', '100 ~ 200', '200 ~ 300', '300 ~ 400', '400 ~ 500', '500 ~ 600', '600 ~ 700']

    df_depth['depth_range'] = pd.cut(df_depth['depth'], bins=depth_bins, labels=depth_labels, right=False)
    depth_range_counts = df_depth['depth_range'].value_counts().sort_index()

    print(depth_range_counts)


    # mag 값의 범위 확인

    mag_min = data['mag'].min()
    mag_max = data['mag'].max()
    print(f'mag 범위 : {mag_min:.2f} ~ {mag_max:.2f}')
    print()

    # 시각화
    plt.figure(figsize=(8, 6))
    plt.hist(data['mag'], bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Mag')
    plt.ylabel('Frequency')
    plt.title('Distribution of Mag')
    plt.show()

    # 각 구간별 값의 개수 확인
    df_mag = pd.DataFrame(data, columns=['mag'])

    mag_bins = [4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0]
    mag_labels = ['4.5 ~ 5.0', '5.0 ~ 5.5', '5.5 ~ 6.0', '6.0 ~ 7.0', '7.0 ~ 8.0', '8.0 ~ 9.0', '9.0 ~ 10.0']

    df_mag['mag_range'] = pd.cut(df_mag['mag'], bins=mag_bins, labels=mag_labels, right=False)
    mag_range_counts = df_mag['mag_range'].value_counts().sort_index()

    print(mag_range_counts)


    # magType value 확인
    print(data['magType'].unique())
    print(data['magType'].value_counts())

    # 분포 시각화
    plt.figure(figsize=(8, 6))
    plt.hist(data['magType'], color='skyblue', edgecolor='black')
    plt.xlabel('magType')
    plt.ylabel('Frequency')
    plt.title('Distribution of magType')
    plt.show()


    null_rms = data['rms'].isnull().sum()
    print(f'rms 결측치 수 : {null_rms}')

    null_ratio_rms = data['rms'].isnull().mean()*100
    print(f'rms 결측치 비율 : {null_ratio_rms:.2f}%')


    # rms 값의 범위 확인

    rms_min = data['rms'].min()
    rms_max = data['rms'].max()
    print(f'rms 범위 : {rms_min:.2f} ~ {rms_max:.2f}')
    print()

    # 분포 시각화
    plt.figure(figsize=(8, 6))
    plt.hist(data['rms'], color='skyblue', edgecolor='black')
    plt.xlabel('rms')
    plt.ylabel('Frequency')
    plt.title('Distribution of RMS')
    plt.show()


    # nst 결측치 확인
    null_nst = data['nst'].isnull().sum()
    print(f'nst 결측치 수 : {null_nst}')

    null_ratio_nst = data['nst'].isnull().mean()*100
    print(f'nst 결측치 비율 : {null_ratio_nst:.2f}%')

    # nst 값의 범위 확인
    nst_min = data['nst'].min()
    nst_max = data['nst'].max()
    print(f'nst 범위 : {nst_min} ~ {nst_max}')
    print()

    # 분포 시각화
    plt.figure(figsize=(8, 6))
    plt.hist(data['nst'], bins = 50, color='skyblue', edgecolor='black')
    plt.xlabel('nst')
    plt.ylabel('Frequency')
    plt.title('Distribution of NST')
    plt.show()


    # magNst 결측치 확인
    null_magNst = data['magNst'].isnull().sum()
    print(f'magNst 결측치 수 : {null_magNst}')

    null_ratio_magNst = data['magNst'].isnull().mean()*100
    print(f'magNst 결측치 비율 : {null_ratio_magNst:.2f}%')

    # magNst 값의 범위 확인
    magNst_min = data['magNst'].min()
    magNst_max = data['magNst'].max()
    print(f'magNst 범위 : {magNst_min} ~ {magNst_max}')
    print()

    # 분포 시각화
    plt.figure(figsize=(8, 6))
    plt.hist(data['magNst'], bins = 50, color='skyblue', edgecolor='black')
    plt.xlabel('magNst')
    plt.ylabel('Frequency')
    plt.title('Distribution of magNst')
    plt.show()


    # gap 결측치 확인
    null_gap = data['gap'].isnull().sum()
    print(f'gap 결측치 수 : {null_gap}')

    null_ratio_gap = data['gap'].isnull().mean()*100
    print(f'gap 결측치 비율 : {null_ratio_gap:.2f}%')

    # gap 값의 범위 확인
    gap_min = data['gap'].min()
    gap_max = data['gap'].max()
    print(f'gap 범위 : {gap_min} ~ {gap_max}')
    print()

    # 분포 시각화
    plt.figure(figsize=(8, 6))
    plt.hist(data['gap'], bins = 50, color='skyblue', edgecolor='black')
    plt.xlabel('gap')
    plt.ylabel('Frequency')
    plt.title('Distribution of gap')
    plt.show()


    # dmin 결측치 확인
    null_dmin = data['dmin'].isnull().sum()
    print(f'dmin 결측치 수 : {null_dmin}')

    null_ratio_dmin = data['dmin'].isnull().mean()*100
    print(f'dmin 결측치 비율 : {null_ratio_dmin:.2f}%')

    # dmin 값의 범위 확인
    dmin_min = data['dmin'].min()
    dmin_max = data['dmin'].max()
    print(f'dmin 범위 : {dmin_min} ~ {dmin_max}')
    print()

    # 분포 시각화
    plt.figure(figsize=(8, 6))
    plt.hist(data['dmin'], bins = 50, color='skyblue', edgecolor='black')
    plt.xlabel('dmin')
    plt.ylabel('Frequency')
    plt.title('Distribution of dmin')
    plt.show()


    # horizontalError 결측치 확인
    null_horizontalError = data['horizontalError'].isnull().sum()
    print(f'horizontalError 결측치 수 : {null_horizontalError}')

    null_ratio_horizontalError = data['horizontalError'].isnull().mean()*100
    print(f'horizontalError 결측치 비율 : {null_ratio_horizontalError:.2f}%')

    # horizontalError 값의 범위 확인
    horizontalError_min = data['horizontalError'].min()
    horizontalError_max = data['horizontalError'].max()
    print(f'horizontalError 범위 : {horizontalError_min} ~ {horizontalError_max}')
    print()

    # 분포 시각화
    plt.figure(figsize=(8, 6))
    plt.hist(data['horizontalError'], bins = 50, color='skyblue', edgecolor='black')
    plt.xlabel('horizontalError')
    plt.ylabel('Frequency')
    plt.title('Distribution of horizontalError')
    plt.show()


    # depthError 결측치 확인
    null_depthError = data['depthError'].isnull().sum()
    print(f'depthError 결측치 수 : {null_depthError}')

    null_ratio_depthError = data['depthError'].isnull().mean()*100
    print(f'depthError 결측치 비율 : {null_ratio_depthError:.2f}%')

    # depthError 값의 범위 확인
    depthError_min = data['depthError'].min()
    depthError_max = data['depthError'].max()
    print(f'depthError 범위 : {depthError_min} ~ {depthError_max}')
    print()

    # 분포 시각화
    plt.figure(figsize=(8, 6))
    plt.hist(data['depthError'], bins = 50, color='skyblue', edgecolor='black')
    plt.xlabel('depthError')
    plt.ylabel('Frequency')
    plt.title('Distribution of depthError')
    plt.show()


    # magError 결측치 확인
    null_magError = data['magError'].isnull().sum()
    print(f'magError 결측치 수 : {null_magError}')

    null_ratio_magError = data['magError'].isnull().mean()*100
    print(f'magError 결측치 비율 : {null_ratio_magError:.2f}%')

    # magError 값의 범위 확인
    magError_min = data['magError'].min()
    magError_max = data['magError'].max()
    print(f'magError 범위 : {magError_min} ~ {magError_max}')
    print()

    # 분포 시각화
    plt.figure(figsize=(8, 6))
    plt.hist(data['magError'], bins = 50, color='skyblue', edgecolor='black')
    plt.xlabel('magError')
    plt.ylabel('Frequency')
    plt.title('Distribution of magError')
    plt.show()


    # id, type 결측치 확인
    null_id = data['id'].isnull().sum()
    null_type = data['type'].isnull().sum()

    print(f'id 결측치 수 : {null_id}')
    print(f'type 결측치 수 : {null_type}')


    # id value 확인
    print(data['id'].unique())
    print(data['id'].value_counts())

    # id 값 중복 여부 확인
    print(data['id'].duplicated().any())


    # type value 확인
    print(data['type'].unique())
    print(data['type'].value_counts())


    # place 결측치 확인

    null_place = data['place'].isnull().sum()
    print(f'place 결측치 수 : {null_place}')


    # place value 확인
    print(data['place'].unique())
    print(data['place'].value_counts())


    data['place']


    # updated, status 결측치 확인

    null_updated = data['updated'].isnull().sum()
    null_status = data['status'].isnull().sum()

    print(f'updated 결측치 수 : {null_updated}')
    print(f'status 결측치 수 : {null_status}')


    data['updated']


    # status value 확인
    print(data['status'].unique())
    print(data['status'].value_counts())


    # net, locationSource, magSource 결측치 확인

    null_net = data['net'].isnull().sum()
    null_locationSource = data['locationSource'].isnull().sum()
    null_magSource= data['magSource'].isnull().sum()

    print(f'net 결측치 수 : {null_net}')
    print(f'locationSource 결측치 수 : {null_locationSource}')
    print(f'magSource 결측치 수 : {null_magSource}')


    # net value 확인
    print(data['net'].unique())
    print(data['net'].value_counts())


    # locationSource value 확인
    print(data['locationSource'].unique())
    print(data['locationSource'].value_counts())


    # magSource value 확인
    print(data['magSource'].unique())
    print(data['magSource'].value_counts())


    # 상관관계를 구할 column 선택
    columns_to_corr = ['latitude', 'longitude', 'depth','mag','rms']
    data_to_corr = data[columns_to_corr]

    # 상관관계 행렬 생성
    corr_matrix = data_to_corr.corr()

    # mag와의 상관관계만 추출
    mag_corr = corr_matrix[['mag']].sort_values(by='mag', ascending=False)

    # 히트맵 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(mag_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation with Magnitude")
    plt.show()


    # 지진 유형 nuclear explosion인 데이터 행 제거
    value_to_remove = 'nuclear explosion'
    data = data[data['type'] != value_to_remove]

    # rms 값이 결측치인 행 제거
    data.dropna(subset=['rms'], axis=0, inplace=True)

    data.info()


    # 학습에 사용하지 않는 column 제거
    data.drop(['place', 'horizontalError', 'depthError','magError', 'dmin','magNst','nst','id','locationSource','magSource','status','type', 'updated','net','gap' ], inplace = True, axis=1)

    data.info()


    # Plate 열 추가
    def label_plate(row):
        lat = row['latitude']
        lon = row['longitude']

        # Eurasian Plate
        if 30 <= lat <= 50 and 100 <= lon <= 140:
            return 'Eurasian Plate'
        # North American Plate
        elif 35 <= lat <= 65 and 140 <= lon <= 180:
            return 'North American Plate'
        # Pacific Plate
        elif 0 <= lat <= 60 and 135 <= lon <= 180:
            return 'Pacific Plate'
        # Philippine Sea Plate
        elif 10 <= lat <= 35 and 120 <= lon <= 145:
            return 'Philippine Sea Plate'
        else:
            return 'Other'

    data['Plate'] = data.apply(label_plate, axis=1)


    # 데이터셋 로드 (예시 데이터)
    data_df = data[['latitude', 'longitude', 'Plate']]
    df = pd.DataFrame(data_df)

    # 데이터프레임을 GeoDataFrame으로 변환
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']))

    # 지역별 색상 지정
    colors = {
        'North American Plate': 'red',
        'Pacific Plate': 'blue',
        'Eurasian Plate': 'purple',
        'Philippine Sea Plate': 'orange',
    }

    # 지도 설정 및 산점도 그리기
    fig, ax = plt.subplots(figsize=(10, 6))

    # 일본 지역만 지도에 표시하도록 설정
    japan_bounds = (122, 146, 24, 46)  # 경도 최소, 최대 / 위도 최소, 최대
    ax.set_xlim(japan_bounds[0], japan_bounds[1])
    ax.set_ylim(japan_bounds[2], japan_bounds[3])

    # 배경에 지도 타일 추가 (OpenStreetMap Mapnik 사용)
    ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik)

    # 산점도 추가
    for region, group in gdf.groupby('Plate'):
        ax.scatter(group.geometry.x, group.geometry.y, label=region, color=colors[region])

    # 그래프 설정
    ax.set_title('일본 지도 위에 위도 및 경도 분포')
    ax.set_xlabel('경도')
    ax.set_ylabel('위도')
    ax.legend()
    plt.show()


    # Region 열 추가
    def label_japan_region2(row):
        lat = row['latitude']  # 위도
        lon = row['longitude']  # 경도

        # 홋카이도 (Hokkaido)
        if 41 <= lat <= 45.5 and 139 <= lon <= 145.5:
            return 'Hokkaido'
        # 간토 (Kanto)
        elif 34 <= lat <= 37.5 and 138 <= lon <= 141:
            return 'Kanto'
        # 주부 (Chubu)
        elif 33 <= lat <= 38 and 136 <= lon <= 138.5:
            return 'Chubu'
        # 간사이 (Kansai)
        elif 32.5 <= lat <= 36 and 135 <= lon <= 136.5:
            return 'Kansai'
        # 시코쿠 (Shikoku)
        elif 33 <= lat <= 34.5 and 132.5 <= lon <= 135:
            return 'Shikoku'
        # 규슈 (Kyushu)
        elif 29 <= lat <= 34 and 129 <= lon <= 132.5:
            return 'Kyushu'
        # 오키나와 (Okinawa)
        elif 24 <= lat <= 28 and 123 <= lon <= 130:
            return 'Okinawa'
        # 도호쿠 (Tohoku)
        elif 37 <= lat <= 41.5 and 139 <= lon <= 142.5:
            return 'Tohoku'
        # 주고쿠 (Chugoku)
        elif 34 <= lat <= 36 and 131 <= lon <= 134.5:
            return 'Chugoku'
        # 킨키 (Kinki)
        elif 34 <= lat <= 35 and 135 <= lon <= 136:
            return 'Kinki'
        else:
            return 'Other'

    data['Region'] = data.apply(label_japan_region2, axis=1)




    # 데이터셋 로드 (예시 데이터)
    data_df = data[['latitude', 'longitude', 'Region']]
    df = pd.DataFrame(data_df)

    # 데이터프레임을 GeoDataFrame으로 변환
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']))

    # 지역별 색상 지정
    colors = {
        'Hokkaido': 'red',
        'Tohoku': 'blue',
        'Kanto': 'purple',
        'Chubu': 'orange',
        'Kinki': 'pink',
        'Kansai': 'cyan',
        'Chugoku': 'yellow',
        'Shikoku': 'brown',
        'Kyushu': 'magenta',
        'Okinawa': 'black',
        'Other': 'gray'
    }

    # 지도 설정 및 산점도 그리기
    fig, ax = plt.subplots(figsize=(10, 6))

    # 일본 지역만 지도에 표시하도록 설정
    japan_bounds = (122, 146, 24, 46)  # 경도 최소, 최대 / 위도 최소, 최대
    ax.set_xlim(japan_bounds[0], japan_bounds[1])
    ax.set_ylim(japan_bounds[2], japan_bounds[3])

    # 배경에 지도 타일 추가 (OpenStreetMap Mapnik 사용)
    ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik)

    # 산점도 추가
    for region, group in gdf.groupby('Region'):
        ax.scatter(group.geometry.x, group.geometry.y, label=region, color=colors[region])

    # 그래프 설정
    ax.set_title('일본 지도 위에 위도 및 경도 분포')
    ax.set_xlabel('경도')
    ax.set_ylabel('위도')
    ax.legend()
    plt.show()


    # 육지와 많이 떨어진 위치에서 발생한 데이터 제거
    value_to_remove = 'Other'
    data = data[data['Region'] != value_to_remove]
    data.info()


    # 정상성 검증
    result = adfuller(data['mag'])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    
    print(type(data.index))
    print(data.index[:5])

    # 범주형 데이터 원-핫 인코딩
    data = pd.get_dummies(data,  columns=['magType', 'Region', 'Plate']).astype(int)
    data.info()
    

    # 데이터를 1시간 단위로 맞춤
    data.index = pd.to_datetime(data.index)
    data.index = data.index.round('h')
    print("데이터프레임 인덱스 고유값 개수:", len(data.index.unique()))
    print("인덱스 값 샘플:", data.index[:5])
    

    # 5️⃣ 데이터 리샘플링 (1시간 단위)
    data_resampled = data.resample('h').mean()
    data_resampled = data_resampled.interpolate(method='linear')
    data_resampled.reset_index(inplace=True)
    

    # ✅ 데이터가 비어있는지 확인
    if data_resampled.empty:
        raise ValueError("🚨 [data_preprocessing.py] 데이터 리샘플링 후 데이터가 비어 있습니다. `data`를 확인하세요.")

    # ✅ 데이터 정보 출력
    print("✅ [data_preprocessing.py] 데이터 리샘플링 완료:")
    print(data_resampled.info())
    


    # 데이터 저장 경로
    DATA_FILE = os.path.join(DATA_DIR, "data.csv")

    # ✅ 데이터 저장
    data_resampled = data_resampled.fillna(0)  # 결측치를 0으로 채움
    data_resampled.to_csv(RESAMPLED_FILE, index=False, date_format="%Y-%m-%d %H:%M:%S")
    
    print("리샘플링 전 데이터 개수:", len(data))
    print("리샘플링 후 데이터 개수:", len(data_resampled))    
    
    

   # ✅ 파일이 정상적으로 저장되었는지 확인
    if os.path.exists(RESAMPLED_FILE):
        print(f"✅ [data_preprocessing.py] 데이터 저장 완료: {RESAMPLED_FILE}")
    else:
        raise FileNotFoundError(f"🚨 [data_preprocessing.py] 데이터 저장 실패: {RESAMPLED_FILE}")
    
if __name__ == '__main__':
    data_preprocessing()
