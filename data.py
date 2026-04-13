import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import random
import time
import traceback
import os
import json
from collections import defaultdict
warnings.filterwarnings('ignore')

# ============== 0. CSV文件路径配置 ==============
CSV_FILES = {
    'pickup_hz': './pickup_hz.csv',
    'pickup_cq': './pickup_cq.csv',
    'pickup_jl': './pickup_jl.csv',
    'pickup_sh': './pickup_sh.csv',
    'pickup_yt': './pickup_yt.csv',
    'pickup_five_cities': './pickup_five_cities.csv',
    'delivery_hz': './delivery_hz.csv',
    'delivery_cq': './delivery_cq.csv',
    'delivery_jl': './delivery_jl.csv',
    'delivery_sh': './delivery_sh.csv',
    'delivery_yt': './delivery_yt.csv',
    'delivery_five_cities': './delivery_five_cities.csv',
    'courier_trajectory': './courier_detailed_trajectory.csv',
    'roads': './roads.csv',
}

# ============== 0.1 数据加载工具 ==============

def load_csv_file(name):
    """加载指定的CSV文件"""
    path = CSV_FILES.get(name)
    if path is None:
        print(f"[load_csv_file] 未知的文件: {name}")
        return None
    try:
        if name == 'roads':
            return pd.read_csv(path, sep='\t')
        else:
            return pd.read_csv(path)
    except Exception as e:
        print(f"[load_csv_file] 读取 {name} 失败: {e}")
        return None

def load_all_csv_files():
    """加载所有CSV文件并返回字典"""
    print("=" * 60)
    print("加载所有CSV数据文件...")
    print("=" * 60)

    data = {}
    for name, path in CSV_FILES.items():
        df = load_csv_file(name)
        if df is not None:
            data[name] = df
            print(f"  ✅ {name}: {len(df):,} 行, {len(df.columns)} 列")
        else:
            print(f"  ❌ {name}: 加载失败")
            data[name] = None

    print(f"\n成功加载 {sum(1 for v in data.values() if v is not None)}/{len(data)} 个文件")
    return data

def get_delivery_data():
    """获取delivery数据（优先使用five_cities版本）"""
    if os.path.exists(CSV_FILES['delivery_five_cities']):
        return pd.read_csv(CSV_FILES['delivery_five_cities'])
    else:
        dfs = []
        for city in ['hz', 'cq', 'jl', 'sh', 'yt']:
            path = CSV_FILES.get(f'delivery_{city}')
            if path and os.path.exists(path):
                dfs.append(pd.read_csv(path))
        if dfs:
            return pd.concat(dfs, ignore_index=True)
    return None

def get_pickup_data():
    """获取pickup数据（优先使用five_cities版本）"""
    if os.path.exists(CSV_FILES['pickup_five_cities']):
        return pd.read_csv(CSV_FILES['pickup_five_cities'])
    else:
        dfs = []
        for city in ['hz', 'cq', 'jl', 'sh', 'yt']:
            path = CSV_FILES.get(f'pickup_{city}')
            if path and os.path.exists(path):
                dfs.append(pd.read_csv(path))
        if dfs:
            return pd.concat(dfs, ignore_index=True)
    return None

def get_roads_data():
    """获取roads数据"""
    path = CSV_FILES.get('roads')
    if path and os.path.exists(path):
        return pd.read_csv(path, sep='\t')
    return None

def get_courier_trajectory_data(max_rows=None):
    """获取快递员轨迹数据"""
    path = CSV_FILES.get('courier_trajectory')
    if path and os.path.exists(path):
        if max_rows:
            return pd.read_csv(path, nrows=max_rows)
        return pd.read_csv(path)
    return None

# ============== 0.2 真实天气特征获取 ==============

class WeatherFeatureProvider:
    """
    真实天气数据获取器 - 使用 Open-Meteo 免费历史天气 API
    
    特性：
    - 按城市中心坐标 + 日期获取历史天气
    - 本地磁盘缓存避免重复请求
    - API 不可用时自动降级为基于城市/月份的统计估算
    - 提取特征：气温、降水、风速、天气类别、交通拥堵估计
    """
    
    # 五城市中心坐标 (WGS84 经纬度)
    CITY_COORDS = {
        '杭州市': (30.27, 120.15),
        '上海市': (31.23, 121.47),
        '重庆市': (29.56, 106.55),
        '吉林市': (43.84, 126.55),
        '烟台市': (37.46, 121.45),
    }
    
    # 城市月份气候统计 (用于 API 不可用时的降级估算)
    # 格式: (平均气温℃, 月均降水mm, 平均风速km/h)
    CITY_CLIMATE = {
        '杭州市': {1:(4,70,10), 2:(6,80,10), 3:(10,130,11), 4:(16,120,11),
                  5:(21,120,10), 6:(25,200,9), 7:(29,150,10), 8:(28,140,9),
                  9:(24,120,9), 10:(18,70,9), 11:(12,60,9), 12:(6,50,10)},
        '上海市': {1:(4,50,13), 2:(5,60,13), 3:(9,90,13), 4:(14,90,12),
                  5:(20,100,11), 6:(24,170,10), 7:(28,130,11), 8:(28,130,11),
                  9:(24,120,11), 10:(19,60,11), 11:(12,50,12), 12:(6,40,12)},
        '重庆市': {1:(8,20,7), 2:(10,25,7), 3:(14,40,7), 4:(19,80,7),
                  5:(22,120,6), 6:(25,170,6), 7:(29,140,6), 8:(30,110,6),
                  9:(24,120,6), 10:(18,80,6), 11:(13,50,6), 12:(9,25,7)},
        '吉林市': {1:(-17,5,10), 2:(-12,8,10), 3:(-3,15,12), 4:(7,25,14),
                  5:(15,50,13), 6:(20,90,11), 7:(23,130,10), 8:(21,110,9),
                  9:(14,55,10), 10:(6,30,11), 11:(-4,15,11), 12:(-14,7,10)},
        '烟台市': {1:(-1,15,14), 2:(1,15,13), 3:(5,20,14), 4:(11,35,14),
                  5:(17,50,12), 6:(21,80,10), 7:(25,130,9), 8:(25,120,9),
                  9:(21,55,10), 10:(15,35,11), 11:(8,25,13), 12:(2,15,14)},
    }
    
    def __init__(self, cache_dir='./weather_cache'):
        self.cache_dir = cache_dir
        self.cache = {}  # 内存缓存 {(city, date_str): weather_dict}
        self._load_disk_cache()
    
    def _load_disk_cache(self):
        """从磁盘加载天气缓存"""
        cache_file = os.path.join(self.cache_dir, 'weather_cache.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                # 将 key 从字符串恢复
                for k, v in raw.items():
                    self.cache[k] = v
                print(f"  [Weather] 从缓存加载 {len(self.cache)} 条天气记录")
            except Exception:
                self.cache = {}
    
    def _save_disk_cache(self):
        """保存天气缓存到磁盘"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, 'weather_cache.json')
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False)
        except Exception as e:
            print(f"  [Weather] 缓存保存失败: {e}")
    
    def _fetch_from_api(self, lat, lng, date_str):
        """
        从 Open-Meteo 历史天气 API 获取数据
        
        Args:
            lat, lng: WGS84 经纬度
            date_str: 日期字符串 'YYYY-MM-DD'
        
        Returns:
            dict: 天气特征 或 None
        """
        try:
            import requests
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                'latitude': lat,
                'longitude': lng,
                'start_date': date_str,
                'end_date': date_str,
                'hourly': 'temperature_2m,precipitation,wind_speed_10m,weather_code',
                'timezone': 'Asia/Shanghai'
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            hourly = data.get('hourly', {})
            
            if not hourly or 'temperature_2m' not in hourly:
                return None
            
            temps = [t for t in hourly['temperature_2m'] if t is not None]
            precips = [p for p in hourly['precipitation'] if p is not None]
            winds = [w for w in hourly['wind_speed_10m'] if w is not None]
            codes = [c for c in hourly.get('weather_code', []) if c is not None]
            
            return {
                'temperature': float(np.mean(temps)) if temps else 20.0,
                'temp_min': float(np.min(temps)) if temps else 15.0,
                'temp_max': float(np.max(temps)) if temps else 25.0,
                'precipitation': float(np.sum(precips)) if precips else 0.0,
                'wind_speed': float(np.mean(winds)) if winds else 10.0,
                'weather_code': int(np.median(codes)) if codes else 0,
            }
        except ImportError:
            print("  [Weather] requests 库未安装，使用估算值")
            return None
        except Exception:
            return None
    
    def _estimate_weather(self, city, month, hour=12):
        """
        基于城市/月份气候统计估算天气（API不可用时的降级方案）
        
        添加合理的随机波动以避免所有样本特征相同
        """
        climate = self.CITY_CLIMATE.get(city)
        if climate is None:
            # 未知城市，使用杭州作为默认
            climate = self.CITY_CLIMATE['杭州市']
        
        base_temp, base_precip_monthly, base_wind = climate.get(month, (20, 60, 10))
        
        # 日均降水 = 月降水 / 30, 加随机波动
        daily_precip_base = base_precip_monthly / 30.0
        
        # 添加合理随机波动
        np.random.seed(None)  # 确保每次不同
        temp_noise = np.random.normal(0, 3)  # ±3℃ 波动
        precip_factor = np.random.exponential(1.0)  # 降水呈指数分布
        wind_noise = np.random.normal(0, 3)  # ±3km/h 波动
        
        temperature = base_temp + temp_noise
        precipitation = max(0, daily_precip_base * precip_factor)
        wind_speed = max(0, base_wind + wind_noise)
        
        # 日内温度变化
        if hour is not None:
            hour_factor = -np.cos(2 * np.pi * (hour - 14) / 24) * 4  # 14点最高
            temperature += hour_factor
        
        # 根据降水和温度确定天气代码
        if precipitation > 5:
            weather_code = 63  # 中雨
        elif precipitation > 1:
            weather_code = 51  # 小雨
        elif precipitation > 0.1:
            weather_code = 3   # 多云
        else:
            weather_code = 0 if np.random.random() > 0.3 else 2  # 晴/少云
        
        return {
            'temperature': float(temperature),
            'temp_min': float(temperature - 4),
            'temp_max': float(temperature + 4),
            'precipitation': float(precipitation),
            'wind_speed': float(wind_speed),
            'weather_code': int(weather_code),
        }
    
    @staticmethod
    def weather_code_to_category(code):
        """
        WMO 天气代码 → 类别字符串
        参考: https://open-meteo.com/en/docs
        """
        if code <= 1:
            return 'sunny'
        elif code <= 3:
            return 'cloudy'
        elif code <= 49:
            return 'foggy'
        elif code <= 69:
            return 'rainy'
        elif code <= 79:
            return 'snowy'
        elif code <= 99:
            return 'stormy'
        else:
            return 'cloudy'
    
    @staticmethod
    def estimate_traffic(hour, precipitation, city=None):
        """
        基于时段和天气估算交通拥堵等级
        
        逻辑：
        - 早晚高峰 (7-9, 17-19) → 基础拥堵高
        - 降水 > 1mm → 拥堵加剧
        - 夜间 (22-6) → 拥堵低
        """
        # 基础小时拥堵分数 (0-10)
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base = 7.5
        elif 9 < hour < 17:
            base = 5.0
        elif 6 <= hour < 7 or 19 < hour <= 22:
            base = 4.0
        else:
            base = 1.5
        
        # 天气修正
        if precipitation > 5:
            base += 2.5  # 中雨以上
        elif precipitation > 1:
            base += 1.5  # 小雨
        elif precipitation > 0.1:
            base += 0.5  # 毛毛雨
        
        # 加随机波动
        base += np.random.normal(0, 0.8)
        base = np.clip(base, 0, 10)
        
        # 转为等级
        if base >= 6.5:
            return 'high'
        elif base >= 3.5:
            return 'medium'
        else:
            return 'low'
    
    def get_weather_for_dataframe(self, df, city_col='from_city_name',
                                   time_col='pickup_time', use_api=True, api_sample_limit=50):
        """
        为 DataFrame 批量获取天气特征
        
        策略：
        1. 按 (城市, 日期) 分组获取天气（同城同日天气相同）
        2. 优先使用 API, 达到请求上限后降级为估算
        3. 本地缓存命中的不计入 API 配额
        
        Args:
            df: 含城市和时间列的 DataFrame
            city_col: 城市列名
            time_col: 时间列名 (已解析为 datetime)
            use_api: 是否尝试调用 API
            api_sample_limit: API 最大请求次数
        
        Returns:
            DataFrame: 新增天气特征列后的 df
        """
        print("\n  [Weather] 获取真实天气特征...")
        
        # 提取日期和月份
        has_time = time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col])
        if has_time:
            df['_weather_month'] = df[time_col].dt.month
            df['_weather_hour'] = df[time_col].dt.hour
            # LaDe数据集的日期没有年份，用固定年份构造日期字符串
            df['_weather_date'] = df[time_col].dt.strftime('2023-%m-%d')
        else:
            df['_weather_month'] = 6  # 默认6月
            df['_weather_hour'] = 12
            df['_weather_date'] = '2023-06-15'
        
        has_city = city_col in df.columns
        if has_city:
            df['_weather_city'] = df[city_col]
        else:
            df['_weather_city'] = '杭州市'  # 默认
        
        # 按 (城市, 日期) 去重获取天气
        unique_keys = df[['_weather_city', '_weather_date']].drop_duplicates()
        print(f"  [Weather] 需获取 {len(unique_keys)} 个城市-日期组合的天气")
        
        weather_lookup = {}  # {(city, date): weather_dict}
        api_count = 0
        cache_hits = 0
        api_hits = 0
        estimate_count = 0
        
        for _, row in unique_keys.iterrows():
            city = row['_weather_city']
            date_str = row['_weather_date']
            cache_key = f"{city}_{date_str}"
            
            # 1. 检查缓存
            if cache_key in self.cache:
                weather_lookup[(city, date_str)] = self.cache[cache_key]
                cache_hits += 1
                continue
            
            # 2. 尝试 API
            weather = None
            if use_api and api_count < api_sample_limit:
                coords = self.CITY_COORDS.get(city)
                if coords:
                    lat, lng = coords
                    weather = self._fetch_from_api(lat, lng, date_str)
                    if weather:
                        api_count += 1
                        api_hits += 1
                        self.cache[cache_key] = weather
            
            # 3. 降级估算
            if weather is None:
                month = int(date_str.split('-')[1]) if '-' in date_str else 6
                weather = self._estimate_weather(city, month)
                self.cache[cache_key] = weather
                estimate_count += 1
            
            weather_lookup[(city, date_str)] = weather
        
        print(f"  [Weather] 缓存命中: {cache_hits}, API获取: {api_hits}, 统计估算: {estimate_count}")
        
        # 保存缓存
        if api_hits > 0:
            self._save_disk_cache()
        
        # 4. 映射回 DataFrame
        temperatures = []
        precipitations = []
        wind_speeds = []
        weather_categories = []
        traffic_levels = []
        temp_ranges = []
        
        for _, row in df.iterrows():
            city = row['_weather_city']
            date_str = row['_weather_date']
            hour = row['_weather_hour']
            
            w = weather_lookup.get((city, date_str))
            if w is None:
                month = row['_weather_month']
                w = self._estimate_weather(city, month, hour)
            
            temperatures.append(w['temperature'])
            precipitations.append(w['precipitation'])
            wind_speeds.append(w['wind_speed'])
            temp_ranges.append(w['temp_max'] - w['temp_min'])
            weather_categories.append(self.weather_code_to_category(w['weather_code']))
            traffic_levels.append(self.estimate_traffic(hour, w['precipitation'], city))
        
        df['temperature'] = temperatures
        df['precipitation'] = precipitations
        df['wind_speed'] = wind_speeds
        df['temp_range'] = temp_ranges
        df['weather'] = weather_categories
        df['traffic'] = traffic_levels
        
        # 清理临时列
        df.drop(columns=['_weather_month', '_weather_hour', '_weather_date', '_weather_city'],
                inplace=True, errors='ignore')
        
        # 统计
        print(f"  [Weather] 天气分布: {pd.Series(weather_categories).value_counts().to_dict()}")
        print(f"  [Weather] 交通分布: {pd.Series(traffic_levels).value_counts().to_dict()}")
        print(f"  [Weather] 气温范围: {np.min(temperatures):.1f}℃ ~ {np.max(temperatures):.1f}℃")
        print(f"  [Weather] 降水范围: {np.min(precipitations):.1f} ~ {np.max(precipitations):.1f} mm")
        
        return df


# ============== 0.3 道路距离加权系数 ==============
ROAD_TYPE_WEIGHTS = {
    'motorway': 1.0,
    'primary': 1.2,
    'secondary': 1.5,
    'tertiary': 1.8,
    'residential': 2.0,
    'service': 2.2,
    'unclassified': 1.5,
    'footway': 2.5,
    'pedestrian': 2.5,
    'track': 2.5,
    'path': 2.5,
    'trunk': 1.1,
    'motorway_link': 1.1,
    'trunk_link': 1.1
}


def calculate_road_distance(start_lng, start_lat, end_lng, end_lat, roads_df):
    """
    计算基于道路类型的加权距离
    输入坐标为EPSG:3857（米）
    """
    if roads_df is None or len(roads_df) == 0:
        return None
    
    def meters_to_latlon(mx, my):
        lon = mx / 111319.49079327357
        lat = (2 * np.arctan(np.exp(my / 6378137.0)) - np.pi / 2) * 180 / np.pi
        return lon, lat
    
    straight_dist = haversine_distance(start_lng, start_lat, end_lng, end_lat)
    
    city = None
    for c in ['上海市', '杭州市', '重庆市', '烟台市', '吉林市']:
        if c in str(start_lng) or c in str(start_lat):
            city = c
            break
    
    if city and 'city' in roads_df.columns:
        city_roads = roads_df[roads_df['city'] == city]
        if len(city_roads) > 0:
            avg_weight = city_roads['fclass'].map(
                lambda x: ROAD_TYPE_WEIGHTS.get(x, 1.5)
            ).mean()
            return straight_dist * avg_weight
    
    return straight_dist * 1.5


# ============== 1. 数据加载与预处理 ==============

def haversine_distance(lng1, lat1, lng2, lat2):
    """
    使用haversine公式计算两点间距离（公里）
    输入坐标为EPSG:3857投影坐标（米），先转换为经纬度再计算
    """
    # EPSG:3857转EPSG:4326近似转换
    def meters_to_latlon(mx, my):
        lon = mx / 111319.49079327357
        lat = (2 * np.arctan(np.exp(my / 6378137.0)) - np.pi / 2) * 180 / np.pi
        return lon, lat
    
    lon1, lat1 = meters_to_latlon(lng1, lat1)
    lon2, lat2 = meters_to_latlon(lng2, lat2)
    
    # haversine公式
    R = 6371  # 地球半径（公里）
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


class LaDeDataProcessor:
    """LaDe数据集处理器 - 适配delivery_five_cities.csv
    
    改进功能：
    1. 按快递员分组抽样，保持轨迹完整性
    2. 支持序列化数据构建
    3. 添加AOI特征提取
    """
    
    def __init__(self, data_path, max_samples=200000, min_courier_orders=5, max_courier_orders=100):
        self.df = pd.read_csv(data_path)
        print(f"[Data Check] sign_time缺失率: {self.df['sign_time'].isna().mean()*100:.2f}%")
        print(f"[Data Check] sign_lng缺失率: {self.df['sign_lng'].isna().mean()*100:.2f}%")
        print(f"[Data Check] sign_lat缺失率: {self.df['sign_lat'].isna().mean()*100:.2f}%")
        
        # sign_lng/sign_lat 缺失时，用 poi_lng/poi_lat 作为终点坐标替代
        if 'sign_lng' in self.df.columns and 'sign_lat' in self.df.columns:
            sign_lng_missing = self.df['sign_lng'].isna().mean()
            sign_lat_missing = self.df['sign_lat'].isna().mean()
            if sign_lng_missing > 0.5 or sign_lat_missing > 0.5:
                print(f"[Data Fix] sign_lng/sign_lat 缺失率过高 ({sign_lng_missing*100:.1f}%/{sign_lat_missing*100:.1f}%)")
                print(f"[Data Fix] 使用 poi_lng/poi_lat 作为终点坐标替代")
                self.df['dest_lng'] = self.df['poi_lng']
                self.df['dest_lat'] = self.df['poi_lat']
                # 对少量有 sign_lng/sign_lat 的行，优先使用真实签收坐标
                has_sign = self.df['sign_lng'].notna() & self.df['sign_lat'].notna()
                self.df.loc[has_sign, 'dest_lng'] = self.df.loc[has_sign, 'sign_lng']
                self.df.loc[has_sign, 'dest_lat'] = self.df.loc[has_sign, 'sign_lat']
                print(f"[Data Fix] 使用真实签收坐标的比例: {has_sign.mean()*100:.1f}%")
            else:
                # 缺失率低，直接用 sign 坐标，缺失部分用 poi 填充
                self.df['dest_lng'] = self.df['sign_lng'].fillna(self.df['poi_lng'])
                self.df['dest_lat'] = self.df['sign_lat'].fillna(self.df['poi_lat'])
        else:
            # 没有 sign 列，直接用 poi
            print(f"[Data Fix] 无 sign_lng/sign_lat 列，使用 poi_lng/poi_lat 作为终点坐标")
            self.df['dest_lng'] = self.df['poi_lng']
            self.df['dest_lat'] = self.df['poi_lat']
        
        # 坐标重命名（避免与真实经纬度混淆）
        self.df.rename(columns={
            'poi_lat': 'dest_lat_3857',
            'poi_lng': 'dest_lng_3857',
            'receipt_lat': 'pickup_lat_3857',
            'receipt_lng': 'pickup_lng_3857'
        }, inplace=True)
        
        # 改进：按快递员分组抽样，保持轨迹完整性
        if max_samples and len(self.df) > max_samples:
            self.df = self._stratified_courier_sampling(max_samples, min_courier_orders, max_courier_orders)
        
        self.scaler = StandardScaler()
        self.weather_encoder = LabelEncoder()
        self.traffic_encoder = LabelEncoder()
        self.poi_type_encoder = LabelEncoder()
        self.aoi_encoder = LabelEncoder()
        
        # 序列化参数
        self.min_seq_len = 5
        self.max_seq_len = 50
        self.sequences = None
    
    def _stratified_courier_sampling(self, max_samples, min_orders, max_orders):
        """
        按快递员分组抽样，保持轨迹完整性
        优先选择订单数适中的快递员（避免极端值）
        """
        print(f"[Sampling] 使用快递员分组抽样策略...")
        
        # 统计每个快递员的订单数
        courier_counts = self.df['delivery_user_id'].value_counts()
        
        # 筛选订单数适中的快递员
        valid_couriers = courier_counts[
            (courier_counts >= min_orders) & 
            (courier_counts <= max_orders)
        ].index.tolist()
        
        print(f"  符合条件的快递员: {len(valid_couriers)} 个")
        
        # 按城市分层抽样，确保各城市样本均衡
        if 'from_city_name' in self.df.columns:
            sampled_df = pd.DataFrame()
            cities = self.df['from_city_name'].unique()
            samples_per_city = max_samples // len(cities)
            
            for city in cities:
                city_couriers = self.df[
                    (self.df['from_city_name'] == city) & 
                    (self.df['delivery_user_id'].isin(valid_couriers))
                ]['delivery_user_id'].unique()
                
                if len(city_couriers) == 0:
                    continue
                
                # 随机选择快递员
                n_couriers = min(len(city_couriers), max(1, samples_per_city // min_orders))
                selected_couriers = np.random.choice(city_couriers, size=n_couriers, replace=False)
                
                city_df = self.df[self.df['delivery_user_id'].isin(selected_couriers)]
                sampled_df = pd.concat([sampled_df, city_df])
                
                if len(sampled_df) >= max_samples:
                    break
            
            result = sampled_df.head(max_samples).reset_index(drop=True)
        else:
            # 简单随机选择快递员
            n_couriers = min(len(valid_couriers), max_samples // min_orders)
            selected_couriers = np.random.choice(valid_couriers, size=n_couriers, replace=False)
            result = self.df[self.df['delivery_user_id'].isin(selected_couriers)].reset_index(drop=True)
        
        print(f"  抽样后数据量: {len(result):,} 条记录")
        print(f"  覆盖快递员数: {result['delivery_user_id'].nunique()} 个")
        return result
        
    def engineer_features(self, trajectory_path='./courier_detailed_trajectory.csv'):
        """
        特征工程：
        - 时间特征：小时、星期、是否周末、时段
        - 空间特征：距离、方向角、区域编码
        - 天气特征：天气状况、交通状况（模拟）
        - 历史特征：司机平均配送时间、路线平均时间
        - 轨迹特征：快递员活跃时长、活动范围、轨迹点数
        """
        df = self.df.copy()
        
        # 列名映射：receipt_time -> pickup_time, sign_time -> delivery_time
        df['pickup_time'] = pd.to_datetime(df['receipt_time'], format='%m-%d %H:%M:%S', errors='coerce')
        df['delivery_time'] = pd.to_datetime(df['sign_time'], format='%m-%d %H:%M:%S', errors='coerce')
        
        # 过滤无效时间数据
        df = df.dropna(subset=['pickup_time', 'delivery_time'])
        
        # 时间特征
        df['hour'] = df['pickup_time'].dt.hour
        df['weekday'] = df['pickup_time'].dt.dayofweek
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['time_period'] = pd.cut(df['hour'], 
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=[0, 1, 2, 3]).astype(int)
        
        # 时段特征：早高峰/晚高峰/夜间
        df['rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) |
                          (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)

        # ====== LaDe论文改进：周期性时间编码 ======
        # 24小时周期编码 (sin/cos)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        # 周周期编码 (sin/cos)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

        # 空间特征：使用haversine公式计算距离
        # pickup_lng_3857/pickup_lat_3857 -> 起点, dest_lng_3857/dest_lat_3857 -> 终点
        df['distance_km'] = haversine_distance(
            df['pickup_lng_3857'].values, df['pickup_lat_3857'].values,
            df['dest_lng_3857'].values, df['dest_lat_3857'].values
        )
        df['distance_km'] = df['distance_km'].fillna(df['distance_km'].median())
        
        # 计算配送时间（目标变量，单位：分钟）
        df['eta_minutes'] = (df['delivery_time'] - df['pickup_time']).dt.total_seconds() / 60
        
        # 处理跨天情况（配送时间可能为负，需要加24小时）
        df.loc[df['eta_minutes'] < 0, 'eta_minutes'] += 24 * 60
        
        # 清洗异常值（配送时间在5分钟~180分钟之间）
        df = df[(df['eta_minutes'] >= 5) & (df['eta_minutes'] <= 180)]
        
        # 司机ID映射
        df['driver_id'] = df['delivery_user_id']
        
        # ====== 历史统计特征（按司机）—— 留一法防止目标泄露 ======
        # 不直接用分组均值（那样每条样本的目标值参与了自身的特征计算）
        # 留一法: driver_avg_eta_i = (sum_all - y_i) / (count - 1)
        print("\n[Feature Engineering] 计算司机历史特征（留一法防泄露）...")
        driver_group = df.groupby('driver_id')['eta_minutes']
        driver_sum = driver_group.transform('sum')
        driver_count = driver_group.transform('count')
        
        # 留一法均值: 排除当前样本自身
        df['driver_avg_eta'] = np.where(
            driver_count > 1,
            (driver_sum - df['eta_minutes']) / (driver_count - 1),
            df['eta_minutes'].median()  # 只有1条记录的用全局中位数
        )
        
        # 留一法标准差的近似: 使用分组std但用count-2自由度修正
        driver_std_raw = driver_group.transform('std')
        df['driver_std_eta'] = driver_std_raw.fillna(0)
        
        # 司机历史订单量（本身不含目标信息，可直接用）
        df['driver_order_count'] = driver_count
        
        # 全局中位数填充剩余NaN
        global_median = df['eta_minutes'].median()
        df['driver_avg_eta'] = df['driver_avg_eta'].fillna(global_median)
        df['driver_std_eta'] = df['driver_std_eta'].fillna(0)
        
        # 真实天气和交通特征（替代原来的模拟数据）
        print("\n[Feature Engineering] 获取真实天气特征...")
        try:
            weather_provider = WeatherFeatureProvider()
            df = weather_provider.get_weather_for_dataframe(
                df,
                city_col='from_city_name',
                time_col='pickup_time',
                use_api=True,
                api_sample_limit=50  # 限制API调用次数，超出后用估算
            )
        except Exception as e:
            print(f"  [Weather] 天气获取失败: {e}，使用估算值")
            # 降级: 统计估算
            weather_provider = WeatherFeatureProvider()
            month = df['pickup_time'].dt.month if 'pickup_time' in df.columns else 6
            hour = df['pickup_time'].dt.hour if 'pickup_time' in df.columns else 12
            city = df['from_city_name'] if 'from_city_name' in df.columns else '杭州市'
            # 生成合理估算值
            weather_conditions = ['sunny', 'cloudy', 'rainy', 'foggy']
            traffic_conditions = ['low', 'medium', 'high']
            df['temperature'] = np.random.normal(20, 8, size=len(df))
            df['precipitation'] = np.random.exponential(2, size=len(df))
            df['wind_speed'] = np.random.normal(10, 4, size=len(df)).clip(0)
            df['temp_range'] = np.random.normal(8, 2, size=len(df)).clip(2)
            df['weather'] = np.random.choice(weather_conditions, size=len(df))
            df['traffic'] = np.random.choice(traffic_conditions, size=len(df))
        
        # 编码分类变量
        df['weather_encoded'] = self.weather_encoder.fit_transform(df['weather'])
        df['traffic_encoded'] = self.traffic_encoder.fit_transform(df['traffic'])

        # POI类型编码
        poi_type_mapping = {
            '4602b38053ece07a9ca5153f1df2e404': 'residential',
            '203ac3454d75e02ebb0a3c6f51d735e4': 'commercial',
            'fe76dff35bb199cdb7329eba2b918f18': 'office',
            '339d14e62a5bbd67de62f461a5f7db1e': 'shopping',
            '73ffcbd1b26557b462b14e4dd4c57fcb': 'education',
            '14cca3f2714c7c0faf2cbac10ba12d3b': 'residential',
        }
        df['poi_type'] = df['typecode'].map(poi_type_mapping).fillna('other')
        df['poi_type_encoded'] = self.poi_type_encoder.fit_transform(df['poi_type'])

        # ====== LaDe论文改进：AOI区域特征 ======
        # 根据LaDe论文，AOI（Area of Interest）是城市内的功能区域
        # 使用Geohash进行空间网格编码，捕捉局部空间模式
        try:
            import hashlib
            def geohash_encode(lng, lat, precision=6):
                """简单的Geohash编码近似实现"""
                if pd.isna(lng) or pd.isna(lat):
                    return 'unknown'
                lat_norm = (lat + 180) / 360
                lng_norm = (lng + 180) / 360
                combined = f"{lat_norm:.6f},{lng_norm:.6f}"
                return hashlib.md5(combined.encode()).hexdigest()[:precision]
            df['dest_aoi'] = df.apply(lambda x: geohash_encode(x['dest_lng_3857'], x['dest_lat_3857']), axis=1)
            df['pickup_aoi'] = df.apply(lambda x: geohash_encode(x['pickup_lng_3857'], x['pickup_lat_3857']), axis=1)
            df['aoi_encoded'] = self.aoi_encoder.fit_transform(df['dest_aoi'])
            print(f"  AOI区域数: {df['dest_aoi'].nunique()} 个")
        except Exception as e:
            print(f"  [Warning] AOI特征提取失败: {e}")
            df['dest_aoi'] = 'unknown'
            df['pickup_aoi'] = 'unknown'
            df['aoi_encoded'] = 0
        
        # 轨迹特征：从轨迹数据提取快递员特征
        try:
            print("\n[Feature Engineering] 提取轨迹特征...")
            courier_features = extract_courier_features(trajectory_path, max_rows=500000)
            
            if courier_features is not None and len(courier_features) > 0:
                # 合并轨迹特征到 delivery 数据
                df = df.merge(courier_features, on='driver_id', how='left')
                
                # 对缺失的轨迹特征用均值填充（某些 driver_id 在轨迹中不存在）
                traj_feature_cols = ['courier_total_points', 'courier_active_hours', 
                                     'courier_lat_range', 'courier_lng_range',
                                     'courier_stationary_points_pct', 'courier_moving_distance_km',
                                     'courier_actual_active_hours']
                for col in traj_feature_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(df[col].mean())
                
                print(f"  成功合并轨迹特征，覆盖 {df['courier_total_points'].notna().sum()} 条记录")
            else:
                print("  轨迹特征为空，使用默认值")
                df['courier_total_points'] = 0
                df['courier_active_hours'] = 0
                df['courier_lat_range'] = 0
                df['courier_lng_range'] = 0
                df['courier_stationary_points_pct'] = 0
                df['courier_moving_distance_km'] = 0
                df['courier_actual_active_hours'] = 0
        except Exception as e:
            print(f"  [Warning] 轨迹特征提取失败: {e}")
            print("  使用默认轨迹特征值")
            df['courier_total_points'] = 0
            df['courier_active_hours'] = 0
            df['courier_lat_range'] = 0
            df['courier_lng_range'] = 0
            df['courier_stationary_points_pct'] = 0
            df['courier_moving_distance_km'] = 0
            df['courier_actual_active_hours'] = 0
        
        # 新增：实时负载特征
        try:
            print("\n[Feature Engineering] 计算实时负载...")
            workload_features = calculate_realtime_workload('./delivery_five_cities.csv', max_samples=50000)
            if workload_features is not None and len(workload_features) > 0:
                df = df.merge(workload_features, on='driver_id', how='left')
                for col in ['avg_workload', 'max_workload', 'std_workload']:
                    if col in df.columns:
                        df[col] = df[col].fillna(df[col].mean())
                print(f"  成功合并负载特征，覆盖 {df['avg_workload'].notna().sum()} 条记录")
            else:
                print("  负载特征为空，使用默认值")
                df['avg_workload'] = 1
                df['max_workload'] = 1
                df['std_workload'] = 0
        except Exception as e:
            print(f"  [Warning] 负载特征计算失败: {e}")
            df['avg_workload'] = 1
            df['max_workload'] = 1
            df['std_workload'] = 0
        
        # ====== 路线复杂度 → 真实空间特征 ======
        # 曼哈顿距离/欧氏距离比（衡量路线弯曲程度）
        if all(c in df.columns for c in ['pickup_lng_3857', 'pickup_lat_3857', 'dest_lng_3857', 'dest_lat_3857']):
            dx = np.abs(df['dest_lng_3857'] - df['pickup_lng_3857'])
            dy = np.abs(df['dest_lat_3857'] - df['pickup_lat_3857'])
            euclidean = np.sqrt(dx**2 + dy**2) + 1e-6
            manhattan = dx + dy
            df['detour_ratio'] = manhattan / euclidean  # 范围 [1, sqrt(2)]
        else:
            df['detour_ratio'] = 1.0
        
        # ====== 新增空间/时间交互特征 ======
        # 距离×高峰时段交互
        df['distance_rush_interaction'] = df['distance_km'] * df['rush_hour']
        
        # 城市频率编码
        if 'from_city_name' in df.columns:
            city_freq = df['from_city_name'].value_counts(normalize=True)
            df['city_freq_encoded'] = df['from_city_name'].map(city_freq).fillna(0)
        else:
            df['city_freq_encoded'] = 0
        
        # 方向角特征（起终点方位角）
        if all(c in df.columns for c in ['pickup_lng_3857', 'pickup_lat_3857', 'dest_lng_3857', 'dest_lat_3857']):
            delta_lng = df['dest_lng_3857'] - df['pickup_lng_3857']
            delta_lat = df['dest_lat_3857'] - df['pickup_lat_3857']
            bearing = np.arctan2(delta_lng, delta_lat)
            df['bearing_sin'] = np.sin(bearing)
            df['bearing_cos'] = np.cos(bearing)
        else:
            df['bearing_sin'] = 0
            df['bearing_cos'] = 0
        
        # 天气-距离交互: 降水天远距离配送更慢
        df['precip_dist_interaction'] = df.get('precipitation', pd.Series(0, index=df.index)) * df['distance_km']
        
        # 距离分箱: 短/中/长距离
        df['distance_bin'] = pd.cut(
            df['distance_km'], bins=[0, 2, 5, 10, 999], labels=[0, 1, 2, 3]
        ).astype(int)
        
        # 司机订单量 log变换（缩小偏度）
        if 'driver_order_count' in df.columns:
            df['driver_order_count_log'] = np.log1p(df['driver_order_count'])
        else:
            df['driver_order_count_log'] = 0

        # ====== 历史拥堵特征 —— 用距离指标代替ETA避免目标泄露 ======
        # 不用 eta_minutes 聚合（会泄露目标），改用 distance_km 中位数作为通行效率代理
        if 'from_city_name' in df.columns and 'hour' in df.columns:
            # 城市+时段的平均配送距离（不含目标信息）
            congestion_stats = df.groupby(['from_city_name', 'hour'])['distance_km'].agg(
                ['median', 'count']
            ).reset_index()
            congestion_stats.columns = ['from_city_name', 'hour', 'city_hour_median_dist', 'city_hour_order_count']
            df = df.merge(congestion_stats, on=['from_city_name', 'hour'], how='left')
            df['city_hour_median_dist'] = df['city_hour_median_dist'].fillna(df['distance_km'].median())
            df['city_hour_order_count'] = df['city_hour_order_count'].fillna(1)
            
            # 订单密度特征: 该城市该时段的订单量（归一化）
            max_count = df['city_hour_order_count'].max()
            df['city_hour_density'] = df['city_hour_order_count'] / (max_count + 1)
        else:
            df['city_hour_median_dist'] = df['distance_km'].median()
            df['city_hour_order_count'] = 1
            df['city_hour_density'] = 0
        
        self.df = df
        return df
    
    def build_sequences(self, min_seq_len=5, max_seq_len=50):
        """
        构建快递员配送序列数据（参考 LaDe 的 DeliveryDataset）n        
        每个序列包含：
        - features: 时序特征矩阵 (seq_len, n_features)
        - eta_labels: ETA标签 (seq_len,)
        - route_label: 配送顺序标签
        - courier_id: 快递员ID
        - distance_matrix: 配送点间距离矩阵
        """
        print(f"\n[Sequence Building] 构建配送序列...")
        print(f"  最小序列长度: {min_seq_len}, 最大序列长度: {max_seq_len}")
        
        sequences = []
        courier_ids = self.df['driver_id'].unique()
        
        for courier_id in courier_ids:
            courier_df = self.df[self.df['driver_id'] == courier_id].copy()
            courier_df = courier_df.sort_values('pickup_time').reset_index(drop=True)
            
            if len(courier_df) < min_seq_len:
                continue
            
            # 滑动窗口构建序列
            feature_cols = [
                'distance_km', 'hour', 'weekday', 'is_weekend', 'time_period', 'rush_hour',
                'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
                'temperature', 'precipitation', 'wind_speed', 'temp_range',
                'weather_encoded', 'traffic_encoded', 'dest_lat_3857', 'dest_lng_3857',
                'driver_avg_eta', 'driver_std_eta', 'driver_order_count_log',
                'detour_ratio', 'distance_rush_interaction', 'city_freq_encoded',
                'bearing_sin', 'bearing_cos', 'precip_dist_interaction', 'distance_bin',
                'city_hour_median_dist', 'city_hour_density',
                'poi_type_encoded', 'aoi_encoded'
            ]
            
            # 添加轨迹特征（如果存在）
            traj_cols = ['courier_total_points', 'courier_active_hours', 'courier_lat_range', 
                        'courier_lng_range', 'courier_stationary_points_pct', 
                        'courier_moving_distance_km', 'courier_actual_active_hours']
            for col in traj_cols:
                if col in courier_df.columns:
                    feature_cols.append(col)
            
            # 添加负载特征（如果存在）
            workload_cols = ['avg_workload', 'max_workload', 'std_workload']
            for col in workload_cols:
                if col in courier_df.columns:
                    feature_cols.append(col)
            
            # 确保所有特征列都存在
            available_cols = [c for c in feature_cols if c in courier_df.columns]
            
            for start_idx in range(0, len(courier_df) - min_seq_len + 1):
                end_idx = min(start_idx + max_seq_len, len(courier_df))
                seq_df = courier_df.iloc[start_idx:end_idx]
                
                if len(seq_df) < min_seq_len:
                    continue
                
                # 提取特征（清理NaN）
                features = seq_df[available_cols].values.astype(np.float32)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                eta_labels = seq_df['eta_minutes'].values.astype(np.float32)
                eta_labels = np.nan_to_num(eta_labels, nan=0.0)
                
                # 构建距离矩阵（配送点之间的空间关系）
                coords = seq_df[['dest_lat_3857', 'dest_lng_3857']].values
                distance_matrix = self._compute_distance_matrix(coords)
                
                # 时间差特征（相对于序列起点）
                time_deltas = (seq_df['pickup_time'] - seq_df['pickup_time'].iloc[0]).dt.total_seconds() / 60
                
                sequences.append({
                    'courier_id': courier_id,
                    'features': features,
                    'eta_labels': eta_labels,
                    'route_label': list(range(len(seq_df))),  # 实际配送顺序
                    'distance_matrix': distance_matrix,
                    'time_deltas': time_deltas.values,
                    'seq_len': len(seq_df),
                    'city': seq_df['from_city_name'].iloc[0] if 'from_city_name' in seq_df.columns else 'unknown'
                })
        
        self.sequences = sequences
        
        # 对所有序列特征做标准化
        if len(sequences) > 0:
            all_features = np.vstack([s['features'] for s in sequences])
            from sklearn.preprocessing import StandardScaler
            seq_scaler = StandardScaler()
            seq_scaler.fit(all_features)
            for s in sequences:
                s['features'] = seq_scaler.transform(s['features']).astype(np.float32)
        
        print(f"  构建完成: {len(sequences)} 个序列")
        print(f"  平均序列长度: {np.mean([s['seq_len'] for s in sequences]):.1f}")
        
        return sequences
    
    def _compute_distance_matrix(self, coords):
        """计算配送点之间的距离矩阵"""
        n = len(coords)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i, j] = haversine_distance(
                        coords[i][1], coords[i][0],  # lng, lat
                        coords[j][1], coords[j][0]
                    )
        return dist_matrix
    
    def extract_aoi_features(self):
        """
        提取 AOI (Area of Interest) 特征
        基于配送点坐标进行空间聚类，识别配送区域
        """
        print("\n[AOI Feature Extraction] 提取配送区域特征...")
        
        from sklearn.cluster import DBSCAN
        
        # 使用 DBSCAN 对配送点进行空间聚类
        coords = self.df[['dest_lat_3857', 'dest_lng_3857']].values
        
        # EPS 设置为 500 米（EPSG:3857坐标单位）
        clustering = DBSCAN(eps=500, min_samples=5).fit(coords)
        self.df['aoi_cluster'] = clustering.labels_
        
        # 统计 AOI 信息
        n_aois = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        n_noise = list(clustering.labels_).count(-1)
        
        print(f"  识别 AOI 区域: {n_aois} 个")
        print(f"  噪声点数量: {n_noise} 个")
        
        # 计算每个 AOI 的统计特征
        aoi_stats = self.df[self.df['aoi_cluster'] != -1].groupby('aoi_cluster').agg({
            'eta_minutes': ['mean', 'std', 'count'],
            'distance_km': 'mean',
            'dest_lat_3857': 'mean',
            'dest_lng_3857': 'mean'
        }).reset_index()
        
        aoi_stats.columns = ['aoi_cluster', 'aoi_avg_eta', 'aoi_std_eta', 
                             'aoi_order_count', 'aoi_avg_distance', 
                             'aoi_center_lat', 'aoi_center_lng']
        
        # 合并 AOI 特征
        self.df = self.df.merge(aoi_stats, on='aoi_cluster', how='left')
        
        # 对噪声点填充默认值
        for col in ['aoi_avg_eta', 'aoi_std_eta', 'aoi_order_count', 'aoi_avg_distance']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median() if self.df[col].notna().sum() > 0 else 0)
        
        # 编码 AOI 类别
        self.df['aoi_encoded'] = self.aoi_encoder.fit_transform(
            self.df['aoi_cluster'].astype(str)
        )
        
        print(f"  AOI 特征提取完成")
        return self.df
    
    def prepare_features(self):
        """准备特征矩阵"""
        feature_cols = [
            'distance_km', 'hour', 'weekday', 'is_weekend', 'time_period', 'rush_hour',
            'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
            'temperature', 'precipitation', 'wind_speed', 'temp_range',
            'weather_encoded', 'traffic_encoded', 'dest_lat_3857', 'dest_lng_3857',
            'driver_avg_eta', 'driver_std_eta', 'driver_order_count_log',
            'courier_total_points', 'courier_active_hours', 'courier_lat_range', 'courier_lng_range',
            'courier_stationary_points_pct', 'courier_moving_distance_km', 'courier_actual_active_hours',
            'avg_workload', 'max_workload', 'std_workload',
            'detour_ratio', 'distance_rush_interaction', 'city_freq_encoded',
            'bearing_sin', 'bearing_cos', 'precip_dist_interaction', 'distance_bin',
            'city_hour_median_dist', 'city_hour_density',
            'poi_type_encoded', 'aoi_encoded'
        ]

        df_filtered = self.df.copy()

        # 只保留存在的特征列
        available_cols = [c for c in feature_cols if c in df_filtered.columns]
        missing_cols = [c for c in feature_cols if c not in df_filtered.columns]
        for col in missing_cols:
            df_filtered[col] = 0

        # 用中位数/0填充NaN，而非丢弃行
        for col in feature_cols:
            if df_filtered[col].isna().any():
                fill_val = df_filtered[col].median()
                if pd.isna(fill_val):
                    fill_val = 0
                df_filtered[col] = df_filtered[col].fillna(fill_val)

        # 过滤目标变量缺失的行
        df_filtered = df_filtered[df_filtered['eta_minutes'].notna()]
        nan_count_before = len(self.df)
        nan_count_after = len(df_filtered)
        if nan_count_before > nan_count_after:
            print(f"[prepare_features] Removed {nan_count_before - nan_count_after} rows with NaN in eta_minutes")

        X = df_filtered[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = df_filtered['eta_minutes'].values.astype(np.float32)

        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y, feature_cols


# ============== 2. 序列数据集 ==============

class SequenceDataset(Dataset):
    """
    序列化配送数据集 - 支持时序建模
    参考 LaDe 的 DeliveryDataset 设计
    """
    
    def __init__(self, sequences, max_seq_len=50, feature_dim=None):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim or sequences[0]['features'].shape[1]
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # 特征填充/截断（确保无NaN）
        features = np.nan_to_num(seq['features'].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        seq_len = len(features)
        
        if seq_len < self.max_seq_len:
            # 填充
            pad_len = self.max_seq_len - seq_len
            padded_features = np.vstack([
                features,
                np.zeros((pad_len, features.shape[1]))
            ])
            padding_mask = np.concatenate([
                np.ones(seq_len),
                np.zeros(pad_len)
            ])
        else:
            padded_features = features[:self.max_seq_len]
            padding_mask = np.ones(self.max_seq_len)
            seq_len = self.max_seq_len
        
        # 距离矩阵填充
        dist_matrix = seq['distance_matrix']
        if dist_matrix.shape[0] < self.max_seq_len:
            padded_dist = np.zeros((self.max_seq_len, self.max_seq_len))
            padded_dist[:dist_matrix.shape[0], :dist_matrix.shape[1]] = dist_matrix
        else:
            padded_dist = dist_matrix[:self.max_seq_len, :self.max_seq_len]
        
        # ETA 标签
        eta_labels = seq['eta_labels']
        if len(eta_labels) < self.max_seq_len:
            padded_eta = np.concatenate([
                eta_labels,
                np.zeros(self.max_seq_len - len(eta_labels))
            ])
        else:
            padded_eta = eta_labels[:self.max_seq_len]
        
        return {
            'features': torch.FloatTensor(padded_features),
            'distance_matrix': torch.FloatTensor(padded_dist),
            'eta_labels': torch.FloatTensor(padded_eta),
            'padding_mask': torch.BoolTensor(padding_mask),
            'seq_len': seq_len,
            'courier_id': seq['courier_id']
        }


# ============== 3. 图神经网络模型 (Graph2Route风格) ==============

class GraphConvolution(nn.Module):
    """图卷积层 - 用于学习配送点之间的空间关系"""
    
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        
    def forward(self, x, adj):
        """
        x: (batch, seq_len, features)
        adj: (batch, seq_len, seq_len) - 邻接矩阵/距离矩阵
        """
        # 归一化邻接矩阵
        degree = adj.sum(dim=-1, keepdim=True) + 1e-6
        norm_adj = adj / degree
        
        # 图卷积: H' = A * X * W
        support = self.linear(x)
        output = torch.bmm(norm_adj, support)
        return self.activation(output)


class Graph2RouteETA(nn.Module):
    """
    Graph2Route风格的GNN模型用于ETA预测
    结合图神经网络和时序建模
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_gcn_layers=2, 
                 num_lstm_layers=2, dropout=0.2):
        super(Graph2RouteETA, self).__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 图卷积层
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim)
            for _ in range(num_gcn_layers)
        ])
        
        # LSTM时序建模
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, features, distance_matrix, padding_mask):
        """
        features: (batch, seq_len, input_dim)
        distance_matrix: (batch, seq_len, seq_len)
        padding_mask: (batch, seq_len)
        """
        batch_size, seq_len, _ = features.shape
        
        # 输入投影
        x = self.input_proj(features)  # (batch, seq_len, hidden_dim)
        
        # 图卷积 - 学习空间关系
        adj = torch.exp(-distance_matrix / (distance_matrix.mean() + 1e-6))  # 高斯核
        for gcn in self.gcn_layers:
            x = gcn(x, adj)
        
        # LSTM时序建模
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # 应用padding mask
        lstm_out = lstm_out * padding_mask.unsqueeze(-1).float()
        
        # 注意力聚合
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = attn_weights.masked_fill(
            ~padding_mask.unsqueeze(-1), float('-inf')
        )
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)
        
        # 预测
        output = self.fc(context).squeeze(-1)  # (batch,)
        return output


# ============== 3.5 STGNN模型 (时空图神经网络) ==============

class SpatialGraphConv(nn.Module):
    """空间图卷积层 - 基于Chebyshev多项式近似"""

    def __init__(self, in_features, out_features, kernel_size=3):
        super(SpatialGraphConv, self).__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size, padding=kernel_size//2)
        self.activation = nn.ReLU()

    def forward(self, x, adj):
        """
        x: (batch, nodes, features)
        adj: (batch, nodes, nodes) - 邻接矩阵
        """
        x = x.transpose(1, 2)  # (batch, features, nodes)
        x = self.conv(x)  # (batch, out_features, nodes)
        x = x.transpose(1, 2)  # (batch, nodes, out_features)

        if adj is not None and x.shape[1] == adj.shape[1]:
            x = torch.bmm(adj, x)
        return self.activation(x)


class STGNN(nn.Module):
    """
    时空图神经网络 (STGNN)
    论文参考: LaDe - Spatial Temporal Graph Neural Network for ETA

    结构: GCN层 -> 1D卷积 -> GRU层 -> 全连接层
    适用于: 时空预测任务，如交通流量预测、ETA预测
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super(STGNN, self).__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gcn_layers = nn.ModuleList([
            SpatialGraphConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.temporal_conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, adj=None, padding_mask=None):
        """
        x: (batch, seq_len, features) - 配送序列，seq_len=nodes(配送点数)
        adj: (batch, nodes, nodes) - 距离矩阵/邻接矩阵
        padding_mask: (batch, seq_len)
        """
        # 输入: (batch, seq_len, features)
        # 对于配送场景: seq_len 即为 nodes (每个配送点是一个节点)
        if len(x.shape) == 3:
            batch_size, nodes, feat_dim = x.shape
        else:
            # 4D输入: (batch, nodes, seq_len, features) -> 压缩为3D
            batch_size, nodes, seq_len_4d, feat_dim = x.shape
            x = x.reshape(batch_size, nodes * seq_len_4d, feat_dim)
            nodes = nodes * seq_len_4d

        # 输入投影
        x = self.input_proj(x)  # (batch, nodes, hidden)

        # 空间图卷积: 学习配送点之间的空间关系
        if adj is not None:
            if len(adj.shape) == 2:
                adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
            # 确保 adj 与 nodes 维度匹配
            if adj.shape[1] != nodes:
                adj = None  # 尺寸不匹配时跳过图卷积中的邻接矩阵

        for gcn in self.gcn_layers:
            x = gcn(x, adj)  # (batch, nodes, hidden)

        # 时间卷积: 沿节点序列方向做1D卷积
        x = x.transpose(1, 2)  # (batch, hidden, nodes)
        x = self.temporal_conv(x)  # (batch, hidden, nodes)
        x = x.transpose(1, 2)  # (batch, nodes, hidden)

        # GRU 时序建模
        gru_out, _ = self.gru(x)  # (batch, nodes, hidden*2)

        # 应用 padding mask
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).float()  # (batch, nodes, 1)
            gru_out = gru_out * mask

        # 注意力聚合
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)  # (batch, nodes, 1)
        if padding_mask is not None:
            attn_weights = attn_weights.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        context = torch.sum(attn_weights * gru_out, dim=1)  # (batch, hidden*2)

        # 输出预测
        output = self.fc(context).squeeze(-1)  # (batch,)

        return output


# ============== 4. LSTM模型 ==============

class LSTMETA(nn.Module):
    """
    LSTM模型用于时序ETA预测
    适用于序列化的配送历史数据
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super(LSTMETA, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, features) -> 需要扩展seq_len维度
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention机制
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)
        
        output = self.fc(context)
        return output


# ============== 3. Transformer模型 ==============

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerETA(nn.Module):
    """
    Transformer模型用于ETA预测
    利用自注意力机制捕捉时空特征交互
    """
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerETA, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, features) or (batch, seq_len, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # 取最后一个时间步或平均
        x = x.mean(dim=1)  # (batch, d_model)
        output = self.fc(x)
        return output


# ============== 5. 评估指标 (LaDe论文标准) ==============

def calculate_hr_k(predicted_routes, actual_routes, k=3):
    """
    HR@K: Hit Rate at K
    预测路径中前K个节点与实际路径的匹配率
    """
    hits = 0
    total = 0
    for pred, actual in zip(predicted_routes, actual_routes):
        pred_k = set(pred[:k])
        actual_set = set(actual)
        hits += len(pred_k & actual_set)
        total += min(k, len(actual))
    return hits / total if total > 0 else 0


def calculate_krc(predicted_routes, actual_routes):
    """
    KRC: Kendall Rank Correlation
    衡量预测顺序与实际顺序的相关性
    """
    from scipy.stats import kendalltau
    
    correlations = []
    for pred, actual in zip(predicted_routes, actual_routes):
        if len(pred) < 2 or len(actual) < 2:
            continue
        
        # 创建排名向量
        min_len = min(len(pred), len(actual))
        pred_ranks = np.argsort(pred[:min_len])
        actual_ranks = np.argsort(actual[:min_len])
        
        if len(pred_ranks) > 1:
            tau, _ = kendalltau(pred_ranks, actual_ranks)
            if not np.isnan(tau):
                correlations.append(tau)
    
    return np.mean(correlations) if correlations else 0


def calculate_lsd(predicted_coords, actual_coords):
    """
    LSD: Location Square Deviation
    预测位置与实际位置的均方偏差（公里）
    """
    deviations = []
    for pred_coord, actual_coord in zip(predicted_coords, actual_coords):
        if len(pred_coord) == 0 or len(actual_coord) == 0:
            continue
        
        min_len = min(len(pred_coord), len(actual_coord))
        for i in range(min_len):
            dist = haversine_distance(
                pred_coord[i][1], pred_coord[i][0],  # lng, lat
                actual_coord[i][1], actual_coord[i][0]
            )
            deviations.append(dist ** 2)
    
    return np.sqrt(np.mean(deviations)) if deviations else 0


def calculate_ed(predicted_routes, actual_routes):
    """
    ED: Edit Distance (Levenshtein Distance)
    衡量预测路径与实际路径的差异
    归一化到 [0, 1] 范围
    """
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    distances = []
    for pred, actual in zip(predicted_routes, actual_routes):
        max_len = max(len(pred), len(actual))
        if max_len == 0:
            continue
        dist = levenshtein_distance(pred, actual)
        distances.append(dist / max_len)  # 归一化
    
    return np.mean(distances) if distances else 0


def evaluate_route_metrics(model, sequences, device, max_samples=100):
    """
    评估路径预测指标 (HR@K, KRC, LSD, ED)
    """
    model.eval()
    
    # 采样评估
    eval_sequences = sequences[:max_samples] if len(sequences) > max_samples else sequences
    
    predicted_routes = []
    actual_routes = []
    predicted_coords = []
    actual_coords = []
    
    with torch.no_grad():
        for seq in eval_sequences:
            # 这里简化处理：使用 ETA 预测排序作为路径预测
            features = torch.FloatTensor(seq['features']).unsqueeze(0).to(device)
            dist_matrix = torch.FloatTensor(seq['distance_matrix']).unsqueeze(0).to(device)
            padding_mask = torch.ones(1, len(seq['features'])).bool().to(device)
            
            # 预测 ETA
            if hasattr(model, 'forward') and 'distance_matrix' in model.forward.__code__.co_varnames:
                eta_pred = model(features, dist_matrix, padding_mask).cpu().numpy()
            else:
                eta_pred = model(features).cpu().numpy()
            
            # 基于 ETA 排序生成预测路径（ETA 短的先配送）
            pred_route = np.argsort(eta_pred).tolist()
            
            predicted_routes.append(pred_route)
            actual_routes.append(seq['route_label'])
            
            # 坐标（用于 LSD 计算）
            coords = seq['features'][:, 8:10]  # dest_lat, dest_lng 列
            predicted_coords.append([coords[i] for i in pred_route])
            actual_coords.append(coords)
    
    metrics = {
        'HR@1': calculate_hr_k(predicted_routes, actual_routes, k=1),
        'HR@3': calculate_hr_k(predicted_routes, actual_routes, k=3),
        'HR@5': calculate_hr_k(predicted_routes, actual_routes, k=5),
        'KRC': calculate_krc(predicted_routes, actual_routes),
        'LSD': calculate_lsd(predicted_coords, actual_coords),
        'ED': calculate_ed(predicted_routes, actual_routes)
    }
    
    return metrics


# ============== 6. 训练与评估 ==============

def train_tree_baselines(X_train, y_train, X_test, y_test):
    """训练 XGBoost 和 LightGBM baseline 模型"""
    import time
    tree_results = {}
    
    # XGBoost
    try:
        from xgboost import XGBRegressor
        print("\n--- Training XGBoost ---")
        start = time.time()
        xgb_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train, y_train)
        train_time = time.time() - start
        
        xgb_pred = xgb_model.predict(X_test)
        xgb_mae = np.mean(np.abs(xgb_pred - y_test))
        xgb_rmse = np.sqrt(np.mean((xgb_pred - y_test) ** 2))
        
        tree_results['XGBoost'] = {
            'mae': xgb_mae, 'rmse': xgb_rmse, 'train_time': train_time,
            'predictions': xgb_pred, 'actuals': y_test, 'model': xgb_model
        }
        print(f"\nXGBoost Test Results:")
        print(f"  MAE: {xgb_mae:.2f} min")
        print(f"  RMSE: {xgb_rmse:.2f} min")
        print(f"  Train Time: {train_time:.2f}s")
    except ImportError:
        print("[XGBoost] Not installed, skipping. Run: pip install xgboost")
    except Exception as e:
        print(f"[XGBoost] Error: {e}")
    
    # LightGBM
    try:
        from lightgbm import LGBMRegressor
        print("\n--- Training LightGBM ---")
        start = time.time()
        lgbm_model = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)
        lgbm_model.fit(X_train, y_train)
        train_time = time.time() - start
        
        lgbm_pred = lgbm_model.predict(X_test)
        lgbm_mae = np.mean(np.abs(lgbm_pred - y_test))
        lgbm_rmse = np.sqrt(np.mean((lgbm_pred - y_test) ** 2))
        
        tree_results['LightGBM'] = {
            'mae': lgbm_mae, 'rmse': lgbm_rmse, 'train_time': train_time,
            'predictions': lgbm_pred, 'actuals': y_test, 'model': lgbm_model
        }
        print(f"\nLightGBM Test Results:")
        print(f"  MAE: {lgbm_mae:.2f} min")
        print(f"  RMSE: {lgbm_rmse:.2f} min")
        print(f"  Train Time: {train_time:.2f}s")
    except ImportError:
        print("[LightGBM] Not installed, skipping. Run: pip install lightgbm")
    except Exception as e:
        print(f"[LightGBM] Error: {e}")
    
    return tree_results


class ETADataset(Dataset):
    """ETA预测数据集"""
    
    def __init__(self, X, y, seq_len=10):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # 创建序列（用于学习时间依赖）
        start_idx = max(0, idx - self.seq_len + 1)
        seq = self.X[start_idx:idx + 1]
        
        # Padding到固定长度
        if len(seq) < self.seq_len:
            pad = torch.zeros(self.seq_len - len(seq), seq.shape[1])
            seq = torch.cat([pad, seq], dim=0)
        
        return seq, self.y[idx]


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.squeeze(), y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            
            loss = criterion(output.squeeze(), y)
            total_loss += loss.item()
            
            predictions.extend(output.squeeze().cpu().numpy())
            actuals.extend(y.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    return total_loss / len(dataloader), mae, rmse, predictions, actuals


def eta_main(use_sequences=True):
    """
    ETA预测主函数 - 支持序列化建模
    
    Args:
        use_sequences: 是否使用序列化数据构建（推荐True）
    """
    import time
    
    # 配置
    DATA_PATH = './delivery_five_cities.csv'
    SEQ_LEN = 50  # 序列最大长度
    BATCH_SIZE = 64  # 序列模型使用较小batch
    EPOCHS = 10
    LEARNING_RATE = 0.001
    MAX_SAMPLES = 200000  # 放宽抽样限制，增加训练数据
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # 1. 数据加载与预处理
    print("=" * 50)
    print("Step 1: Data Loading and Preprocessing")
    print("=" * 50)
    
    processor = LaDeDataProcessor(DATA_PATH, max_samples=MAX_SAMPLES)
    df = processor.engineer_features()
    
    # 提取 AOI 特征
    processor.extract_aoi_features()
    
    print(f"Dataset size: {len(df)} records")
    
    # 2. 序列化数据构建（新功能）
    if use_sequences:
        print("\n" + "=" * 50)
        print("Step 1.5: Building Sequences (LaDe Style)")
        print("=" * 50)
        sequences = processor.build_sequences(min_seq_len=5, max_seq_len=SEQ_LEN)
        
        # 划分训练/验证/测试集（按快递员划分，避免数据泄露）
        courier_ids = list(set([s['courier_id'] for s in sequences]))
        np.random.shuffle(courier_ids)
        
        n_train = int(len(courier_ids) * 0.7)
        n_val = int(len(courier_ids) * 0.15)
        
        train_couriers = set(courier_ids[:n_train])
        val_couriers = set(courier_ids[n_train:n_train+n_val])
        test_couriers = set(courier_ids[n_train+n_val:])
        
        train_seqs = [s for s in sequences if s['courier_id'] in train_couriers]
        val_seqs = [s for s in sequences if s['courier_id'] in val_couriers]
        test_seqs = [s for s in sequences if s['courier_id'] in test_couriers]
        
        print(f"Train sequences: {len(train_seqs)}")
        print(f"Val sequences: {len(val_seqs)}")
        print(f"Test sequences: {len(test_seqs)}")
        
        # 创建序列数据集
        feature_dim = train_seqs[0]['features'].shape[1]
        train_dataset = SequenceDataset(train_seqs, max_seq_len=SEQ_LEN, feature_dim=feature_dim)
        val_dataset = SequenceDataset(val_seqs, max_seq_len=SEQ_LEN, feature_dim=feature_dim)
        test_dataset = SequenceDataset(test_seqs, max_seq_len=SEQ_LEN, feature_dim=feature_dim)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        input_dim = feature_dim
        
        # 为树模型 baseline 准备扁平化特征
        X_flat, y_flat, _ = processor.prepare_features()
        X_train, X_temp, y_train, y_temp = train_test_split(X_flat, y_flat, test_size=0.3, random_state=42)
        _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    else:
        # 传统方法
        X, y, feature_cols = processor.prepare_features()
        print(f"Feature dimensions: {X.shape[1]}")
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        train_dataset = ETADataset(X_train, y_train, SEQ_LEN)
        val_dataset = ETADataset(X_val, y_val, SEQ_LEN)
        test_dataset = ETADataset(X_test, y_test, SEQ_LEN)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        input_dim = X_train.shape[1]
    
    # 2. 模型训练
    print("\n" + "=" * 50)
    print("Step 2: Model Training")
    print("=" * 50)
    
    # 定义模型
    models = {
        'LSTM': LSTMETA(input_dim, hidden_dim=128, num_layers=2),
        'Transformer': TransformerETA(input_dim, d_model=128, nhead=8, num_layers=3),
        'Graph2Route': Graph2RouteETA(input_dim, hidden_dim=128, num_gcn_layers=2, num_lstm_layers=2),
        'STGNN': STGNN(input_dim, hidden_dim=128, num_layers=3)
    }
    
    results = {}
    criterion = nn.MSELoss()
    
    for model_name, model in models.items():
        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        print(f"\n--- Training {model_name} ---")
        start_time = time.time()
        best_mae = float('inf')
        best_model_state = None
        patience_counter = 0
        early_stop_patience = 10

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            num_batches = 0

            for batch in train_loader:
                if use_sequences and isinstance(batch, dict):
                    features = batch['features'].to(DEVICE)
                    distance_matrix = batch['distance_matrix'].to(DEVICE)
                    eta_labels = batch['eta_labels'].to(DEVICE)
                    padding_mask = batch['padding_mask'].to(DEVICE)

                    optimizer.zero_grad()

                    if model_name == 'Graph2Route':
                        output = model(features, distance_matrix, padding_mask)
                    elif model_name == 'STGNN':
                        output = model(features, adj=distance_matrix, padding_mask=padding_mask)
                    else:
                        output = model(features)

                    output = output.squeeze(-1) if output.dim() > 1 else output
                    loss = criterion(output, eta_labels[:, 0])
                else:
                    X_batch, y_batch = batch
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

                    optimizer.zero_grad()
                    if model_name == 'STGNN':
                        output = model(X_batch.unsqueeze(1))
                    else:
                        output = model(X_batch)
                    loss = criterion(output.squeeze(), y_batch)

                if torch.isnan(loss):
                    continue

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            if num_batches == 0:
                print(f"  [Warning] No valid batches in epoch {epoch+1}, skipping")
                continue

            train_loss = total_loss / num_batches

            model.eval()
            val_loss = 0
            val_predictions = []
            val_actuals = []
            num_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    if use_sequences and isinstance(batch, dict):
                        features = batch['features'].to(DEVICE)
                        distance_matrix = batch['distance_matrix'].to(DEVICE)
                        eta_labels = batch['eta_labels'].to(DEVICE)
                        padding_mask = batch['padding_mask'].to(DEVICE)

                        if model_name == 'Graph2Route':
                            output = model(features, distance_matrix, padding_mask)
                        elif model_name == 'STGNN':
                            output = model(features, adj=distance_matrix, padding_mask=padding_mask)
                        else:
                            output = model(features)

                        output = output.squeeze(-1) if output.dim() > 1 else output
                        val_predictions.extend(output.cpu().numpy())
                        val_actuals.extend(eta_labels[:, 0].cpu().numpy())
                        loss = criterion(output, eta_labels[:, 0])
                    else:
                        X_batch, y_batch = batch
                        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                        output = model(X_batch)
                        val_predictions.extend(output.squeeze().cpu().numpy())
                        val_actuals.extend(y_batch.cpu().numpy())
                        loss = criterion(output.squeeze(), y_batch)

                    if not torch.isnan(loss):
                        val_loss += loss.item()
                        num_val_batches += 1

            if num_val_batches == 0:
                print(f"  [Warning] No valid validation batches in epoch {epoch+1}, skipping")
                continue

            val_loss = val_loss / num_val_batches
            val_predictions = np.array(val_predictions)
            val_actuals = np.array(val_actuals)

            val_predictions = np.nan_to_num(val_predictions, nan=best_mae)
            val_actuals = np.nan_to_num(val_actuals, nan=0)

            val_mae = np.mean(np.abs(val_predictions - val_actuals))
            val_rmse = np.sqrt(np.mean((val_predictions - val_actuals) ** 2))

            if np.isnan(val_mae):
                val_mae = best_mae

            scheduler.step(val_loss)

            if val_mae < best_mae:
                best_mae = val_mae
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, "
                      f"Val MAE: {val_mae:.2f}min, Val RMSE: {val_rmse:.2f}")

            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_model_state is None:
            print(f"  [Warning] No valid model state found for {model_name}, using initial state")
            best_model_state = model.state_dict().copy()

        train_time = time.time() - start_time

        model.load_state_dict(best_model_state)
        
        # 最终测试集评估
        model.eval()
        test_predictions = []
        test_actuals = []
        
        with torch.no_grad():
            for batch in test_loader:
                if use_sequences and isinstance(batch, dict):
                    features = batch['features'].to(DEVICE)
                    distance_matrix = batch['distance_matrix'].to(DEVICE)
                    eta_labels = batch['eta_labels'].to(DEVICE)
                    padding_mask = batch['padding_mask'].to(DEVICE)
                    
                    if model_name == 'Graph2Route':
                        output = model(features, distance_matrix, padding_mask)
                    elif model_name == 'STGNN':
                        output = model(features, adj=distance_matrix, padding_mask=padding_mask)
                    else:
                        output = model(features)

                    output = output.squeeze(-1) if output.dim() > 1 else output
                    test_predictions.extend(output.cpu().numpy())
                    test_actuals.extend(eta_labels[:, 0].cpu().numpy())
                else:
                    X_batch, y_batch = batch
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    if model_name == 'STGNN':
                        output = model(X_batch.unsqueeze(1))
                    else:
                        output = model(X_batch)
                    test_predictions.extend(output.squeeze().cpu().numpy())
                    test_actuals.extend(y_batch.cpu().numpy())
        
        test_predictions = np.array(test_predictions)
        test_actuals = np.array(test_actuals)
        test_mae = np.mean(np.abs(test_predictions - test_actuals))
        test_rmse = np.sqrt(np.mean((test_predictions - test_actuals) ** 2))
        
        results[model_name] = {
            'mae': test_mae,
            'rmse': test_rmse,
            'train_time': train_time,
            'predictions': test_predictions,
            'actuals': test_actuals
        }
        
        print(f"\n{model_name} Test Results:")
        print(f"  MAE: {test_mae:.2f} min")
        print(f"  RMSE: {test_rmse:.2f} min")
        print(f"  Train Time: {train_time:.2f}s")
        
        # 评估路径指标（仅序列模式）
        if use_sequences and model_name in ['Graph2Route', 'STGNN']:
            print(f"\n{model_name} Route Metrics:")
            route_metrics = evaluate_route_metrics(model, test_seqs, DEVICE, max_samples=100)
            for metric, value in route_metrics.items():
                print(f"  {metric}: {value:.4f}")
                results[model_name][f'route_{metric}'] = value
    
    # 2.5 性能监控初始化
    monitor = PerformanceMonitor()
    monitor.start_timer('总流程')
    monitor.record_memory('流程启动')
    
    # 2.6 特征选择
    print("\n" + "=" * 50)
    print("Step 2.5: 特征选择")
    print("=" * 50)
    
    monitor.start_timer('特征选择')
    X_flat_raw, y_flat_raw, feature_cols = processor.prepare_features()
    
    fs = FeatureSelector(X_flat_raw, y_flat_raw, feature_cols)
    selected_indices, selected_features, X_selected = fs.ensemble_selection(k=min(15, len(feature_cols)))
    monitor.stop_timer('特征选择')
    
    # 使用选中特征重新划分数据
    X_train_sel, X_temp_sel, y_train_sel, y_temp_sel = train_test_split(
        X_selected, y_flat_raw, test_size=0.3, random_state=42)
    X_val_sel, X_test_sel, y_val_sel, y_test_sel = train_test_split(
        X_temp_sel, y_temp_sel, test_size=0.5, random_state=42)
    
    # 同时保留完整特征数据
    X_train_full, X_temp_full, y_train_full, y_temp_full = train_test_split(
        X_flat_raw, y_flat_raw, test_size=0.3, random_state=42)
    X_val_full, X_test_full, y_val_full, y_test_full = train_test_split(
        X_temp_full, y_temp_full, test_size=0.5, random_state=42)
    
    # 2.7 超参数自动调优
    print("\n" + "=" * 50)
    print("Step 2.6: 超参数自动调优")
    print("=" * 50)
    
    monitor.start_timer('超参数调优')
    tuner = HyperparameterTuner(X_train_sel, y_train_sel, X_val_sel, y_val_sel)
    best_params = tuner.tune_all(n_trials=30, timeout=180)
    monitor.stop_timer('超参数调优')
    monitor.record_memory('超参数调优完成')
    
    # 2.8 用最优参数训练树模型 baseline
    print("\n" + "=" * 50)
    print("Step 2.7: Tree Model Baselines (优化后)")
    print("=" * 50)
    
    monitor.start_timer('树模型训练')
    
    # 用调优后的参数训练
    tree_results = {}
    try:
        from xgboost import XGBRegressor
        xgb_params = best_params.get('XGBoost', {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'verbosity': 0})
        print("\n--- Training XGBoost (tuned) ---")
        start = time.time()
        xgb_model = XGBRegressor(**xgb_params)
        xgb_model.fit(X_train_sel, y_train_sel)
        xgb_train_time = time.time() - start
        xgb_pred = xgb_model.predict(X_test_sel)
        xgb_mae = np.mean(np.abs(xgb_pred - y_test_sel))
        xgb_rmse = np.sqrt(np.mean((xgb_pred - y_test_sel) ** 2))
        tree_results['XGBoost'] = {
            'mae': xgb_mae, 'rmse': xgb_rmse, 'train_time': xgb_train_time,
            'predictions': xgb_pred, 'actuals': y_test_sel, 'model': xgb_model
        }
        monitor.comprehensive_eval('XGBoost', y_test_sel, xgb_pred)
        print(f"  MAE: {xgb_mae:.2f} min, RMSE: {xgb_rmse:.2f} min, Time: {xgb_train_time:.2f}s")
    except Exception as e:
        print(f"[XGBoost] Error: {e}")
    
    try:
        from lightgbm import LGBMRegressor
        lgbm_params = best_params.get('LightGBM', {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'verbose': -1})
        print("\n--- Training LightGBM (tuned) ---")
        start = time.time()
        lgbm_model = LGBMRegressor(**lgbm_params)
        lgbm_model.fit(X_train_sel, y_train_sel)
        lgbm_train_time = time.time() - start
        lgbm_pred = lgbm_model.predict(X_test_sel)
        lgbm_mae = np.mean(np.abs(lgbm_pred - y_test_sel))
        lgbm_rmse = np.sqrt(np.mean((lgbm_pred - y_test_sel) ** 2))
        tree_results['LightGBM'] = {
            'mae': lgbm_mae, 'rmse': lgbm_rmse, 'train_time': lgbm_train_time,
            'predictions': lgbm_pred, 'actuals': y_test_sel, 'model': lgbm_model
        }
        monitor.comprehensive_eval('LightGBM', y_test_sel, lgbm_pred)
        print(f"  MAE: {lgbm_mae:.2f} min, RMSE: {lgbm_rmse:.2f} min, Time: {lgbm_train_time:.2f}s")
    except Exception as e:
        print(f"[LightGBM] Error: {e}")
    
    results.update(tree_results)
    monitor.stop_timer('树模型训练')
    
    # 2.9 偏差-方差分析
    print("\n" + "=" * 50)
    print("Step 2.8: 偏差-方差分析")
    print("=" * 50)
    
    monitor.start_timer('偏差方差分析')
    bv_analyzer = BiasVarianceAnalyzer(X_train_sel, y_train_sel, X_test_sel, y_test_sel)
    bv_results = bv_analyzer.analyze_all_models(n_bootstraps=15)
    monitor.stop_timer('偏差方差分析')
    
    # 2.10 学习曲线分析
    print("\n" + "=" * 50)
    print("Step 2.9: 学习曲线分析")
    print("=" * 50)
    
    monitor.start_timer('学习曲线分析')
    lc_analyzer = LearningCurveAnalyzer(X_train_sel, y_train_sel)
    lc_results = lc_analyzer.analyze_all_models(
        train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    monitor.stop_timer('学习曲线分析')
    monitor.record_memory('学习曲线分析完成')
    
    # 2.11 SHAP可解释性分析
    print("\n" + "=" * 50)
    print("Step 2.10: SHAP可解释性分析")
    print("=" * 50)
    
    monitor.start_timer('SHAP分析')
    shap_results = {}
    
    # 对最佳树模型做SHAP
    best_tree_name = None
    best_tree_model = None
    best_tree_mae = float('inf')
    for name in ['XGBoost', 'LightGBM']:
        if name in tree_results and tree_results[name]['mae'] < best_tree_mae:
            best_tree_mae = tree_results[name]['mae']
            best_tree_name = name
            best_tree_model = tree_results[name]['model']
    
    if best_tree_model is not None:
        print(f"\n  对最佳树模型 {best_tree_name} 进行SHAP分析...")
        explainer = SHAPExplainer(
            best_tree_model, X_test_sel, selected_features, model_type='tree'
        )
        shap_results = explainer.full_analysis(top_k=min(15, len(selected_features)), sample_idx=0)
    else:
        print("  [SHAP] 无可用树模型，跳过")
    
    monitor.stop_timer('SHAP分析')
    
    # 3. 模型对比
    print("\n" + "=" * 50)
    print("Step 3: Model Comparison")
    print("=" * 50)
    
    # 对深度学习模型也做综合评估
    for model_name in ['LSTM', 'Transformer', 'Graph2Route', 'STGNN']:
        if model_name in results and 'predictions' in results[model_name]:
            monitor.comprehensive_eval(
                model_name,
                results[model_name]['actuals'],
                results[model_name]['predictions']
            )
    
    print("\n| Model        | MAE (min) | RMSE (min) | Train Time (s) |")
    print("|--------------|-----------|------------|----------------|")
    for name, result in results.items():
        train_time = result.get('train_time', 0)
        print(f"| {name:<12} | {result['mae']:<9.2f} | {result['rmse']:<10.2f} | {train_time:<14.2f} |")
    
    # 找到最佳模型
    best_model_name = min(results, key=lambda x: results[x]['mae'])
    print(f"\nBest Model: {best_model_name} (MAE: {results[best_model_name]['mae']:.2f}min)")
    
    # 4. 性能监控汇总
    monitor.stop_timer('总流程')
    monitor.record_memory('流程结束')
    monitor.print_summary()
    
    try:
        monitor.save_report()
    except Exception as e:
        print(f"  [Monitor] 保存报告失败: {e}")
    
    # 5. 保存模型
    print("\n" + "=" * 50)
    print("Step 5: Save Model")
    print("=" * 50)
    
    import joblib
    if best_model_name in ['XGBoost', 'LightGBM']:
        save_data = {
            'model': results[best_model_name]['model'],
            'selected_features': selected_features,
            'best_params': best_params.get(best_model_name, {}),
            'feature_selection_results': fs.selection_results
        }
        joblib.dump(save_data, f'./eta_model_{best_model_name.lower()}.joblib')
        print(f"Model saved: eta_model_{best_model_name.lower()}.joblib")
    else:
        torch.save({
            'model_state': models[best_model_name].state_dict(),
            'scaler': processor.scaler,
            'weather_encoder': processor.weather_encoder,
            'traffic_encoder': processor.traffic_encoder,
            'feature_cols': feature_cols,
            'selected_features': selected_features
        }, f'./eta_model_{best_model_name.lower()}.pth')
        print(f"Model saved: eta_model_{best_model_name.lower()}.pth")
    
    # 汇总所有分析结果
    results['_analysis'] = {
        'feature_selection': fs.selection_results,
        'hyperparameter_tuning': best_params,
        'bias_variance': bv_results,
        'learning_curves': lc_results,
        'shap_analysis': shap_results
    }
    
    return results


# ============== Pickup数据分析 ==============

def pickup_analysis():
    """揽件数据分析主函数"""
    print("\n" + "=" * 60)
    print("Pickup Analysis - 揽件ETA预测分析")
    print("=" * 60)
    
    try:
        DATA_PATH = './pickup_five_cities.csv'
        MAX_SAMPLES = 50000
        
        # 1. 数据加载
        print("\n[1/4] 加载揽件数据...")
        df = pd.read_csv(DATA_PATH)
        print(f"  原始数据量: {len(df):,} 条")
        
        # 抽样
        if len(df) > MAX_SAMPLES:
            df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
            print(f"  抽样后数据量: {len(df):,} 条")
        
        # 2. 数据预处理
        print("\n[2/4] 数据预处理...")
        
        # 解析时间格式 (格式如: 03-19 07:25:00)
        df['accept_time'] = pd.to_datetime(df['accept_time'], format='%m-%d %H:%M:%S', errors='coerce')
        df['got_time'] = pd.to_datetime(df['got_time'], format='%m-%d %H:%M:%S', errors='coerce')
        df['expect_got_time'] = pd.to_datetime(df['expect_got_time'], format='%m-%d %H:%M:%S', errors='coerce')
        
        # 过滤无效时间数据
        df = df.dropna(subset=['accept_time', 'got_time'])
        print(f"  有效时间数据: {len(df):,} 条")
        
        # 计算揽件耗时 (分钟): got_time - accept_time
        df['pickup_eta_minutes'] = (df['got_time'] - df['accept_time']).dt.total_seconds() / 60
        
        # 处理跨天情况
        df.loc[df['pickup_eta_minutes'] < 0, 'pickup_eta_minutes'] += 24 * 60
        
        # 清洗异常值 (揽件时间在1分钟~300分钟之间)
        df = df[(df['pickup_eta_minutes'] >= 1) & (df['pickup_eta_minutes'] <= 300)]
        print(f"  清洗后数据: {len(df):,} 条")
        
        # 计算计划准确度 (实际揽件时间 - 预期揽件时间)
        df['plan_accuracy_minutes'] = (df['got_time'] - df['expect_got_time']).dt.total_seconds() / 60
        df.loc[df['plan_accuracy_minutes'] < -720, 'plan_accuracy_minutes'] += 24 * 60
        df.loc[df['plan_accuracy_minutes'] > 720, 'plan_accuracy_minutes'] -= 24 * 60
        
        # 计划偏差特征
        df['plan_deviation'] = (df['got_time'] - df['expect_got_time']).dt.total_seconds() / 60
        df.loc[df['plan_deviation'] < -720, 'plan_deviation'] += 24 * 60
        df.loc[df['plan_deviation'] > 720, 'plan_deviation'] -= 24 * 60
        
        # 计划等待时间
        df['planned_wait_minutes'] = (df['expect_got_time'] - df['accept_time']).dt.total_seconds() / 60
        df.loc[df['planned_wait_minutes'] < 0, 'planned_wait_minutes'] += 24 * 60
        
        # 时间特征提取
        df['hour'] = df['accept_time'].dt.hour
        df['weekday'] = df['accept_time'].dt.dayofweek
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        
        # 高峰时段标记 (7-9点早高峰, 17-19点晚高峰)
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                              (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        
        # 3. 统计分析
        print("\n[3/4] 统计分析...")
        
        # 整体揽件时间统计
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │         揽件时间整体统计                │")
        print("  ├─────────────────────────────────────────┤")
        print(f"  │  平均揽件时间: {df['pickup_eta_minutes'].mean():.2f} 分钟          │")
        print(f"  │  中位数揽件时间: {df['pickup_eta_minutes'].median():.2f} 分钟        │")
        print(f"  │  标准差: {df['pickup_eta_minutes'].std():.2f} 分钟                │")
        print(f"  │  最小值: {df['pickup_eta_minutes'].min():.2f} 分钟                │")
        print(f"  │  最大值: {df['pickup_eta_minutes'].max():.2f} 分钟               │")
        print("  └─────────────────────────────────────────┘")
        
        # 按城市统计
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │         各城市揽件时间统计              │")
        print("  ├─────────────────────────────────────────┤")
        city_stats = df.groupby('from_city_name')['pickup_eta_minutes'].agg(['mean', 'median', 'std', 'count']).round(2)
        for city, row in city_stats.iterrows():
            print(f"  │  {city:8s}  平均: {row['mean']:6.2f}分  中位: {row['median']:6.2f}分  样本: {int(row['count']):5d}  │")
        print("  └─────────────────────────────────────────┘")
        
        # 按时段统计
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │         各时段揽件效率统计              │")
        print("  ├─────────────────────────────────────────┤")
        hour_stats = df.groupby('hour')['pickup_eta_minutes'].agg(['mean', 'count']).round(2)
        for hour, row in hour_stats.iterrows():
            period = "早高峰" if 7 <= hour <= 9 else ("晚高峰" if 17 <= hour <= 19 else ("白天" if 9 < hour < 17 else "夜间"))
            print(f"  │  {hour:2d}:00  [{period:4s}]  平均: {row['mean']:6.2f}分  订单: {int(row['count']):5d}    │")
        print("  └─────────────────────────────────────────┘")
        
        # 高峰 vs 非高峰对比
        rush_hour_stats = df.groupby('is_rush_hour')['pickup_eta_minutes'].agg(['mean', 'median', 'std']).round(2)
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │         高峰/非高峰对比                 │")
        print("  ├─────────────────────────────────────────┤")
        print(f"  │  高峰时段    平均: {rush_hour_stats.loc[1, 'mean']:6.2f}分  中位: {rush_hour_stats.loc[1, 'median']:6.2f}分  │")
        print(f"  │  非高峰时段  平均: {rush_hour_stats.loc[0, 'mean']:6.2f}分  中位: {rush_hour_stats.loc[0, 'median']:6.2f}分  │")
        print("  └─────────────────────────────────────────┘")
        
        # 计划准确度统计
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │         计划准确度统计                  │")
        print("  ├─────────────────────────────────────────┤")
        print(f"  │  平均偏差: {df['plan_accuracy_minutes'].mean():+.2f} 分钟           │")
        print(f"  │  中位数偏差: {df['plan_accuracy_minutes'].median():+.2f} 分钟         │")
        print(f"  │  准时率(±15分钟内): {(abs(df['plan_accuracy_minutes']) <= 15).mean()*100:.1f}%         │")
        print(f"  │  提前率(>15分钟): {(df['plan_accuracy_minutes'] < -15).mean()*100:.1f}%           │")
        print(f"  │  延迟率(>15分钟): {(df['plan_accuracy_minutes'] > 15).mean()*100:.1f}%           │")
        print("  └─────────────────────────────────────────┘")
        
        # 4. 城市-时段热力数据
        print("\n[4/4] 城市-时段揽件效率矩阵...")
        print("\n  各城市各时段平均揽件时间(分钟):")
        city_hour_pivot = df.pivot_table(
            values='pickup_eta_minutes', 
            index='from_city_name', 
            columns='hour', 
            aggfunc='mean'
        ).round(1)
        print("  " + "-" * 80)
        print("  城市      ", end="")
        for h in range(6, 22, 2):
            print(f"{h:>6d}时", end="")
        print()
        print("  " + "-" * 80)
        for city in city_hour_pivot.index:
            print(f"  {city:8s}  ", end="")
            for h in range(6, 22, 2):
                val = city_hour_pivot.loc[city, h] if h in city_hour_pivot.columns else 0
                print(f"{val:>7.1f}", end="")
            print()
        print("  " + "-" * 80)

        # 样本城市分布分析
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │         城市样本分布                   │")
        print("  ├─────────────────────────────────────────┤")
        city_counts = df['from_city_name'].value_counts()
        total = len(df)
        for city, count in city_counts.items():
            pct = count / total * 100
            bar = '█' * int(pct / 2)
            print(f"  │  {city:8s} {count:5d} ({pct:5.1f}%) {bar:<25} │")
        print("  └─────────────────────────────────────────┘")

        # 时段样本分布
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │         时段样本分布                   │")
        print("  ├─────────────────────────────────────────┤")
        hour_counts = df.groupby('hour').size()
        max_count = hour_counts.max()
        for hour in range(6, 21):
            count = hour_counts.get(hour, 0)
            pct = count / total * 100
            bar = '█' * int(pct)
            period = "早高峰" if 7 <= hour <= 9 else ("晚高峰" if 17 <= hour <= 19 else ("白天" if 9 < hour < 17 else "夜间"))
            print(f"  │  {hour:2d}:00 [{period:4s}] {count:5d} ({pct:4.1f}%) {bar:<20} │")
        print("  └─────────────────────────────────────────┘")

        print("\n[Pickup Analysis] 揽件数据分析完成!")
        return df
        
    except Exception as e:
        print(f"\n[Pickup Analysis Error] {e}")
        import traceback
        traceback.print_exc()
        return None


# ============== 轨迹数据分析 ==============

def trajectory_analysis(max_rows=500000):
    """快递员轨迹数据分析主函数"""
    print("\n" + "=" * 60)
    print("Trajectory Analysis - 快递员轨迹数据分析")
    print("=" * 60)
    
    try:
        DATA_PATH = './courier_detailed_trajectory.csv'
        
        # 1. 数据加载（限制行数以控制内存）
        print(f"\n[1/4] 加载轨迹数据（限制 {max_rows:,} 行）...")
        df = pd.read_csv(DATA_PATH, nrows=max_rows)
        print(f"  加载数据量: {len(df):,} 条")
        
        # 2. 基本统计
        print("\n[2/4] 基本统计信息...")
        total_points = len(df)
        unique_couriers = df['postman_id'].nunique()
        date_range = df['ds'].unique()
        
        print("  ┌─────────────────────────────────────────┐")
        print("  │         轨迹数据基本统计                │")
        print("  ├─────────────────────────────────────────┤")
        print(f"  │  总轨迹点数: {total_points:>10,}                    │")
        print(f"  │  快递员数量: {unique_couriers:>10,}                    │")
        print(f"  │  日期范围: ds={min(date_range)} 到 ds={max(date_range)}           │")
        print("  └─────────────────────────────────────────┘")
        
        # 3. 按快递员统计
        print("\n[3/4] 快递员级别统计...")
        
        # 解析时间
        df['gps_time'] = pd.to_datetime(df['gps_time'], format='%m-%d %H:%M:%S', errors='coerce')
        
        # 按快递员和日期分组计算统计信息
        courier_daily = df.groupby(['postman_id', 'ds']).agg(
            first_gps=('gps_time', 'min'),
            last_gps=('gps_time', 'max'),
            point_count=('gps_time', 'count'),
            avg_lat=('lat', 'mean'),
            avg_lng=('lng', 'mean'),
            min_lat=('lat', 'min'),
            max_lat=('lat', 'max'),
            min_lng=('lng', 'min'),
            max_lng=('lng', 'max')
        ).reset_index()
        
        # 计算每日活跃时长（小时）
        courier_daily['active_hours'] = (
            courier_daily['last_gps'] - courier_daily['first_gps']
        ).dt.total_seconds() / 3600
        
        # 处理跨天情况
        courier_daily.loc[courier_daily['active_hours'] < 0, 'active_hours'] += 24
        
        # 计算活动范围（米，EPSG:3857坐标）
        courier_daily['lat_range'] = courier_daily['max_lat'] - courier_daily['min_lat']
        courier_daily['lng_range'] = courier_daily['max_lng'] - courier_daily['min_lng']
        
        # 按快递员聚合
        courier_stats = courier_daily.groupby('postman_id').agg(
            total_days=('ds', 'nunique'),
            avg_daily_points=('point_count', 'mean'),
            avg_active_hours=('active_hours', 'mean'),
            max_active_hours=('active_hours', 'max'),
            avg_lat_range=('lat_range', 'mean'),
            avg_lng_range=('lng_range', 'mean')
        ).reset_index()
        
        print("  ┌─────────────────────────────────────────┐")
        print("  │      快递员工作统计（Top 10）           │")
        print("  ├─────────────────────────────────────────┤")
        print("  │ 快递员ID              │ 活跃天数 │ 日均点数 │ 日均时长 │")
        print("  ├─────────────────────────────────────────┤")
        
        top_couriers = courier_stats.nlargest(10, 'avg_daily_points')
        for _, row in top_couriers.iterrows():
            postman_id_str = str(row['postman_id'])[:15]
            print(f"  │ {postman_id_str:<15s} │ {int(row['total_days']):>8d} │ "
                  f"{row['avg_daily_points']:>8.0f} │ {row['avg_active_hours']:>8.1f}h │")
        print("  └─────────────────────────────────────────┘")
        
        # 4. 按时段统计轨迹密度
        print("\n[4/4] 时段轨迹密度分析...")
        df['hour'] = df['gps_time'].dt.hour
        hour_density = df.groupby('hour').size().reset_index(name='point_count')
        
        print("  ┌─────────────────────────────────────────┐")
        print("  │         各时段轨迹密度                  │")
        print("  ├─────────────────────────────────────────┤")
        for _, row in hour_density.iterrows():
            hour = int(row['hour'])
            count = int(row['point_count'])
            bar = '█' * int(count / max(hour_density['point_count']) * 30)
            period = "早高峰" if 7 <= hour <= 9 else ("晚高峰" if 17 <= hour <= 19 else 
                                                     ("白天" if 9 < hour < 17 else "夜间"))
            print(f"  │  {hour:2d}:00 [{period:4s}] {count:>7,} {bar:<30} │")
        print("  └─────────────────────────────────────────┘")
        
        # 总体统计
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │         轨迹数据总体统计                │")
        print("  ├─────────────────────────────────────────┤")
        print(f"  │  平均每日轨迹点数: {courier_daily['point_count'].mean():.0f}                   │")
        print(f"  │  平均每日活跃时长: {courier_daily['active_hours'].mean():.1f} 小时                │")
        print(f"  │  平均活动纬度范围: {courier_stats['avg_lat_range'].mean():.0f} 米                 │")
        print(f"  │  平均活动经度范围: {courier_stats['avg_lng_range'].mean():.0f} 米                 │")
        print("  └─────────────────────────────────────────┘")
        
        # 5. 轨迹过滤效果分析
        print("\n[5/5] 轨迹过滤效果分析...")
        
        # 计算过滤统计
        total_points_before = 0
        total_points_after = 0
        total_stationary = 0
        total_outliers = 0
        total_rest_periods = 0
        
        for postman_id, group in df.groupby('postman_id'):
            group = group.sort_values('gps_time').reset_index(drop=True)
            
            if len(group) < 2:
                continue
            
            total_points_before += len(group)
            
            # 计算相邻点间的距离和速度
            group['lat_diff'] = group['lat'].diff().abs()
            group['lng_diff'] = group['lng'].diff().abs()
            group['distance_m'] = np.sqrt(group['lat_diff']**2 + group['lng_diff']**2)
            group['time_diff_sec'] = group['gps_time'].diff().dt.total_seconds()
            group['speed_m_s'] = np.where(
                group['time_diff_sec'] > 0, 
                group['distance_m'] / group['time_diff_sec'], 
                0
            )
            
            # 过滤标记
            group['is_stationary'] = group['speed_m_s'] < 0.5
            group['is_outlier'] = group['speed_m_s'] > 30
            group['is_rest'] = group['time_diff_sec'] > 1800
            
            # 统计
            total_stationary += group['is_stationary'].sum()
            total_outliers += group['is_outlier'].sum()
            total_rest_periods += group['is_rest'].sum()
            
            # 有效点（非静止且非离群）
            valid_mask = (~group['is_stationary']) & (~group['is_outlier'])
            total_points_after += valid_mask.sum()
        
        # 显示过滤统计
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │         轨迹点过滤统计                  │")
        print("  ├─────────────────────────────────────────┤")
        print(f"  │  原始轨迹点总数: {total_points_before:>12,}              │")
        print(f"  │  过滤后有效点数: {total_points_after:>12,}              │")
        print(f"  │  过滤比例: {((total_points_before - total_points_after) / total_points_before * 100) if total_points_before > 0 else 0:>15.1f}%              │")
        print("  ├─────────────────────────────────────────┤")
        print(f"  │  静止点 (v<0.5m/s): {total_stationary:>10,} ({total_stationary/total_points_before*100 if total_points_before > 0 else 0:.1f}%)        │")
        print(f"  │  GPS漂移 (v>30m/s): {total_outliers:>10,} ({total_outliers/total_points_before*100 if total_points_before > 0 else 0:.1f}%)        │")
        print(f"  │  休息段 (>30分钟): {total_rest_periods:>11,} ({total_rest_periods/total_points_before*100 if total_points_before > 0 else 0:.1f}%)        │")
        print("  └─────────────────────────────────────────┘")
        
        # 真实工作时长 vs 原始时长对比
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │    真实工作时长 vs 原始时长对比         │")
        print("  ├─────────────────────────────────────────┤")
        
        # 计算真实工作时长的样本
        actual_hours_samples = []
        raw_hours_samples = []
        
        for postman_id, group in df.groupby('postman_id'):
            group = group.sort_values('gps_time').reset_index(drop=True)
            
            if len(group) < 2:
                continue
            
            # 原始时长
            raw_hours = (group['gps_time'].max() - group['gps_time'].min()).total_seconds() / 3600
            if raw_hours < 0:
                raw_hours += 24
            
            # 真实时长（排除休息段）
            group['time_diff_sec'] = group['gps_time'].diff().dt.total_seconds()
            group['is_rest'] = group['time_diff_sec'] > 1800
            active_time_sec = group.loc[~group['is_rest'], 'time_diff_sec'].dropna().clip(lower=0).sum()
            actual_hours = active_time_sec / 3600
            
            actual_hours_samples.append(actual_hours)
            raw_hours_samples.append(raw_hours)
        
        if actual_hours_samples:
            print(f"  │  原始时长均值: {np.mean(raw_hours_samples):>8.1f} 小时                    │")
            print(f"  │  真实时长均值: {np.mean(actual_hours_samples):>8.1f} 小时                    │")
            print(f"  │  休息占比: {((np.mean(raw_hours_samples) - np.mean(actual_hours_samples)) / np.mean(raw_hours_samples) * 100):>11.1f}%                    │")
        print("  └─────────────────────────────────────────┘")
        
        print("\n[Trajectory Analysis] 轨迹数据分析完成!")
        return courier_stats
        
    except FileNotFoundError:
        print(f"\n[Trajectory Analysis] 轨迹文件未找到: {DATA_PATH}")
        print("  跳过轨迹分析...")
        return None
    except Exception as e:
        print(f"\n[Trajectory Analysis Error] {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_courier_features(trajectory_path, max_rows=500000):
    """
    从轨迹数据提取快递员特征，用于增强 ETA 模型
    
    改进功能：
    - 静止点过滤：速度 < 0.5 m/s 视为静止点
    - GPS漂移过滤：速度 > 30 m/s 视为离群点
    - 休息段检测：相邻点时间间隔 > 30 分钟
    - 真实工作时长：排除休息段后的实际活跃时间
    
    Args:
        trajectory_path: 轨迹文件路径
        max_rows: 最大读取行数（控制内存）
    
    Returns:
        DataFrame: 快递员特征表，包含 driver_id 和各项特征
    """
    try:
        # 读取数据（限制行数）
        traj_df = pd.read_csv(trajectory_path, nrows=max_rows)
        
        if len(traj_df) == 0:
            print("[extract_courier_features] 轨迹数据为空")
            return None
        
        # 解析时间
        traj_df['gps_time'] = pd.to_datetime(
            traj_df['gps_time'], format='%m-%d %H:%M:%S', errors='coerce'
        )
        
        # 过滤无效时间数据
        traj_df = traj_df.dropna(subset=['gps_time'])
        
        # 按快递员分组计算特征（带静止点过滤）
        courier_features_list = []
        
        for postman_id, group in traj_df.groupby('postman_id'):
            group = group.sort_values('gps_time').reset_index(drop=True)
            
            if len(group) < 2:
                # 轨迹点太少，使用默认值
                courier_features_list.append({
                    'driver_id': postman_id,
                    'courier_total_points': len(group),
                    'courier_active_hours': 0,
                    'courier_actual_active_hours': 0,
                    'courier_avg_lat': group['lat'].mean() if len(group) > 0 else 0,
                    'courier_avg_lng': group['lng'].mean() if len(group) > 0 else 0,
                    'courier_lat_range': 0,
                    'courier_lng_range': 0,
                    'courier_avg_speed': 0,
                    'courier_max_speed': 0,
                    'courier_stationary_points_pct': 0,
                    'courier_moving_distance_km': 0
                })
                continue
            
            # 计算相邻点位移（EPSG:3857坐标，单位米）
            group['lat_diff'] = group['lat'].diff().abs()
            group['lng_diff'] = group['lng'].diff().abs()
            group['distance_m'] = np.sqrt(group['lat_diff']**2 + group['lng_diff']**2)
            group['time_diff_sec'] = group['gps_time'].diff().dt.total_seconds()
            
            # 计算速度（米/秒）
            group['speed_m_s'] = np.where(
                group['time_diff_sec'] > 0, 
                group['distance_m'] / group['time_diff_sec'], 
                0
            )
            
            # 过滤标记
            group['is_stationary'] = group['speed_m_s'] < 0.5  # 静止点: < 0.5 m/s
            group['is_outlier'] = group['speed_m_s'] > 30      # GPS漂移: > 30 m/s
            group['is_rest'] = group['time_diff_sec'] > 1800   # 休息段: > 30分钟
            
            # 有效速度（排除静止和离群）
            valid_mask = (~group['is_stationary']) & (~group['is_outlier'])
            valid_speeds = group.loc[valid_mask, 'speed_m_s']
            
            # 运动距离（km）- 只计算有效移动
            moving_distance_km = group.loc[valid_mask, 'distance_m'].sum() / 1000
            
            # 原始活跃时长（小时）
            raw_active_hours = (group['gps_time'].max() - group['gps_time'].min()).total_seconds() / 3600
            if raw_active_hours < 0:
                raw_active_hours += 24  # 处理跨天
            
            # 真实活跃时长（排除休息段）
            active_time_sec = group.loc[~group['is_rest'], 'time_diff_sec'].dropna().clip(lower=0).sum()
            actual_active_hours = active_time_sec / 3600
            
            # 静止点占比
            stationary_pct = group['is_stationary'].mean() * 100
            
            courier_features_list.append({
                'driver_id': postman_id,
                'courier_total_points': len(group),
                'courier_active_hours': raw_active_hours,
                'courier_actual_active_hours': actual_active_hours,
                'courier_avg_lat': group['lat'].mean(),
                'courier_avg_lng': group['lng'].mean(),
                'courier_lat_range': group['lat'].max() - group['lat'].min(),
                'courier_lng_range': group['lng'].max() - group['lng'].min(),
                'courier_avg_speed': valid_speeds.mean() if len(valid_speeds) > 0 else 0,
                'courier_max_speed': valid_speeds.max() if len(valid_speeds) > 0 else 0,
                'courier_stationary_points_pct': stationary_pct,
                'courier_moving_distance_km': moving_distance_km
            })
        
        courier_features = pd.DataFrame(courier_features_list)
        
        print(f"[extract_courier_features] 提取了 {len(courier_features)} 个快递员的特征")
        print(f"  - 包含静止点占比、实际运动距离、真实工作时长等新特征")
        return courier_features
        
    except FileNotFoundError:
        print(f"[extract_courier_features] 轨迹文件未找到: {trajectory_path}")
        return None
    except Exception as e:
        print(f"[extract_courier_features] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============== Roads数据分析 ==============

def roads_analysis():
    """道路网络分析主函数"""
    print("\n" + "=" * 60)
    print("Roads Analysis - 道路网络分析")
    print("=" * 60)
    
    try:
        DATA_PATH = './roads.csv'
        
        # 1. 数据加载
        print("\n[1/3] 加载道路数据...")
        df = pd.read_csv(DATA_PATH, sep='\t')
        print(f"  道路总数: {len(df):,} 条")
        
        # 2. 数据预处理
        print("\n[2/3] 数据预处理...")
        
        # 处理maxspeed列 - 提取数字
        def parse_maxspeed(val):
            if pd.isna(val) or val == '' or val == '0':
                return np.nan
            try:
                # 尝试直接转换
                return float(val)
            except:
                # 提取数字部分
                import re
                nums = re.findall(r'\d+', str(val))
                if nums:
                    return float(nums[0])
                return np.nan
        
        df['maxspeed_num'] = df['maxspeed'].apply(parse_maxspeed)

        # 缺失率统计
        missing_before = df['maxspeed_num'].isna().sum()
        missing_pct = missing_before / len(df) * 100
        print(f"  原始maxspeed缺失率: {missing_pct:.1f}% ({missing_before:,} 条)")

        # 清洗异常速度值 (0-200 km/h 为合理范围)
        df.loc[(df['maxspeed_num'] <= 0) | (df['maxspeed_num'] > 200), 'maxspeed_num'] = np.nan

        # 填充缺失速度 (按道路类型中位数填充)
        speed_by_fclass = df.groupby('fclass')['maxspeed_num'].median()
        for fclass in df['fclass'].unique():
            mask = (df['fclass'] == fclass) & (df['maxspeed_num'].isna())
            if fclass in speed_by_fclass.index and not pd.isna(speed_by_fclass[fclass]):
                df.loc[mask, 'maxspeed_num'] = speed_by_fclass[fclass]
            else:
                df.loc[mask, 'maxspeed_num'] = 40

        missing_after = df['maxspeed_num'].isna().sum()
        print(f"  填充后缺失: {missing_after:,} 条")
        
        # 3. 统计分析
        print("\n[3/3] 道路网络统计...")
        
        # 按城市统计
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │         各城市道路统计                  │")
        print("  ├─────────────────────────────────────────┤")
        city_stats = df.groupby('city').agg({
            'osm_id': 'count',
            'maxspeed_num': 'mean'
        }).round(2)
        city_stats.columns = ['道路数', '平均限速(km/h)']
        for city, row in city_stats.iterrows():
            print(f"  │  {city:8s}  道路数: {int(row['道路数']):6,}  平均限速: {row['平均限速(km/h)']:5.1f} km/h  │")
        print("  └─────────────────────────────────────────┘")
        
        # 按道路类型统计
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │         各类型道路统计                  │")
        print("  ├─────────────────────────────────────────┤")
        fclass_stats = df.groupby('fclass').agg({
            'osm_id': 'count',
            'maxspeed_num': ['mean', 'median']
        }).round(2)
        fclass_stats.columns = ['数量', '平均限速', '中位限速']
        fclass_stats = fclass_stats.sort_values('数量', ascending=False)
        
        total_roads = len(df)
        for fclass, row in fclass_stats.iterrows():
            pct = row['数量'] / total_roads * 100
            print(f"  │  {fclass:12s} 数量: {int(row['数量']):6,} ({pct:5.1f}%)  限速: {row['平均限速']:5.1f} km/h │")
        print("  └─────────────────────────────────────────┘")
        
        # 道路类型分布 (按城市)
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │      各城市主要道路类型分布             │")
        print("  ├─────────────────────────────────────────┤")
        city_fclass = df.groupby(['city', 'fclass']).size().unstack(fill_value=0)
        main_types = ['primary', 'secondary', 'tertiary', 'service', 'residential']
        
        for city in city_fclass.index:
            print(f"  │  {city:8s}:", end="")
            for fclass in main_types:
                if fclass in city_fclass.columns:
                    count = city_fclass.loc[city, fclass]
                    print(f" {fclass[:3]}:{count:5,}", end="")
            print("  │")
        print("  └─────────────────────────────────────────┘")
        
        # 单向道路统计
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │         单向道路统计                    │")
        print("  ├─────────────────────────────────────────┤")
        oneway_stats = df['oneway'].value_counts()
        for val, count in oneway_stats.items():
            pct = count / total_roads * 100
            desc = "双向" if val == 'F' else ("单向" if val == 'T' else ("反向" if val == 'B' else val))
            print(f"  │  {desc:8s} ({val}): {count:7,} 条 ({pct:5.1f}%)              │")
        print("  └─────────────────────────────────────────┘")

        # 道路基础设施评估
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │      城市道路基础设施评估               │")
        print("  ├─────────────────────────────────────────┤")
        city_road_stats = df.groupby('city').agg({
            'osm_id': 'count',
            'maxspeed_num': ['mean', 'median']
        }).round(2)
        city_road_stats.columns = ['道路数', '平均限速', '中位限速']
        city_road_stats['高速率道路占比'] = df.groupby('city').apply(
            lambda x: (x['maxspeed_num'] >= 60).mean() * 100
        ).round(1)
        for city, row in city_road_stats.iterrows():
            print(f"  │  {city:8s} 道路:{int(row['道路数']):6,}  均速:{row['平均限速']:5.1f}km/h  高速率占比:{row['高速率道路占比']:5.1f}% │")
        print("  └─────────────────────────────────────────┘")

        print("\n[Roads Analysis] 道路网络分析完成!")
        return df
        
    except Exception as e:
        print(f"\n[Roads Analysis Error] {e}")
        import traceback
        traceback.print_exc()
        return None


# ============== 综合分析 ==============

def comprehensive_analysis():
    """综合分析 - 整合delivery、pickup和roads数据"""
    print("\n" + "=" * 60)
    print("Comprehensive Analysis - 综合分析")
    print("=" * 60)
    
    try:
        # 1. 加载三个数据源
        print("\n[1/3] 加载各数据源...")
        
        # Delivery数据
        delivery_df = pd.read_csv('./delivery_five_cities.csv', nrows=50000)
        delivery_df['receipt_time'] = pd.to_datetime(delivery_df['receipt_time'], format='%m-%d %H:%M:%S', errors='coerce')
        delivery_df['sign_time'] = pd.to_datetime(delivery_df['sign_time'], format='%m-%d %H:%M:%S', errors='coerce')
        delivery_df['delivery_eta'] = (delivery_df['sign_time'] - delivery_df['receipt_time']).dt.total_seconds() / 60
        delivery_df.loc[delivery_df['delivery_eta'] < 0, 'delivery_eta'] += 24 * 60
        delivery_df = delivery_df[(delivery_df['delivery_eta'] >= 5) & (delivery_df['delivery_eta'] <= 180)]
        
        # Pickup数据
        pickup_df = pd.read_csv('./pickup_five_cities.csv', nrows=50000)
        pickup_df['accept_time'] = pd.to_datetime(pickup_df['accept_time'], format='%m-%d %H:%M:%S', errors='coerce')
        pickup_df['got_time'] = pd.to_datetime(pickup_df['got_time'], format='%m-%d %H:%M:%S', errors='coerce')
        pickup_df['pickup_eta'] = (pickup_df['got_time'] - pickup_df['accept_time']).dt.total_seconds() / 60
        pickup_df.loc[pickup_df['pickup_eta'] < 0, 'pickup_eta'] += 24 * 60
        pickup_df = pickup_df[(pickup_df['pickup_eta'] >= 1) & (pickup_df['pickup_eta'] <= 300)]
        
        # Roads数据
        roads_df = pd.read_csv('./roads.csv', sep='\t')
        
        print(f"  Delivery数据: {len(delivery_df):,} 条")
        print(f"  Pickup数据: {len(pickup_df):,} 条")
        print(f"  Roads数据: {len(roads_df):,} 条")
        
        # 2. 按城市聚合统计
        print("\n[2/3] 按城市维度聚合...")
        
        # Delivery按城市统计
        delivery_city = delivery_df.groupby('from_city_name')['delivery_eta'].agg(['mean', 'median', 'count']).reset_index()
        delivery_city.columns = ['city', 'delivery_mean', 'delivery_median', 'delivery_count']
        
        # Pickup按城市统计
        pickup_city = pickup_df.groupby('from_city_name')['pickup_eta'].agg(['mean', 'median', 'count']).reset_index()
        pickup_city.columns = ['city', 'pickup_mean', 'pickup_median', 'pickup_count']
        
        # Roads按城市统计
        roads_city = roads_df.groupby('city').agg({
            'osm_id': 'count',
            'maxspeed': lambda x: pd.to_numeric(x, errors='coerce').mean()
        }).reset_index()
        roads_city.columns = ['city', 'road_count', 'avg_maxspeed']
        
        # 3. 合并分析
        print("\n[3/3] 生成综合分析报告...")
        
        # 合并三个数据源
        merged = delivery_city.merge(pickup_city, on='city', how='outer')
        merged = merged.merge(roads_city, on='city', how='outer')
        
        print("\n  ┌──────────────────────────────────────────────────────────────────────────┐")
        print("  │                     城市综合物流效率分析报告                              │")
        print("  ├──────────────────────────────────────────────────────────────────────────┤")
        print("  │  城市    │ 配送时间 │ 揽件时间 │ 道路数  │ 平均限速 │ 配送单量 │ 揽件单量 │")
        print("  │          │  (分钟)  │  (分钟)  │         │  (km/h)  │          │          │")
        print("  ├──────────────────────────────────────────────────────────────────────────┤")
        
        for _, row in merged.iterrows():
            city = row['city'] if pd.notna(row['city']) else 'Unknown'
            d_time = f"{row['delivery_mean']:.1f}" if pd.notna(row['delivery_mean']) else 'N/A'
            p_time = f"{row['pickup_mean']:.1f}" if pd.notna(row['pickup_mean']) else 'N/A'
            r_count = f"{int(row['road_count']):,}" if pd.notna(row['road_count']) else 'N/A'
            r_speed = f"{row['avg_maxspeed']:.1f}" if pd.notna(row['avg_maxspeed']) else 'N/A'
            d_count = f"{int(row['delivery_count']):,}" if pd.notna(row['delivery_count']) else 'N/A'
            p_count = f"{int(row['pickup_count']):,}" if pd.notna(row['pickup_count']) else 'N/A'
            
            print(f"  │ {city:8s} │ {d_time:>8s} │ {p_time:>8s} │ {r_count:>7s} │ {r_speed:>8s} │ {d_count:>8s} │ {p_count:>8s} │")
        
        print("  └──────────────────────────────────────────────────────────────────────────┘")
        
        # 计算效率指标
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │         城市物流效率指标                │")
        print("  ├─────────────────────────────────────────┤")
        
        for _, row in merged.iterrows():
            city = row['city'] if pd.notna(row['city']) else 'Unknown'
            
            # 综合效率评分 (越低越好)
            efficiency_score = 0
            if pd.notna(row['delivery_mean']) and pd.notna(row['pickup_mean']):
                efficiency_score = (row['delivery_mean'] + row['pickup_mean']) / 2
                
                # 道路密度指标 (道路数/订单量)
                total_orders = 0
                if pd.notna(row['delivery_count']):
                    total_orders += row['delivery_count']
                if pd.notna(row['pickup_count']):
                    total_orders += row['pickup_count']
                
                road_density = row['road_count'] / total_orders if total_orders > 0 and pd.notna(row['road_count']) else 0
                
                print(f"  │  {city:8s}                              │")
                print(f"  │    综合时效: {efficiency_score:.1f} 分钟/单                │")
                print(f"  │    道路密度: {road_density:.4f} (道路数/订单数)          │")
                if pd.notna(row['avg_maxspeed']):
                    print(f"  │    道路质量: {row['avg_maxspeed']:.1f} km/h 平均限速        │")
                print("  │                                         │")
        
        print("  └─────────────────────────────────────────┘")
        
        # 总体对比
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │         五城市总体对比                  │")
        print("  ├─────────────────────────────────────────┤")
        
        avg_delivery = merged['delivery_mean'].mean()
        avg_pickup = merged['pickup_mean'].mean()
        total_roads = merged['road_count'].sum()
        total_delivery = merged['delivery_count'].sum()
        total_pickup = merged['pickup_count'].sum()
        
        print(f"  │  平均配送时间: {avg_delivery:.1f} 分钟                      │")
        print(f"  │  平均揽件时间: {avg_pickup:.1f} 分钟                      │")
        print(f"  │  道路总数: {int(total_roads):,} 条                          │")
        print(f"  │  配送订单总数: {int(total_delivery):,} 条                     │")
        print(f"  │  揽件订单总数: {int(total_pickup):,} 条                     │")
        print("  └─────────────────────────────────────────┘")
        
        print("\n[Comprehensive Analysis] 综合分析完成!")
        return merged
        
    except Exception as e:
        print(f"\n[Comprehensive Analysis Error] {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_realtime_workload(delivery_path, max_samples=50000):
    """
    计算快递员实时负载（在手订单数）
    
    对每个快递员，计算其在每个订单 receipt_time 时刻的并行订单数（在手订单数）
    
    Args:
        delivery_path: 配送数据文件路径
        max_samples: 最大读取样本数（控制计算量）
    
    Returns:
        DataFrame: 快递员负载统计，包含 driver_id, avg_workload, max_workload, std_workload
    """
    try:
        df = pd.read_csv(delivery_path, nrows=max_samples)
        df['receipt_time'] = pd.to_datetime(df['receipt_time'], format='%m-%d %H:%M:%S', errors='coerce')
        df['sign_time'] = pd.to_datetime(df['sign_time'], format='%m-%d %H:%M:%S', errors='coerce')
        df = df.dropna(subset=['receipt_time', 'sign_time'])
        
        # 处理跨天情况：sign_time < receipt_time 时加24小时
        mask = df['sign_time'] < df['receipt_time']
        df.loc[mask, 'sign_time'] = df.loc[mask, 'sign_time'] + pd.Timedelta(days=1)
        
        # 按快递员计算并行订单数
        workload_stats = []
        
        # 性能优化：限制处理的快递员数量
        driver_counts = df['delivery_user_id'].value_counts()
        # 只处理订单数适中的快递员（5-100单），排除极端值
        valid_drivers = driver_counts[(driver_counts >= 5) & (driver_counts <= 100)].index[:200]
        
        for driver_id in valid_drivers:
            driver_orders = df[df['delivery_user_id'] == driver_id].copy()
            
            if len(driver_orders) < 2:
                workload_stats.append({
                    'driver_id': driver_id, 
                    'avg_workload': 1, 
                    'max_workload': 1, 
                    'std_workload': 0
                })
                continue
            
            # 使用向量化计算提高效率
            # 将时间转换为数值（秒）
            receipt_times = driver_orders['receipt_time'].astype('int64') // 10**9
            sign_times = driver_orders['sign_time'].astype('int64') // 10**9
            
            # 抽样计算
            sample_size = min(30, len(driver_orders))
            sample_indices = np.random.choice(len(driver_orders), sample_size, replace=False)
            
            concurrent_counts = []
            for idx in sample_indices:
                t = receipt_times.iloc[idx]
                # 向量化计算并行订单数
                in_hand = ((receipt_times <= t) & (sign_times >= t)).sum()
                concurrent_counts.append(in_hand)
            
            workload_stats.append({
                'driver_id': driver_id,
                'avg_workload': np.mean(concurrent_counts),
                'max_workload': np.max(concurrent_counts),
                'std_workload': np.std(concurrent_counts)
            })
        
        return pd.DataFrame(workload_stats)
        
    except Exception as e:
        print(f"[calculate_realtime_workload] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============== VRP部分（可选） ==============

class VRPCustomNeighborhoodOperators:
    """
    自定义邻域算子集合
    用于提升VRP求解的解质量
    """

    def __init__(self, data):
        self.data = data
        self.num_clients = len(data.clients)

    def segment_swap(self, solution, route1_idx, route2_idx,
                     start1, end1, start2, end2):
        """
        段交换算子 (Segment Swap)
        在两条路线之间交换连续的客户段
        """
        routes = [list(r) for r in solution.get_routes()]

        if route1_idx >= len(routes) or route2_idx >= len(routes):
            return solution

        route1 = routes[route1_idx]
        route2 = routes[route2_idx]

        segment1 = route1[start1:end1+1]
        segment2 = route2[start2:end2+1]

        route1[start1:start1] = segment2
        route2[start2:start2] = segment1

        from pyvrp import Solution
        return Solution(routes)

    def cross_exchange(self, solution):
        """
        交叉交换算子 (Cross Exchange)
        在两条路线之间交换单个客户，保持路线平衡
        """
        routes = [list(r) for r in solution.get_routes()]
        neighbors = []

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if len(routes[i]) == 0 or len(routes[j]) == 0:
                    continue

                for pos_i in range(len(routes[i])):
                    for pos_j in range(len(routes[j])):
                        new_routes = [r.copy() for r in routes]
                        new_routes[i][pos_i], new_routes[j][pos_j] = \
                            new_routes[j][pos_j], new_routes[i][pos_i]

                        try:
                            from pyvrp import Solution
                            neighbors.append(Solution(new_routes))
                        except:
                            continue

        return neighbors

    def relocate_segment(self, solution, route_idx, start, end, new_route_idx, new_pos):
        """
        段迁移算子 (Relocate Segment)
        将一条路线中的连续客户段迁移到另一条路线
        """
        routes = [list(r) for r in solution.get_routes()]

        segment = routes[route_idx][start:end+1]
        del routes[route_idx][start:end+1]
        routes[new_route_idx].insert(new_pos, *segment)

        from pyvrp import Solution
        return Solution(routes)

    def arc_exchange_2opt(self, solution, route_idx, node1, node2):
        """
        弧交换2-Opt (Arc Exchange 2-Opt)
        反转路线的一部分，优化路径交叉
        """
        routes = [list(r) for r in solution.get_routes()]
        route = routes[route_idx]

        if node1 >= len(route) or node2 >= len(route):
            return solution

        pos1 = route.index(node1) if node1 in route else -1
        pos2 = route.index(node2) if node2 in route else -1

        if pos1 == -1 or pos2 == -1 or pos1 >= pos2:
            return solution

        route[pos1:pos2+1] = reversed(route[pos1:pos2+1])

        from pyvrp import Solution
        return Solution(routes)

    def sequential_lambda_opt(self, solution, lambda_val=2):
        """
        顺序λ-OPT算子
        从路线中移除λ个节点并重新插入最佳位置
        """
        routes = [list(r) for r in solution.get_routes()]
        neighbors = []

        for r_idx, route in enumerate(routes):
            if len(route) < lambda_val:
                continue

            for start in range(len(route) - lambda_val + 1):
                removed_nodes = route[start:start + lambda_val]
                remaining = route[:start] + route[start + lambda_val:]

                for insert_pos in range(len(remaining) + 1):
                    new_route = remaining[:insert_pos] + removed_nodes + remaining[insert_pos:]
                    new_routes = [r.copy() for r in routes]
                    new_routes[r_idx] = new_route

                    try:
                        from pyvrp import Solution
                        neighbors.append(Solution(new_routes))
                    except:
                        continue

        return neighbors


class VRPImprovedHGSSolver:
    """
    改进的混合遗传搜索求解器
    集成自定义邻域算子
    """

    def __init__(self, data):
        self.data = data
        self.neighborhood_ops = VRPCustomNeighborhoodOperators(data)

    def local_search(self, solution):
        """
        使用自定义邻域算子进行局部搜索
        搜索顺序：交叉交换 -> 段交换 -> 迁移 -> 2-Opt
        """
        current = solution

        cross_neighbors = self.neighborhood_ops.cross_exchange(current)
        for neighbor in cross_neighbors:
            if neighbor.cost() < current.cost():
                current = neighbor
                break

        for r_idx in range(len(list(current.get_routes()))):
            for n1 in range(self.neighborhood_ops.num_clients):
                for n2 in range(n1 + 1, self.neighborhood_ops.num_clients):
                    neighbor = self.neighborhood_ops.arc_exchange_2opt(
                        current, r_idx, n1, n2
                    )
                    if neighbor.cost() < current.cost():
                        current = neighbor

        lambda_neighbors = self.neighborhood_ops.sequential_lambda_opt(current, 2)
        for neighbor in lambda_neighbors:
            if neighbor.cost() < current.cost():
                current = neighbor
                break

        return current

    def solve(self, max_generations=100, population_size=50, mutation_rate=0.3):
        """
        改进的HGS算法
        """
        import random
        from pyvrp import Model
        from pyvrp.stop import MaxIterations

        print("=" * 60)
        print("改进HGS求解器")
        print("=" * 60)
        print(f"参数: max_generations={max_generations}, "
              f"population_size={population_size}, mutation_rate={mutation_rate}")

        model = Model.load(self.data)
        initial_result = model.solve(stop=MaxIterations(100))
        population = [initial_result]

        best_solution = initial_result
        best_cost = initial_result.cost()

        print(f"初始解成本: {best_cost:.2f}")

        for gen in range(max_generations):
            parent1, parent2 = random.sample(population, 2)

            child = self.order_crossover(parent1, parent2)

            if random.random() < mutation_rate:
                child = self.mutate(child)

            child = self.local_search(child)

            population.append(child)
            if len(population) > population_size:
                population.sort(key=lambda x: x.cost())
                population = population[:population_size]

            if child.cost() < best_cost:
                best_cost = child.cost()
                best_solution = child
                print(f"Generation {gen+1}: 发现更优解! 成本: {best_cost:.2f}")

        return best_solution

    def order_crossover(self, parent1, parent2):
        """顺序交叉算子 (OX)"""
        return parent1 if parent1.cost() < parent2.cost() else parent2

    def mutate(self, solution):
        """变异操作：随机交换两个客户"""
        import random
        from pyvrp import Solution

        routes = [list(r) for r in solution.get_routes()]

        if len(routes) == 0:
            return solution

        route_idx = random.randint(0, len(routes) - 1)
        route = routes[route_idx]

        if len(route) >= 2:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]

        return Solution(routes)


def vrp_compare_algorithms(data):
    """
    算法性能对比实验
    """
    import time
    from pyvrp import Model
    from pyvrp.stop import MaxIterations, MaxTime

    print("\n" + "=" * 60)
    print("算法性能对比")
    print("=" * 60)

    results = {}

    print("\n[1] 基础PyVRP求解...")
    start = time.time()
    model = Model.load(data)
    result1 = model.solve(stop=MaxIterations(500) | MaxTime(30))
    time1 = time.time() - start
    results['基础PyVRP'] = {'cost': result1.cost(), 'time': time1, 'vehicles': result1.num_vehicles()}

    print("[2] 增加迭代次数...")
    start = time.time()
    result2 = model.solve(stop=MaxIterations(2000) | MaxTime(60),
                          nb_iter_no_improvement=200)
    time2 = time.time() - start
    results['增加迭代'] = {'cost': result2.cost(), 'time': time2, 'vehicles': result2.num_vehicles()}

    print("[3] 自定义邻域算子...")
    solver = VRPImprovedHGSSolver(data)
    start = time.time()
    result3 = solver.solve(max_generations=50, population_size=30)
    time3 = time.time() - start
    results['自定义算子'] = {'cost': result3.cost(), 'time': time3, 'vehicles': result3.num_vehicles()}

    print("\n" + "-" * 60)
    print(f"{'算法':<15} {'成本':<12} {'车辆数':<10} {'时间(秒)':<10}")
    print("-" * 60)

    baseline = results['基础PyVRP']['cost']
    for name, res in results.items():
        improvement = (baseline - res['cost']) / baseline * 100
        print(f"{name:<15} {res['cost']:<12.2f} {res['vehicles']:<10} {res['time']:<10.2f}")
        if improvement > 0:
            print(f"{'':>15} 提升: {improvement:.1f}%")

    return results


def vrp_main():
    """VRP路径优化主函数 - 如果pyvrp可用则运行"""
    try:
        from pyvrp import Model
        from pyvrp.stop import MaxIterations, MaxRuntime, MultipleCriteria

        print("\n" + "=" * 60)
        print("Supply Chain AI - PyVRP Routing Optimization")
        print("=" * 60)

        model = Model()

        depot = model.add_depot(x=30, y=40, name="Warehouse")

        clients_data = [
            (20, 25, 15, "C1"),
            (35, 20, 25, "C2"),
            (45, 35, 10, "C3"),
            (15, 45, 30, "C4"),
            (50, 30, 20, "C5"),
        ]

        clients = []
        client_names = {}
        for idx, (x, y, demand, name) in enumerate(clients_data, start=1):
            client = model.add_client(x=x, y=y, delivery=demand, name=name)
            clients.append(client)
            client_names[idx] = name

        model.add_vehicle_type(
            num_available=3,
            capacity=100,
            name="StandardTruck",
            start_depot=depot
        )

        locations = [depot] + clients
        for i, loc1 in enumerate(locations):
            for j, loc2 in enumerate(locations):
                if i != j:
                    distance = np.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)
                    model.add_edge(loc1, loc2, distance=distance, duration=distance)

        stop = MultipleCriteria([MaxIterations(100), MaxRuntime(10)])
        result = model.solve(stop=stop)

        solution = result.best

        print("\nVRP Results:")
        print(f"Cost: {result.cost():.2f}")
        print(f"Distance: {solution.distance():.0f}")
        print(f"Vehicles: {solution.num_routes()}")

        print("\nRoutes:")
        for idx, route in enumerate(solution.routes()):
            visit_names = [client_names[client_id] for client_id in route.visits()]
            route_str = " -> ".join(["Warehouse"] + visit_names + ["Warehouse"])
            print(f"  Vehicle {idx + 1}: {route_str}")

        return result

    except ImportError as e:
        print(f"\n[VRP] pyvrp import error: {e}")
        print("To use VRP, run: pip install pyvrp")
        return None
    except Exception as e:
        print(f"\n[VRP] VRP error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============== 模型推理脚本 ==============

def load_eta_model(model_path, model_type='transformer'):
    """
    加载训练好的ETA预测模型

    Args:
        model_path: 模型文件路径 (.pth 或 .joblib)
        model_type: 模型类型 ('lstm', 'transformer', 'xgboost', 'lightgbm')

    Returns:
        tuple: (model, checkpoint) 或 (model, None)
    """
    print(f"[load_model] Loading model from {model_path}")

    if model_type in ['xgboost', 'lightgbm']:
        import joblib
        checkpoint = joblib.load(model_path)
        return checkpoint, None
    else:
        checkpoint = torch.load(model_path, map_location='cpu')

        if model_type == 'lstm':
            model = LSTMETA(input_dim=12, hidden_dim=128, num_layers=2)
        elif model_type == 'graph2route':
            model = Graph2RouteETA(input_dim=checkpoint.get('feature_dim', 12), hidden_dim=128)
        elif model_type == 'stgnn':
            model = STGNN(input_dim=checkpoint.get('feature_dim', 12), hidden_dim=128)
        else:
            model = TransformerETA(input_dim=12, d_model=128, nhead=8, num_layers=3)

        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        model.eval()

        return model, checkpoint


def predict_eta(model, checkpoint, order_features):
    """
    预测单个订单的ETA

    Args:
        model: 加载的模型或模型字典
        checkpoint: 模型检查点（包含scaler等信息）
        order_features: 订单特征字典

    Returns:
        float: 预测的ETA（分钟）
    """
    if hasattr(model, 'predict'):
        feature_cols = checkpoint.get('feature_cols', [
            'distance_km', 'hour', 'weekday', 'is_weekend', 'time_period', 'rush_hour',
            'weather_encoded', 'traffic_encoded', 'dest_lat', 'dest_lng',
            'driver_avg_eta', 'driver_std_eta'
        ]) if checkpoint else None

        if feature_cols:
            features = np.array([[order_features.get(col, 0) for col in feature_cols]])
            eta = model.predict(features)[0]
        else:
            eta = model.predict(np.array([list(order_features.values())]))[0]
    else:
        scaler = checkpoint.get('scaler') if checkpoint else None
        feature_cols = checkpoint.get('feature_cols', [
            'distance_km', 'hour', 'weekday', 'is_weekend', 'time_period', 'rush_hour',
            'weather_encoded', 'traffic_encoded', 'dest_lat_3857', 'dest_lng_3857',
            'driver_avg_eta', 'driver_std_eta'
        ]) if checkpoint else None

        if scaler and feature_cols:
            features = np.array([[order_features.get(col, 0) for col in feature_cols]])
            features_scaled = scaler.transform(features)
            features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)
            with torch.no_grad():
                eta = model(features_tensor).item()
        else:
            features_tensor = torch.FloatTensor(np.array([list(order_features.values())]))
            with torch.no_grad():
                eta = model(features_tensor.unsqueeze(0)).item()

    return max(5, eta)


def batch_predict_eta(model, checkpoint, orders_df):
    """
    批量预测订单ETA

    Args:
        model: 加载的模型
        checkpoint: 模型检查点
        orders_df: 订单DataFrame

    Returns:
        np.ndarray: 预测的ETA数组
    """
    predictions = []

    for idx, row in orders_df.iterrows():
        order_features = row.to_dict()
        eta = predict_eta(model, checkpoint, order_features)
        predictions.append(eta)

    return np.array(predictions)


# ============== GPU显存优化工具 ==============

class GPUMemoryOptimizer:
    """
    GPU显存优化工具类
    提供梯度累积和混合精度训练支持
    """

    @staticmethod
    def gradient_accumulation_training(model, dataloader, criterion, optimizer,
                                       accumulation_steps=4, device='cuda'):
        """
        梯度累积训练 - 用于显存不足时模拟大批量

        Args:
            model: PyTorch模型
            dataloader: 数据加载器
            criterion: 损失函数
            optimizer: 优化器
            accumulation_steps: 累积步数
            device: 设备

        Returns:
            float: 平均损失
        """
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output.squeeze(), y)
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        return total_loss / len(dataloader)

    @staticmethod
    def mixed_precision_training(model, dataloader, criterion, optimizer, device='cuda'):
        """
        混合精度训练 - 减少显存占用

        Args:
            model: PyTorch模型
            dataloader: 数据加载器
            criterion: 损失函数
            optimizer: 优化器
            device: 设备

        Returns:
            float: 平均损失
        """
        from torch.cuda.amp import autocast, GradScaler

        model.train()
        total_loss = 0
        scaler = GradScaler()
        optimizer.zero_grad()

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            with autocast():
                output = model(X)
                loss = criterion(output.squeeze(), y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    @staticmethod
    def check_gpu_memory():
        """检查GPU显存使用情况"""
        if torch.cuda.is_available():
            print(f"[GPU Memory] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"[GPU Memory] Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            print(f"[GPU Memory] Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        else:
            print("[GPU Memory] CUDA not available")


# ============== PyVRP求解优化 ==============

def optimize_vrp_solution(model, max_time=60, nb_iter_no_improvement=50,
                          weight_wait_time=1.0, weight_time_warp=3.0):
    """
    PyVRP求解优化配置

    Args:
        model: PyVRP Model实例
        max_time: 最大运行时间（秒）
        nb_iter_no_improvement: 无改进最大迭代次数
        weight_wait_time: 时间窗等待惩罚权重
        weight_time_warp: 时间窗违背惩罚权重

    Returns:
        求解结果
    """
    from pyvrp.stop import MaxTime

    result = model.solve(
        stop=MaxTime(max_time),
        nb_iter_no_improvement=nb_iter_no_improvement,
        weight_wait_time=weight_wait_time,
        weight_time_warp=weight_time_warp,
    )
    return result


# ============== 模型保存与部署 ==============

def save_pytorch_model(model, scaler, feature_cols, save_path='./eta_model.pth'):
    """
    保存PyTorch ETA模型

    Args:
        model: PyTorch模型
        scaler: StandardScaler标准化器
        feature_cols: 特征列名列表
        save_path: 保存路径
    """
    torch.save({
        'model_state': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols
    }, save_path)
    print(f"[save_pytorch_model] Model saved to {save_path}")


def save_sklearn_model(model, save_path='./eta_model.joblib'):
    """
    保存sklearn风格模型（XGBoost, LightGBM等）

    Args:
        model: 训练好的模型
        save_path: 保存路径
    """
    import joblib
    joblib.dump(model, save_path)
    print(f"[save_sklearn_model] Model saved to {save_path}")


def load_sklearn_model(model_path):
    """
    加载sklearn风格模型

    Args:
        model_path: 模型路径

    Returns:
        加载的模型
    """
    import joblib
    return joblib.load(model_path)


# ============== 7. 特征选择模块 ==============

class FeatureSelector:
    """
    特征选择模块 - 综合多种策略选出最优特征子集
    
    支持策略：
    1. 互信息回归 (Mutual Information)
    2. F统计量回归 (F-regression)
    3. 树模型特征重要性 (XGBoost/LightGBM)
    4. 相关性过滤 (去除高度相关冗余特征)
    5. 综合投票 (多策略交叉验证)
    """
    
    def __init__(self, X, y, feature_names):
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)
        self.feature_names = list(feature_names)
        self.n_features = self.X.shape[1]
        self.selection_results = {}
    
    def mutual_information_selection(self, k=15):
        """基于互信息的特征选择"""
        print("\n  [MI] 计算互信息得分...")
        mi_scores = mutual_info_regression(self.X, self.y, random_state=42, n_neighbors=5)
        mi_scores = np.nan_to_num(mi_scores, nan=0.0)
        
        ranked_idx = np.argsort(mi_scores)[::-1]
        top_k = ranked_idx[:k]
        
        self.selection_results['mutual_info'] = {
            'scores': {self.feature_names[i]: float(mi_scores[i]) for i in ranked_idx},
            'selected_indices': top_k.tolist(),
            'selected_features': [self.feature_names[i] for i in top_k]
        }
        
        print(f"  [MI] Top-{k} 特征:")
        for i in top_k[:10]:
            print(f"    {self.feature_names[i]:35s} MI={mi_scores[i]:.4f}")
        
        return top_k
    
    def f_regression_selection(self, k=15):
        """基于F统计量的特征选择"""
        print("\n  [F-reg] 计算F统计量...")
        selector = SelectKBest(f_regression, k=min(k, self.n_features))
        selector.fit(self.X, self.y)
        
        f_scores = np.nan_to_num(selector.scores_, nan=0.0)
        ranked_idx = np.argsort(f_scores)[::-1]
        top_k = ranked_idx[:k]
        
        self.selection_results['f_regression'] = {
            'scores': {self.feature_names[i]: float(f_scores[i]) for i in ranked_idx},
            'selected_indices': top_k.tolist(),
            'selected_features': [self.feature_names[i] for i in top_k]
        }
        
        print(f"  [F-reg] Top-{k} 特征:")
        for i in top_k[:10]:
            print(f"    {self.feature_names[i]:35s} F={f_scores[i]:.2f}")
        
        return top_k
    
    def tree_importance_selection(self, k=15):
        """基于树模型特征重要性的特征选择"""
        print("\n  [Tree] 计算树模型特征重要性...")
        importances = {}
        
        try:
            from xgboost import XGBRegressor
            xgb = XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
            xgb.fit(self.X, self.y)
            importances['xgboost'] = xgb.feature_importances_
            print("    XGBoost 特征重要性计算完成")
        except ImportError:
            print("    [跳过] XGBoost 未安装")
        
        try:
            from lightgbm import LGBMRegressor
            lgbm = LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
            lgbm.fit(self.X, self.y)
            importances['lightgbm'] = lgbm.feature_importances_.astype(float)
            # Normalize LightGBM importance
            total = importances['lightgbm'].sum()
            if total > 0:
                importances['lightgbm'] = importances['lightgbm'] / total
            print("    LightGBM 特征重要性计算完成")
        except ImportError:
            print("    [跳过] LightGBM 未安装")
        
        if not importances:
            print("    [警告] 无树模型可用，使用F回归替代")
            return self.f_regression_selection(k)
        
        # 平均重要性
        avg_importance = np.zeros(self.n_features)
        for imp in importances.values():
            avg_importance += imp
        avg_importance /= len(importances)
        
        ranked_idx = np.argsort(avg_importance)[::-1]
        top_k = ranked_idx[:k]
        
        self.selection_results['tree_importance'] = {
            'scores': {self.feature_names[i]: float(avg_importance[i]) for i in ranked_idx},
            'selected_indices': top_k.tolist(),
            'selected_features': [self.feature_names[i] for i in top_k]
        }
        
        print(f"  [Tree] Top-{k} 特征:")
        for i in top_k[:10]:
            print(f"    {self.feature_names[i]:35s} Imp={avg_importance[i]:.4f}")
        
        return top_k
    
    def correlation_filter(self, threshold=0.95):
        """去除高度相关的冗余特征"""
        print(f"\n  [Corr] 过滤相关系数 > {threshold} 的冗余特征...")
        corr_matrix = np.corrcoef(self.X.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        to_drop = set()
        for i in range(self.n_features):
            if i in to_drop:
                continue
            for j in range(i + 1, self.n_features):
                if j in to_drop:
                    continue
                if abs(corr_matrix[i, j]) > threshold:
                    to_drop.add(j)
                    print(f"    去除 '{self.feature_names[j]}' (与 '{self.feature_names[i]}' 相关={corr_matrix[i,j]:.3f})")
        
        remaining = [i for i in range(self.n_features) if i not in to_drop]
        print(f"  [Corr] 保留 {len(remaining)}/{self.n_features} 个特征")
        
        self.selection_results['correlation_filter'] = {
            'dropped_features': [self.feature_names[i] for i in to_drop],
            'remaining_features': [self.feature_names[i] for i in remaining]
        }
        
        return remaining
    
    def ensemble_selection(self, k=15, corr_threshold=0.95):
        """
        综合投票特征选择 - 结合多种策略
        
        Returns:
            tuple: (selected_indices, selected_feature_names, X_selected)
        """
        print("\n" + "=" * 50)
        print("特征选择 - 综合投票")
        print("=" * 50)
        
        # 1. 相关性过滤
        non_redundant = self.correlation_filter(corr_threshold)
        
        # 2. 各策略投票
        mi_top = set(self.mutual_information_selection(k).tolist())
        f_top = set(self.f_regression_selection(k).tolist())
        tree_top = set(self.tree_importance_selection(k).tolist())
        
        # 3. 投票计数
        vote_count = defaultdict(int)
        for idx in mi_top:
            vote_count[idx] += 1
        for idx in f_top:
            vote_count[idx] += 1
        for idx in tree_top:
            vote_count[idx] += 1
        
        # 4. 按票数排序，取top-k且非冗余
        non_redundant_set = set(non_redundant)
        candidates = [(idx, votes) for idx, votes in vote_count.items() if idx in non_redundant_set]
        candidates.sort(key=lambda x: (-x[1], x[0]))
        
        selected = [idx for idx, _ in candidates[:k]]
        
        # 确保至少有一定数量的特征
        if len(selected) < min(5, self.n_features):
            for idx in non_redundant:
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= min(5, self.n_features):
                    break
        
        selected_names = [self.feature_names[i] for i in selected]
        X_selected = self.X[:, selected]
        
        print(f"\n  [综合] 最终选择 {len(selected)} 个特征:")
        for idx in selected:
            votes = vote_count.get(idx, 0)
            print(f"    {self.feature_names[idx]:35s} 票数={votes}/3")
        
        self.selection_results['ensemble'] = {
            'selected_indices': selected,
            'selected_features': selected_names,
            'vote_counts': {self.feature_names[idx]: vote_count.get(idx, 0) for idx in selected}
        }
        
        return selected, selected_names, X_selected


# ============== 8. 超参数自动调优模块 ==============

class HyperparameterTuner:
    """
    超参数自动调优 - 支持 Optuna 贝叶斯优化和网格搜索
    
    支持模型：XGBoost, LightGBM
    优化指标：MAE (最小化)
    """
    
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.best_params = {}
        self.tuning_history = []
    
    def tune_xgboost(self, n_trials=50, timeout=300):
        """Optuna调优XGBoost超参数"""
        print("\n  [Tuner] XGBoost 超参数调优...")
        
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            from xgboost import XGBRegressor
        except ImportError:
            print("    [跳过] optuna 或 xgboost 未安装，使用默认参数")
            return self._xgboost_grid_search()
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'verbosity': 0
            }
            model = XGBRegressor(**params)
            model.fit(self.X_train, self.y_train)
            pred = model.predict(self.X_val)
            return mean_absolute_error(self.y_val, pred)
        
        study = optuna.create_study(direction='minimize', study_name='xgboost_tune')
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        
        best = study.best_params
        best['random_state'] = 42
        best['verbosity'] = 0
        self.best_params['XGBoost'] = best
        
        print(f"    最佳MAE: {study.best_value:.4f}")
        print(f"    最佳参数: n_estimators={best.get('n_estimators')}, max_depth={best.get('max_depth')}, lr={best.get('learning_rate', 0):.4f}")
        
        return best
    
    def tune_lightgbm(self, n_trials=50, timeout=300):
        """Optuna调优LightGBM超参数"""
        print("\n  [Tuner] LightGBM 超参数调优...")
        
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            from lightgbm import LGBMRegressor
        except ImportError:
            print("    [跳过] optuna 或 lightgbm 未安装，使用默认参数")
            return self._lightgbm_grid_search()
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'random_state': 42,
                'verbose': -1
            }
            model = LGBMRegressor(**params)
            model.fit(self.X_train, self.y_train)
            pred = model.predict(self.X_val)
            return mean_absolute_error(self.y_val, pred)
        
        study = optuna.create_study(direction='minimize', study_name='lightgbm_tune')
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        
        best = study.best_params
        best['random_state'] = 42
        best['verbose'] = -1
        self.best_params['LightGBM'] = best
        
        print(f"    最佳MAE: {study.best_value:.4f}")
        print(f"    最佳参数: n_estimators={best.get('n_estimators')}, max_depth={best.get('max_depth')}, lr={best.get('learning_rate', 0):.4f}")
        
        return best
    
    def _xgboost_grid_search(self):
        """XGBoost简易网格搜索（Optuna不可用时的备选）"""
        from xgboost import XGBRegressor
        print("    使用网格搜索备选方案...")
        
        best_mae = float('inf')
        best_params = {}
        
        for n_est in [100, 200, 300]:
            for depth in [4, 6, 8]:
                for lr in [0.05, 0.1, 0.2]:
                    model = XGBRegressor(n_estimators=n_est, max_depth=depth,
                                         learning_rate=lr, random_state=42, verbosity=0)
                    model.fit(self.X_train, self.y_train)
                    pred = model.predict(self.X_val)
                    mae = mean_absolute_error(self.y_val, pred)
                    if mae < best_mae:
                        best_mae = mae
                        best_params = {'n_estimators': n_est, 'max_depth': depth,
                                       'learning_rate': lr, 'random_state': 42, 'verbosity': 0}
        
        self.best_params['XGBoost'] = best_params
        print(f"    最佳MAE: {best_mae:.4f}")
        return best_params
    
    def _lightgbm_grid_search(self):
        """LightGBM简易网格搜索"""
        from lightgbm import LGBMRegressor
        print("    使用网格搜索备选方案...")
        
        best_mae = float('inf')
        best_params = {}
        
        for n_est in [100, 200, 300]:
            for depth in [4, 6, 8]:
                for lr in [0.05, 0.1, 0.2]:
                    model = LGBMRegressor(n_estimators=n_est, max_depth=depth,
                                          learning_rate=lr, random_state=42, verbose=-1)
                    model.fit(self.X_train, self.y_train)
                    pred = model.predict(self.X_val)
                    mae = mean_absolute_error(self.y_val, pred)
                    if mae < best_mae:
                        best_mae = mae
                        best_params = {'n_estimators': n_est, 'max_depth': depth,
                                       'learning_rate': lr, 'random_state': 42, 'verbose': -1}
        
        self.best_params['LightGBM'] = best_params
        print(f"    最佳MAE: {best_mae:.4f}")
        return best_params
    
    def tune_all(self, n_trials=30, timeout=180):
        """
        调优所有树模型
        
        Returns:
            dict: {model_name: best_params}
        """
        print("\n" + "=" * 50)
        print("超参数自动调优")
        print("=" * 50)
        
        self.tune_xgboost(n_trials=n_trials, timeout=timeout)
        self.tune_lightgbm(n_trials=n_trials, timeout=timeout)
        
        return self.best_params


# ============== 9. 偏差-方差分析模块 ==============

class BiasVarianceAnalyzer:
    """
    偏差-方差分解分析
    
    使用 Bootstrap 方法估计模型的偏差、方差和噪声
    帮助诊断模型是欠拟合（高偏差）还是过拟合（高方差）
    """
    
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def bootstrap_bias_variance(self, model_factory, n_bootstraps=20, sample_ratio=0.8):
        """
        Bootstrap偏差-方差分解
        
        Args:
            model_factory: 返回新模型实例的可调用对象
            n_bootstraps: Bootstrap采样次数
            sample_ratio: 每次采样比例
        
        Returns:
            dict: {'bias': float, 'variance': float, 'noise': float, 'mse': float}
        """
        n_test = len(self.X_test)
        n_train = len(self.X_train)
        sample_size = int(n_train * sample_ratio)
        
        # 收集所有Bootstrap预测
        all_predictions = np.zeros((n_bootstraps, n_test))
        
        for b in range(n_bootstraps):
            # Bootstrap采样
            indices = np.random.choice(n_train, size=sample_size, replace=True)
            X_boot = self.X_train[indices]
            y_boot = self.y_train[indices]
            
            model = model_factory()
            model.fit(X_boot, y_boot)
            all_predictions[b] = model.predict(self.X_test)
        
        # 计算偏差-方差分解
        mean_pred = all_predictions.mean(axis=0)  # 平均预测
        
        # 偏差² = E[(E[h(x)] - y)²]
        bias_sq = np.mean((mean_pred - self.y_test) ** 2)
        
        # 方差 = E[E[(h(x) - E[h(x)])²]]
        variance = np.mean(np.var(all_predictions, axis=0))
        
        # 总MSE ≈ 偏差² + 方差
        total_mse = np.mean(np.mean((all_predictions - self.y_test[np.newaxis, :]) ** 2, axis=0))
        noise = max(0, total_mse - bias_sq - variance)
        
        return {
            'bias_squared': float(bias_sq),
            'variance': float(variance),
            'noise': float(noise),
            'total_mse': float(total_mse),
            'bias': float(np.sqrt(bias_sq)),
            'std_dev': float(np.sqrt(variance))
        }
    
    def analyze_all_models(self, n_bootstraps=15):
        """
        对所有可用树模型进行偏差-方差分析
        
        Returns:
            dict: {model_name: bv_result}
        """
        print("\n" + "=" * 50)
        print("偏差-方差分析")
        print("=" * 50)
        
        results = {}
        
        # XGBoost
        try:
            from xgboost import XGBRegressor
            print("\n  [B-V] XGBoost 分析中...")
            
            # 不同复杂度的模型对比
            for depth in [3, 6, 9]:
                name = f'XGBoost(depth={depth})'
                factory = lambda d=depth: XGBRegressor(n_estimators=150, max_depth=d, random_state=42, verbosity=0)
                bv = self.bootstrap_bias_variance(factory, n_bootstraps=n_bootstraps)
                results[name] = bv
                print(f"    {name:30s} Bias²={bv['bias_squared']:.4f}  Var={bv['variance']:.4f}  MSE={bv['total_mse']:.4f}")
        except ImportError:
            print("  [跳过] XGBoost 未安装")
        
        # LightGBM
        try:
            from lightgbm import LGBMRegressor
            print("\n  [B-V] LightGBM 分析中...")
            
            for depth in [3, 6, 9]:
                name = f'LightGBM(depth={depth})'
                factory = lambda d=depth: LGBMRegressor(n_estimators=150, max_depth=d, random_state=42, verbose=-1)
                bv = self.bootstrap_bias_variance(factory, n_bootstraps=n_bootstraps)
                results[name] = bv
                print(f"    {name:30s} Bias²={bv['bias_squared']:.4f}  Var={bv['variance']:.4f}  MSE={bv['total_mse']:.4f}")
        except ImportError:
            print("  [跳过] LightGBM 未安装")
        
        # 诊断建议
        if results:
            print("\n  [诊断建议]")
            for name, bv in results.items():
                ratio = bv['bias_squared'] / (bv['variance'] + 1e-8)
                if ratio > 3:
                    diagnosis = "高偏差（欠拟合）→ 增加模型复杂度、添加特征"
                elif ratio < 0.3:
                    diagnosis = "高方差（过拟合）→ 增加正则化、减少特征、增加数据"
                else:
                    diagnosis = "偏差-方差平衡较好"
                print(f"    {name:30s} Bias²/Var={ratio:.2f}  {diagnosis}")
        
        return results


# ============== 10. 学习曲线分析模块 ==============

class LearningCurveAnalyzer:
    """
    学习曲线分析 - 评估模型随训练数据量变化的表现
    
    帮助判断：
    - 是否需要更多数据
    - 模型复杂度是否合适
    - 训练和验证误差收敛趋势
    """
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def compute_learning_curve(self, model_factory, train_sizes=None, cv=5, n_repeats=3):
        """
        计算学习曲线
        
        Args:
            model_factory: 返回新模型实例的可调用对象
            train_sizes: 训练集大小比例列表
            cv: 交叉验证折数
            n_repeats: 重复次数
        
        Returns:
            dict: {'train_sizes': [...], 'train_mae': [...], 'val_mae': [...],
                   'train_mae_std': [...], 'val_mae_std': [...]}
        """
        if train_sizes is None:
            train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        n_total = len(self.X)
        results = {'train_sizes': [], 'train_mae': [], 'val_mae': [],
                   'train_mae_std': [], 'val_mae_std': []}
        
        for size_ratio in train_sizes:
            train_maes = []
            val_maes = []
            
            for repeat in range(n_repeats):
                # 随机划分
                indices = np.random.permutation(n_total)
                n_train = max(int(n_total * size_ratio * 0.8), 10)  # 80%用于训练
                n_val = max(int(n_total * 0.2), 10)  # 20%用于验证
                
                train_idx = indices[:n_train]
                val_idx = indices[n_train:n_train + n_val]
                
                if len(val_idx) == 0:
                    continue
                
                X_tr, y_tr = self.X[train_idx], self.y[train_idx]
                X_vl, y_vl = self.X[val_idx], self.y[val_idx]
                
                model = model_factory()
                model.fit(X_tr, y_tr)
                
                train_pred = model.predict(X_tr)
                val_pred = model.predict(X_vl)
                
                train_maes.append(mean_absolute_error(y_tr, train_pred))
                val_maes.append(mean_absolute_error(y_vl, val_pred))
            
            if train_maes:
                actual_size = max(int(n_total * size_ratio * 0.8), 10)
                results['train_sizes'].append(actual_size)
                results['train_mae'].append(float(np.mean(train_maes)))
                results['val_mae'].append(float(np.mean(val_maes)))
                results['train_mae_std'].append(float(np.std(train_maes)))
                results['val_mae_std'].append(float(np.std(val_maes)))
        
        return results
    
    def analyze_all_models(self, train_sizes=None):
        """
        对所有可用模型生成学习曲线
        
        Returns:
            dict: {model_name: learning_curve_result}
        """
        print("\n" + "=" * 50)
        print("学习曲线分析")
        print("=" * 50)
        
        all_curves = {}
        
        # XGBoost
        try:
            from xgboost import XGBRegressor
            print("\n  [LC] XGBoost 学习曲线...")
            factory = lambda: XGBRegressor(n_estimators=150, max_depth=6, random_state=42, verbosity=0)
            curve = self.compute_learning_curve(factory, train_sizes=train_sizes, n_repeats=3)
            all_curves['XGBoost'] = curve
            self._print_curve('XGBoost', curve)
        except ImportError:
            print("  [跳过] XGBoost 未安装")
        
        # LightGBM
        try:
            from lightgbm import LGBMRegressor
            print("\n  [LC] LightGBM 学习曲线...")
            factory = lambda: LGBMRegressor(n_estimators=150, max_depth=6, random_state=42, verbose=-1)
            curve = self.compute_learning_curve(factory, train_sizes=train_sizes, n_repeats=3)
            all_curves['LightGBM'] = curve
            self._print_curve('LightGBM', curve)
        except ImportError:
            print("  [跳过] LightGBM 未安装")
        
        # 诊断
        if all_curves:
            print("\n  [学习曲线诊断]")
            for name, curve in all_curves.items():
                if len(curve['train_mae']) >= 2 and len(curve['val_mae']) >= 2:
                    final_gap = curve['val_mae'][-1] - curve['train_mae'][-1]
                    val_trend = curve['val_mae'][-1] - curve['val_mae'][-2]
                    
                    if final_gap > curve['train_mae'][-1] * 0.5:
                        advice = "训练/验证差距大 → 可能过拟合，考虑增加数据或正则化"
                    elif val_trend < -0.01 * curve['val_mae'][-2]:
                        advice = "验证误差仍在下降 → 增加数据可能继续改善"
                    else:
                        advice = "曲线趋于收敛，模型容量匹配"
                    print(f"    {name:15s} 最终Gap={final_gap:.3f}  {advice}")
        
        return all_curves
    
    def _print_curve(self, model_name, curve):
        """打印学习曲线表格"""
        print(f"    {'数据量':>8s}  {'训练MAE':>10s}  {'验证MAE':>10s}  {'Gap':>8s}")
        print(f"    {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}")
        for i in range(len(curve['train_sizes'])):
            gap = curve['val_mae'][i] - curve['train_mae'][i]
            print(f"    {curve['train_sizes'][i]:>8d}  {curve['train_mae'][i]:>10.3f}  "
                  f"{curve['val_mae'][i]:>10.3f}  {gap:>8.3f}")


# ============== 11. SHAP可解释性分析模块 ==============

class SHAPExplainer:
    """
    SHAP可解释性分析
    
    提供模型预测的特征归因分析：
    - 全局特征重要性
    - 特征交互效应
    - 单样本解释
    """
    
    def __init__(self, model, X, feature_names, model_type='tree'):
        """
        Args:
            model: 训练好的模型
            X: 特征数据(numpy array)
            feature_names: 特征名列表
            model_type: 'tree' 或 'generic'
        """
        self.model = model
        self.X = X
        self.feature_names = feature_names
        self.model_type = model_type
        self.shap_values = None
        self.explainer = None
    
    def compute_shap_values(self, max_samples=500):
        """
        计算SHAP值
        
        Args:
            max_samples: 最大样本数（控制计算时间）
        
        Returns:
            np.ndarray: SHAP值矩阵
        """
        try:
            import shap
        except ImportError:
            print("  [SHAP] shap库未安装，使用置换重要性替代")
            print("  运行: pip install shap")
            return self._permutation_importance(max_samples)
        
        # 采样
        if len(self.X) > max_samples:
            indices = np.random.choice(len(self.X), max_samples, replace=False)
            X_sample = self.X[indices]
        else:
            X_sample = self.X
        
        print(f"  [SHAP] 计算SHAP值 (样本数={len(X_sample)})...")
        
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        else:
            background = shap.sample(X_sample, min(100, len(X_sample)))
            self.explainer = shap.KernelExplainer(self.model.predict, background)
        
        self.shap_values = self.explainer.shap_values(X_sample)
        print(f"  [SHAP] 计算完成，SHAP矩阵形状: {self.shap_values.shape}")
        
        return self.shap_values
    
    def global_importance(self, top_k=15):
        """
        全局特征重要性（基于平均|SHAP|值）
        
        Returns:
            list: [(feature_name, importance), ...] 按重要性降序
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        if self.shap_values is None:
            return []
        
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        ranked_idx = np.argsort(mean_abs_shap)[::-1]
        
        print(f"\n  [SHAP] 全局特征重要性 Top-{top_k}:")
        results = []
        for rank, idx in enumerate(ranked_idx[:top_k]):
            name = self.feature_names[idx] if idx < len(self.feature_names) else f'feature_{idx}'
            imp = float(mean_abs_shap[idx])
            results.append((name, imp))
            bar = '█' * int(imp / (mean_abs_shap[ranked_idx[0]] + 1e-8) * 30)
            print(f"    {rank+1:2d}. {name:35s} |SHAP|={imp:.4f}  {bar}")
        
        return results
    
    def feature_interaction(self, feature_idx1, feature_idx2):
        """
        分析两个特征的交互效应
        
        Returns:
            dict: 交互统计
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        if self.shap_values is None:
            return {}
        
        shap1 = self.shap_values[:, feature_idx1]
        shap2 = self.shap_values[:, feature_idx2]
        correlation = np.corrcoef(shap1, shap2)[0, 1]
        
        name1 = self.feature_names[feature_idx1]
        name2 = self.feature_names[feature_idx2]
        
        return {
            'feature_1': name1,
            'feature_2': name2,
            'shap_correlation': float(correlation),
            'interaction_strength': float(abs(correlation))
        }
    
    def explain_single(self, sample_idx=0):
        """
        解释单个样本的预测
        
        Returns:
            dict: {feature_name: shap_value}
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        if self.shap_values is None:
            return {}
        
        idx = min(sample_idx, len(self.shap_values) - 1)
        sample_shap = self.shap_values[idx]
        
        explanation = {}
        sorted_idx = np.argsort(np.abs(sample_shap))[::-1]
        
        print(f"\n  [SHAP] 样本 #{idx} 预测解释:")
        for i in sorted_idx[:10]:
            name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
            val = float(sample_shap[i])
            direction = "↑" if val > 0 else "↓"
            explanation[name] = val
            print(f"    {name:35s} SHAP={val:+.4f} {direction}")
        
        return explanation
    
    def _permutation_importance(self, max_samples=500):
        """置换重要性作为SHAP的备选方案"""
        print("  [PI] 使用置换重要性替代SHAP...")
        
        if len(self.X) > max_samples:
            indices = np.random.choice(len(self.X), max_samples, replace=False)
            X_sample = self.X[indices]
        else:
            X_sample = self.X
        
        baseline_pred = self.model.predict(X_sample)
        importances = np.zeros(X_sample.shape[1])
        
        for i in range(X_sample.shape[1]):
            X_permuted = X_sample.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            perm_pred = self.model.predict(X_permuted)
            importances[i] = np.mean(np.abs(perm_pred - baseline_pred))
        
        # 构造伪SHAP值矩阵
        self.shap_values = np.tile(importances, (len(X_sample), 1))
        print(f"  [PI] 置换重要性计算完成")
        return self.shap_values
    
    def full_analysis(self, top_k=15, sample_idx=0):
        """
        完整SHAP分析流程
        
        Returns:
            dict: 分析结果
        """
        print("\n" + "=" * 50)
        print("SHAP 可解释性分析")
        print("=" * 50)
        
        self.compute_shap_values()
        importance = self.global_importance(top_k=top_k)
        single_explanation = self.explain_single(sample_idx)
        
        # 分析前两个最重要特征的交互
        interaction = {}
        if importance and len(importance) >= 2:
            idx1 = self.feature_names.index(importance[0][0]) if importance[0][0] in self.feature_names else 0
            idx2 = self.feature_names.index(importance[1][0]) if importance[1][0] in self.feature_names else 1
            interaction = self.feature_interaction(idx1, idx2)
            if interaction:
                print(f"\n  [SHAP] 特征交互: {interaction['feature_1']} × {interaction['feature_2']}")
                print(f"    SHAP相关系数: {interaction['shap_correlation']:.4f}")
        
        return {
            'global_importance': importance,
            'single_explanation': single_explanation,
            'top_interaction': interaction
        }


# ============== 12. 性能监控模块 ==============

class PerformanceMonitor:
    """
    性能监控工具 - 跟踪模型训练和推理的全流程性能指标
    
    监控内容：
    - 各阶段耗时
    - 内存使用
    - 模型指标（MAE/RMSE/R²）
    - GPU显存（如有）
    """
    
    def __init__(self, log_dir='./performance_logs'):
        self.log_dir = log_dir
        self.timers = {}  # {name: start_time}
        self.durations = {}  # {name: duration}
        self.metrics = defaultdict(dict)  # {model_name: {metric_name: value}}
        self.memory_snapshots = []
        self.events = []  # 事件日志
    
    def start_timer(self, name):
        """开始计时"""
        self.timers[name] = time.time()
        self.events.append({'event': 'timer_start', 'name': name, 'time': time.time()})
    
    def stop_timer(self, name):
        """停止计时，返回耗时秒数"""
        if name not in self.timers:
            return 0
        duration = time.time() - self.timers[name]
        self.durations[name] = duration
        self.events.append({'event': 'timer_stop', 'name': name, 'duration': duration})
        return duration
    
    def record_metric(self, model_name, metric_name, value):
        """记录模型指标"""
        self.metrics[model_name][metric_name] = float(value)
    
    def record_memory(self, label=''):
        """记录当前内存使用"""
        import sys
        snapshot = {
            'label': label,
            'time': time.time()
        }
        
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            snapshot['rss_mb'] = mem_info.rss / (1024 * 1024)
            snapshot['vms_mb'] = mem_info.vms / (1024 * 1024)
        except ImportError:
            snapshot['rss_mb'] = 0
            snapshot['note'] = 'psutil not installed'
        
        if torch.cuda.is_available():
            snapshot['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            snapshot['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def comprehensive_eval(self, model_name, y_true, y_pred):
        """
        综合评估模型性能
        
        Returns:
            dict: 全面的评估指标
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 清理NaN
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        
        if len(y_true) == 0:
            return {}
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # 分位数误差
        abs_errors = np.abs(y_true - y_pred)
        p50_error = np.percentile(abs_errors, 50)
        p90_error = np.percentile(abs_errors, 90)
        p95_error = np.percentile(abs_errors, 95)
        
        # 准确率 (误差在一定范围内的比例)
        acc_5min = float(np.mean(abs_errors <= 5) * 100)
        acc_10min = float(np.mean(abs_errors <= 10) * 100)
        acc_15min = float(np.mean(abs_errors <= 15) * 100)
        
        result = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R²': float(r2),
            'MAPE(%)': float(mape),
            'P50_Error': float(p50_error),
            'P90_Error': float(p90_error),
            'P95_Error': float(p95_error),
            'Acc@5min(%)': acc_5min,
            'Acc@10min(%)': acc_10min,
            'Acc@15min(%)': acc_15min
        }
        
        for k, v in result.items():
            self.record_metric(model_name, k, v)
        
        return result
    
    def print_summary(self):
        """打印性能监控汇总报告"""
        print("\n" + "=" * 65)
        print("性能监控汇总报告")
        print("=" * 65)
        
        # 耗时统计
        if self.durations:
            print("\n  ┌──────────────────────────────────────────────────────┐")
            print("  │                   各阶段耗时统计                     │")
            print("  ├──────────────────────────────────────────────────────┤")
            total_time = sum(self.durations.values())
            for name, duration in sorted(self.durations.items(), key=lambda x: -x[1]):
                pct = duration / total_time * 100 if total_time > 0 else 0
                bar = '█' * int(pct / 3)
                print(f"  │  {name:28s} {duration:8.2f}s ({pct:5.1f}%) {bar:<15} │")
            print(f"  │  {'─'*50}  │")
            print(f"  │  {'总计':28s} {total_time:8.2f}s                  │")
            print("  └──────────────────────────────────────────────────────┘")
        
        # 模型指标对比
        if self.metrics:
            print("\n  ┌──────────────────────────────────────────────────────────────────────────┐")
            print("  │                        模型性能指标对比                                  │")
            print("  ├──────────────────────────────────────────────────────────────────────────┤")
            
            # 表头
            header_metrics = ['MAE', 'RMSE', 'R²', 'MAPE(%)', 'Acc@5min(%)', 'Acc@10min(%)']
            print(f"  │  {'模型':15s}", end="")
            for m in header_metrics:
                print(f" {m:>12s}", end="")
            print("  │")
            print(f"  │  {'─'*15}", end="")
            for _ in header_metrics:
                print(f" {'─'*12}", end="")
            print("  │")
            
            for model_name, metrics in self.metrics.items():
                print(f"  │  {model_name:15s}", end="")
                for m in header_metrics:
                    val = metrics.get(m, float('nan'))
                    print(f" {val:>12.3f}", end="")
                print("  │")
            
            print("  └──────────────────────────────────────────────────────────────────────────┘")
        
        # 内存使用
        if self.memory_snapshots:
            print("\n  ┌──────────────────────────────────────────────────────┐")
            print("  │                   内存使用记录                       │")
            print("  ├──────────────────────────────────────────────────────┤")
            for snap in self.memory_snapshots:
                rss = snap.get('rss_mb', 0)
                label = snap.get('label', '')
                gpu = snap.get('gpu_allocated_mb', None)
                line = f"  │  {label:30s} RAM={rss:8.1f}MB"
                if gpu is not None:
                    line += f"  GPU={gpu:.1f}MB"
                line += f"{'':>{52 - len(line)}}  │" if len(line) < 52 else "  │"
                print(line)
            print("  └──────────────────────────────────────────────────────┘")
    
    def save_report(self, filepath=None):
        """保存性能报告为JSON"""
        if filepath is None:
            os.makedirs(self.log_dir, exist_ok=True)
            filepath = os.path.join(self.log_dir, f'perf_report_{int(time.time())}.json')
        
        report = {
            'durations': self.durations,
            'metrics': dict(self.metrics),
            'memory_snapshots': self.memory_snapshots
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n  [Monitor] 性能报告已保存: {filepath}")
        return filepath


# ============== 统一入口 ==============

def main():
    """统一主函数 - 依次运行Pickup分析、Roads分析、综合分析和ETA/VRP"""
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("\n" + "=" * 60)
    print("Supply Chain AI System Demo")
    print("=" * 60)

    # 0. 先加载所有数据
    print("\n>>> Loading all CSV files...")
    all_data = load_all_csv_files()

    # 1. 揽件数据分析
    print("\n>>> Starting Pickup Analysis...")
    try:
        pickup_results = pickup_analysis()
        print("\n[Pickup] Analysis completed!")
    except Exception as e:
        print(f"\n[Pickup] Error: {e}")
        traceback.print_exc()
        pickup_results = None

    # 2. 道路网络分析
    print("\n>>> Starting Roads Analysis...")
    try:
        roads_results = roads_analysis()
        print("\n[Roads] Analysis completed!")
    except Exception as e:
        print(f"\n[Roads] Error: {e}")
        traceback.print_exc()
        roads_results = None

    # 3. 轨迹数据分析
    print("\n>>> Starting Trajectory Analysis...")
    try:
        trajectory_results = trajectory_analysis(max_rows=500000)
        print("\n[Trajectory] Analysis completed!")
    except Exception as e:
        print(f"\n[Trajectory] Error: {e}")
        traceback.print_exc()
        trajectory_results = None

    # 4. 综合分析
    print("\n>>> Starting Comprehensive Analysis...")
    try:
        comp_results = comprehensive_analysis()
        print("\n[Comprehensive] Analysis completed!")
    except Exception as e:
        print(f"\n[Comprehensive] Error: {e}")
        traceback.print_exc()
        comp_results = None

    # 5. 运行ETA预测
    print("\n>>> Starting ETA Prediction...")
    try:
        eta_results = eta_main()
        print("\n[ETA] Prediction completed!")
    except Exception as e:
        print(f"\n[ETA] Error: {e}")
        traceback.print_exc()
        eta_results = None

    # 6. 运行VRP优化（可选）
    print("\n>>> Starting VRP Optimization...")
    vrp_results = vrp_main()

    print("\n" + "=" * 60)
    print("All tasks completed")
    print("=" * 60)

    return pickup_results, roads_results, trajectory_results, comp_results, eta_results, vrp_results


if __name__ == '__main__':
    results = main()
