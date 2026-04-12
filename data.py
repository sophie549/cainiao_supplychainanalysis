import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import warnings
import random
warnings.filterwarnings('ignore')

# ============== 0. 道路距离加权系数 ==============
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
    """LaDe数据集处理器 - 适配delivery_five_cities.csv"""
    
    def __init__(self, data_path, max_samples=50000):
        self.df = pd.read_csv(data_path)
        print(f"[Data Check] sign_time缺失率: {self.df['sign_time'].isna().mean()*100:.2f}%")
        print(f"[Data Check] sign_lng缺失率: {self.df['sign_lng'].isna().mean()*100:.2f}%")
        print(f"[Data Check] sign_lat缺失率: {self.df['sign_lat'].isna().mean()*100:.2f}%")
        if max_samples and len(self.df) > max_samples:
            self.df = self.df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            print(f"Data sampled to {max_samples} records")
        self.scaler = StandardScaler()
        self.weather_encoder = LabelEncoder()
        self.traffic_encoder = LabelEncoder()
        self.poi_type_encoder = LabelEncoder()
        
        # 坐标重命名（避免与真实经纬度混淆）
        self.df.rename(columns={
            'poi_lat': 'dest_lat_3857',
            'poi_lng': 'dest_lng_3857',
            'receipt_lat': 'pickup_lat_3857',
            'receipt_lng': 'pickup_lng_3857'
        }, inplace=True)
        
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
        
        # 历史统计特征（按司机）
        driver_stats = df.groupby('driver_id')['eta_minutes'].agg(['mean', 'std']).reset_index()
        driver_stats.columns = ['driver_id', 'driver_avg_eta', 'driver_std_eta']
        df = df.merge(driver_stats, on='driver_id', how='left')
        df['driver_avg_eta'] = df['driver_avg_eta'].fillna(df['eta_minutes'].mean())
        df['driver_std_eta'] = df['driver_std_eta'].fillna(0)
        
        # 模拟天气和交通特征（原数据不存在这些列）
        weather_conditions = ['sunny', 'cloudy', 'rainy', 'foggy']
        traffic_conditions = ['low', 'medium', 'high']
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
        
        # 路线复杂度（轨迹点数/直线距离）
        if 'courier_total_points' in df.columns and 'distance_km' in df.columns:
            df['route_complexity'] = df['courier_total_points'] / (df['distance_km'] + 1)
        else:
            df['route_complexity'] = 0
        
        # 历史拥堵程度（按城市+时段统计）
        if 'from_city_name' in df.columns and 'hour' in df.columns:
            congestion_stats = df.groupby(['from_city_name', 'hour']).agg({
                'pickup_eta_minutes': 'mean' if 'pickup_eta_minutes' in df.columns else 'count'
            }).reset_index()
            congestion_stats.columns = ['from_city_name', 'hour', 'historical_congestion']
            df = df.merge(congestion_stats, on=['from_city_name', 'hour'], how='left')
            df['historical_congestion'] = df['historical_congestion'].fillna(df['historical_congestion'].mean())
        else:
            df['historical_congestion'] = 0
        
        self.df = df
        return df
    
    def prepare_features(self):
        """准备特征矩阵"""
        feature_cols = [
            'distance_km', 'hour', 'weekday', 'is_weekend', 'time_period', 'rush_hour',
            'weather_encoded', 'traffic_encoded', 'dest_lat_3857', 'dest_lng_3857',
            'driver_avg_eta', 'driver_std_eta',
            'courier_total_points', 'courier_active_hours', 'courier_lat_range', 'courier_lng_range',
            'courier_stationary_points_pct', 'courier_moving_distance_km', 'courier_actual_active_hours',
            'avg_workload', 'max_workload', 'std_workload',
            'route_complexity', 'historical_congestion', 'poi_type_encoded'
        ]
        
        X = self.df[feature_cols].values
        y = self.df['eta_minutes'].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, feature_cols


# ============== 2. LSTM模型 ==============

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


# ============== 4. 训练与评估 ==============

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


def eta_main():
    """ETA预测主函数"""
    import time
    
    # 配置
    DATA_PATH = './delivery_five_cities.csv'
    SEQ_LEN = 10
    BATCH_SIZE = 256
    EPOCHS = 10  # 减少轮数以加速演示
    LEARNING_RATE = 0.001
    MAX_SAMPLES = 50000  # 限制样本数
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # 1. 数据加载与预处理
    print("=" * 50)
    print("Step 1: Data Loading and Preprocessing")
    print("=" * 50)
    
    processor = LaDeDataProcessor(DATA_PATH, max_samples=MAX_SAMPLES)
    df = processor.engineer_features()
    print(f"Dataset size: {len(df)} records")
    
    X, y, feature_cols = processor.prepare_features()
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Features: {feature_cols}")
    
    # 分层抽样确保各城市样本均衡
    if 'from_city_name' in processor.df.columns:
        city_labels = processor.df['from_city_name'].values
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, temp_idx = next(sss.split(X, city_labels))
        X_train, X_temp = X[train_idx], X[temp_idx]
        y_train, y_temp = y[train_idx], y[temp_idx]
        
        city_labels_temp = city_labels[temp_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        val_idx, test_idx = next(sss2.split(X_temp, city_labels_temp))
        X_val, X_test = X_temp[val_idx], X_temp[test_idx]
        y_val, y_test = y_temp[val_idx], y_temp[test_idx]
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 创建DataLoader
    train_dataset = ETADataset(X_train, y_train, SEQ_LEN)
    val_dataset = ETADataset(X_val, y_val, SEQ_LEN)
    test_dataset = ETADataset(X_test, y_test, SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 2. 模型训练
    print("\n" + "=" * 50)
    print("Step 2: Model Training")
    print("=" * 50)
    
    input_dim = X_train.shape[1]
    models = {
        'LSTM': LSTMETA(input_dim, hidden_dim=128, num_layers=2),
        'Transformer': TransformerETA(input_dim, d_model=128, nhead=8, num_layers=3)
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
        patience_counter = 0
        early_stop_patience = 10
        
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_mae, val_rmse, _, _ = evaluate(model, val_loader, criterion, DEVICE)
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
        
        train_time = time.time() - start_time
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        
        # 最终测试集评估
        _, test_mae, test_rmse, predictions, actuals = evaluate(model, test_loader, criterion, DEVICE)
        
        results[model_name] = {
            'mae': test_mae,
            'rmse': test_rmse,
            'train_time': train_time,
            'predictions': predictions,
            'actuals': actuals
        }
        
        print(f"\n{model_name} Test Results:")
        print(f"  MAE: {test_mae:.2f} min")
        print(f"  RMSE: {test_rmse:.2f} min")
        print(f"  Train Time: {train_time:.2f}s")
    
    # 2.5 训练树模型 baseline
    print("\n" + "=" * 50)
    print("Step 2.5: Tree Model Baselines")
    print("=" * 50)
    
    tree_results = train_tree_baselines(X_train, y_train, X_test, y_test)
    results.update(tree_results)
    
    # 3. 模型对比
    print("\n" + "=" * 50)
    print("Step 3: Model Comparison")
    print("=" * 50)
    
    print("\n| Model        | MAE (min) | RMSE (min) | Train Time (s) |")
    print("|--------------|-----------|------------|----------------|")
    for name, result in results.items():
        train_time = result.get('train_time', 0)
        print(f"| {name:<12} | {result['mae']:<9.2f} | {result['rmse']:<10.2f} | {train_time:<14.2f} |")
    
    # 找到最佳模型
    best_model_name = min(results, key=lambda x: results[x]['mae'])
    print(f"\nBest Model: {best_model_name} (MAE: {results[best_model_name]['mae']:.2f}min)")
    
    # 4. 保存模型
    print("\n" + "=" * 50)
    print("Step 4: Save Model")
    print("=" * 50)
    
    import joblib
    if best_model_name in ['XGBoost', 'LightGBM']:
        joblib.dump(results[best_model_name]['model'], f'./eta_model_{best_model_name.lower()}.joblib')
        print(f"Model saved: eta_model_{best_model_name.lower()}.joblib")
    else:
        torch.save({
            'model_state': models[best_model_name].state_dict(),
            'scaler': processor.scaler,
            'weather_encoder': processor.weather_encoder,
            'traffic_encoder': processor.traffic_encoder,
            'feature_cols': feature_cols
        }, f'./eta_model_{best_model_name.lower()}.pth')
        print(f"Model saved: eta_model_{best_model_name.lower()}.pth")
    
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

def vrp_main():
    """VRP路径优化主函数 - 如果pyvrp可用则运行"""
    try:
        from pyvrp import Model
        from pyvrp.stop import MaxIterations, MaxRuntime, MultipleCriteria
        
        print("\n" + "=" * 60)
        print("Supply Chain AI - PyVRP Routing Optimization")
        print("=" * 60)
        
        # 使用 pyvrp 0.13.x API 创建 CVRP 问题
        # 创建 Model 实例
        model = Model()
        
        # 添加配送中心 (depot)
        depot = model.add_depot(x=30, y=40, name="Warehouse")
        
        # 添加客户点 (clients)
        clients_data = [
            (20, 25, 15, "C1"),
            (35, 20, 25, "C2"),
            (45, 35, 10, "C3"),
            (15, 45, 30, "C4"),
            (50, 30, 20, "C5"),
        ]
        
        clients = []
        client_names = {}  # 用于存储 client ID (1-based) 到名称的映射
        for idx, (x, y, demand, name) in enumerate(clients_data, start=1):
            client = model.add_client(x=x, y=y, delivery=demand, name=name)
            clients.append(client)
            client_names[idx] = name  # client ID 是 1-based
        
        # 添加车辆类型
        model.add_vehicle_type(
            num_available=3,
            capacity=100,
            name="StandardTruck",
            start_depot=depot
        )
        
        # 添加边（距离矩阵）- 使用欧几里得距离
        # 获取所有位置（depot + clients）
        locations = [depot] + clients
        for i, loc1 in enumerate(locations):
            for j, loc2 in enumerate(locations):
                if i != j:
                    distance = np.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)
                    # 添加边，设置距离和持续时间
                    model.add_edge(loc1, loc2, distance=distance, duration=distance)
        
        # 求解
        stop = MultipleCriteria([MaxIterations(100), MaxRuntime(10)])
        result = model.solve(stop=stop)
        
        # 获取最佳解决方案
        solution = result.best
        
        print("\nVRP Results:")
        print(f"Cost: {result.cost():.2f}")
        print(f"Distance: {solution.distance():.0f}")
        print(f"Vehicles: {solution.num_routes()}")
        
        # 打印路径详情
        print("\nRoutes:")
        for idx, route in enumerate(solution.routes()):
            # route.visits() 返回 client ID 列表（整数，1-based）
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


# ============== 统一入口 ==============

def main():
    """统一主函数 - 依次运行Pickup分析、Roads分析、综合分析和ETA/VRP"""
    # 设置UTF-8编码（Windows兼容）
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    print("\n" + "=" * 60)
    print("Supply Chain AI System Demo")
    print("=" * 60)
    
    # 1. 揽件数据分析
    print("\n>>> Starting Pickup Analysis...")
    try:
        pickup_results = pickup_analysis()
        print("\n[Pickup] Analysis completed!")
    except Exception as e:
        print(f"\n[Pickup] Error: {e}")
        import traceback
        traceback.print_exc()
        pickup_results = None
    
    # 2. 道路网络分析
    print("\n>>> Starting Roads Analysis...")
    try:
        roads_results = roads_analysis()
        print("\n[Roads] Analysis completed!")
    except Exception as e:
        print(f"\n[Roads] Error: {e}")
        import traceback
        traceback.print_exc()
        roads_results = None
    
    # 3. 轨迹数据分析
    print("\n>>> Starting Trajectory Analysis...")
    try:
        trajectory_results = trajectory_analysis(max_rows=500000)
        print("\n[Trajectory] Analysis completed!")
    except Exception as e:
        print(f"\n[Trajectory] Error: {e}")
        import traceback
        traceback.print_exc()
        trajectory_results = None
    
    # 4. 综合分析
    print("\n>>> Starting Comprehensive Analysis...")
    try:
        comp_results = comprehensive_analysis()
        print("\n[Comprehensive] Analysis completed!")
    except Exception as e:
        print(f"\n[Comprehensive] Error: {e}")
        import traceback
        traceback.print_exc()
        comp_results = None
    
    # 5. 运行ETA预测
    print("\n>>> Starting ETA Prediction...")
    try:
        eta_results = eta_main()
        print("\n[ETA] Prediction completed!")
    except Exception as e:
        print(f"\n[ETA] Error: {e}")
        import traceback
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
