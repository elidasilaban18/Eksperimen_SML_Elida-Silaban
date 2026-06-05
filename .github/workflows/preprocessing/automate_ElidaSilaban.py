import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import os, sys

def load_data(filepath: str) -> pd.DataFrame:
    #Load dataset dari file CSV
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'File tidak ditemukan: {filepath}')
    df = pd.read_csv(filepath)
    print(f'[LOAD] Data berhasil dimuat. Shape: {df.shape}')
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    #Hapus baris duplikat
    before = len(df)
    df = df.drop_duplicates()
    print(f'[DEDUP] Hapus {before - len(df)} duplikat. Sisa: {len(df)} baris')
    return df

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    #Imputasi missing values
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    # Exclude target variable and PassengerId from imputation and scaling
    if 'Survived' in numeric_cols:
        numeric_cols.remove('Survived')
    if 'PassengerId' in numeric_cols:
        numeric_cols.remove('PassengerId')

    df[numeric_cols] = SimpleImputer(strategy='median').fit_transform(df[numeric_cols])
    if cat_cols:
        df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])
    print(f'[IMPUTE] Missing setelah imputasi: {df.isnull().sum().sum()}')
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    #Label encoding untuk kolom kategorikal
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    print(f'[ENCODE] Encoded {len(cat_cols)} kolom kategorik')
    return df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    #StandardScaler pada fitur numerik
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Exclude target variable and PassengerId from imputation and scaling
    if 'Survived' in numeric_cols:
        numeric_cols.remove('Survived')
    if 'PassengerId' in numeric_cols:
        numeric_cols.remove('PassengerId')
    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
    print(f'[SCALE] Scaling selesai pada {len(numeric_cols)} kolom)')
    return df

def run_preprocessing(input_path: str, output_path: str):
    """Fungsi utama: jalankan seluruh pipeline preprocessing."""
    print('='*50)
    print('MULAI PREPROCESSING OTOMATIS')
    print('='*50)

    df = load_data(input_path)
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = encode_categorical(df)
    df = scale_features(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'\n[DONE] Data bersih disimpan ke: {output_path}')
    print(f'[DONE] Shape akhir: {df.shape}')
    return df

if __name__ == '__main__':
    INPUT = "/content/train_raw.csv"
    OUTPUT = "preprocessing/train_preprocessing.csv"
    run_preprocessing(INPUT, OUTPUT)
