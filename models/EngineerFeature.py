import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # mapping’ler burada tutulacak
        self.artist_song_counts_ = None
        self.artist_avg_popularity_ = None
        self.percentile_90_ = None
        self.artist_high_count_ = None
        self.artist_high_ratio_ = None
        self.qcut_bins_ = None        
        self.subgenre_avg_popularity_ = None    
    def fit(self, X, y=None):
        """Train set üzerinde istatistikleri öğrenir."""
        
        X = X.copy()
        
        if y is not None:
            X['popularity'] = y
        
        if 'popularity' not in X.columns:
            raise ValueError("popularity column must be provided either in X or as y parameter")
        
        # 1. Artist şarkı sayısı
        self.artist_song_counts_ = X['artist'].value_counts().to_dict()

        # 2. Artist ortalama popülerlik
        self.artist_avg_popularity_ = X.groupby('artist')['popularity'].mean().to_dict()

        # 3. 90 percentile (train’e özel)
        self.percentile_85_ = X['popularity'].quantile(0.85)

        # 4. Artist yüksek pop sayısı
        self.artist_high_count_ = (
            X[X['popularity'] > self.percentile_85_]
            .groupby('artist')
            .size()
            .to_dict()
        )

        # 5. High pop ratio
        self.artist_high_ratio_ = {
            artist: self.artist_high_count_.get(artist, 0) / self.artist_song_counts_[artist]
            for artist in self.artist_song_counts_
        }

        # 6. Artist song count binning için cut sınırları
        # Gerçek değerler üzerinde quantile'leri hesapla
        artist_song_count_series = X['artist'].map(self.artist_song_counts_)
        quantiles = artist_song_count_series.quantile([0, 0.25, 0.5, 0.75, 1.0]).unique()
        # Unique değerleri al (duplicates varsa)
        self.qcut_bins_ = sorted(quantiles)
        
        # 7. Subgenre ortalama popülerlik
        subgenre_cols = [col for col in X.columns if col.startswith('subgenre_')]
        self.subgenre_avg_popularity_ = {}
        for subgenre_col in subgenre_cols:
            # Her subgenre için, o subgenre'a sahip şarkıların ortalama popülerliğini hesapla
            mask = X[subgenre_col] == 1
            if mask.sum() > 0:
                self.subgenre_avg_popularity_[subgenre_col] = X.loc[mask, 'popularity'].mean()
            else:
                self.subgenre_avg_popularity_[subgenre_col] = X['popularity'].mean()
        
        return self
    
    def transform(self, X):
        """Train veya test verisine mapping’leri uygular."""
        
        X = X.copy()

        # 1. Artist şarkı sayısı
        X['artist_song_count'] = X['artist'].map(self.artist_song_counts_).fillna(0)

        # 2. Artist ortalama popülerlik
        global_avg = pd.Series(self.artist_avg_popularity_).mean()
        X['artist_avg_popularity'] = X['artist'].map(self.artist_avg_popularity_).fillna(global_avg)

        # 3. High popular ratio
        X['artist_high_pop_ratio'] = X['artist'].map(self.artist_high_ratio_).fillna(0)

        # 4. qcut bin uygulama (train’de öğrenilen sınırlar)
        # Bin sayısını dinamik olarak hesapla
        n_bins = len(self.qcut_bins_) - 1
        if n_bins > 1:
            # Label sayısı bin sayısına eşit olmalı
            labels = [i / (n_bins - 1) for i in range(n_bins)]
            X["artist_song_count_bin"] = pd.cut(
                X["artist_song_count"],
                bins=self.qcut_bins_,
                include_lowest=True,
                labels=labels
            )
            # Kategorik değeri float'a çevir ve NaN değerleri 0 ile doldur
            X["artist_song_count_bin"] = pd.to_numeric(X["artist_song_count_bin"], errors='coerce').fillna(0.0)
        else:
            # Bin yoksa veya sadece 1 bin varsa, tüm değerleri 0 yap
            X["artist_song_count_bin"] = 0.0

        # 5. Subgenre ortalama popülerlik hesapla
        subgenre_cols = [col for col in X.columns if col.startswith('subgenre_')]
        if subgenre_cols and self.subgenre_avg_popularity_:
            # Her satır için, aktif subgenre'ların ortalama popülerliğini hesapla
            def calculate_avg_subgenre_popularity(row):
                active_subgenres = [col for col in subgenre_cols if row[col] == 1]
                if active_subgenres:
                    popularities = [self.subgenre_avg_popularity_.get(col, 0) for col in active_subgenres]
                    return np.mean(popularities)
                else:
                    # Hiç subgenre yoksa, genel ortalamayı kullan
                    return np.mean(list(self.subgenre_avg_popularity_.values()))
            
            X['avg_subgenre_popularity'] = X.apply(calculate_avg_subgenre_popularity, axis=1)
            
            # Subgenre kolonlarını kaldır
            X.drop(columns=subgenre_cols, errors='ignore', inplace=True)
        
        # 6. Artist kolonunu kaldır (modelde kullanmıyoruz)
        X.drop(columns=['artist', 'artist_song_count'], errors='ignore', inplace=True)

        return X
