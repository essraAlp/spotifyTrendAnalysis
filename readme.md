# 🎵 Spotify Track Popularity Prediction

![@neonbrand via Unsplash - person holding space gray iPhone 6](https://images.unsplash.com/photo-1495434942214-9b525bba74e9?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1350&q=80)

A comprehensive machine learning project that predicts Spotify track popularity (0-100) using audio features, artist statistics, and temporal information. This project explores multiple regression algorithms and advanced feature engineering techniques to understand what makes a song popular.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Feature Engineering](#-feature-engineering)
- [Models Implemented](#-models-implemented)
- [Results](#-results)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)

## 🎯 Project Overview

This project aims to predict the popularity of Spotify tracks using a combination of:
- **Audio Features**: Danceability, energy, loudness, acousticness, instrumentalness, etc.
- **Artist Statistics**: Song count, average popularity, high-popularity ratio
- **Temporal Features**: Album age
- **Genre Information**: One-hot encoded subgenres
- **Playlist Metrics**: Playlist count per track

The project includes extensive exploratory data analysis (EDA), custom feature engineering pipeline, and implementation of 7 different machine learning algorithms with hyperparameter tuning.

## 📊 Dataset

The dataset comes from Spotify via the [`spotifyr` package](https://www.rcharlie.com/spotifyr/), originally featured in [TidyTuesday 2020-01-21](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21).

### Dataset Statistics
- **Tracks**: ~32,000 songs
- **Features**: 23 original features + engineered features
- **Genres**: 6 main categories (EDM, Latin, Pop, R&B, Rap, Rock)
- **Target Variable**: `track_popularity` (0-100)

### Key Features

| Feature | Type | Description |
|---------|------|-------------|
| `track_popularity` | Target | Song popularity score (0-100) |
| `danceability` | Float | How suitable a track is for dancing (0.0-1.0) |
| `energy` | Float | Intensity and activity measure (0.0-1.0) |
| `loudness` | Float | Overall loudness in decibels (dB) |
| `speechiness` | Float | Presence of spoken words (0.0-1.0) |
| `acousticness` | Float | Confidence measure if track is acoustic (0.0-1.0) |
| `instrumentalness` | Float | Predicts if track contains no vocals (0.0-1.0) |
| `liveness` | Float | Presence of audience in recording (0.0-1.0) |
| `valence` | Float | Musical positiveness (0.0-1.0) |
| `tempo` | Float | Estimated tempo in BPM |
| `duration_ms` | Integer | Song duration in milliseconds |
| `key` | Integer | Estimated key (0-11 using Pitch Class notation) |
| `mode` | Binary | Major (1) or minor (0) |

For complete data dictionary, see the [bottom of this document](#-complete-data-dictionary).

## 📁 Project Structure

```
spotifyTrendAnalysis/
│
├── spotify_songs.csv           # Original dataset
├── final_data.csv             # Processed dataset with engineered features
├── requirements.txt           # Python dependencies
├── readme.md                 # Project documentation
│
├── eda_final.ipynb           # Exploratory Data Analysis
├── feature_eng.ipynb         # Feature engineering notebook
│
└── models/
    ├── EngineerFeature.py    # Custom feature engineering transformer
    │
    ├── lineer.ipynb          # Linear Regression
    ├── lasso_regression.ipynb # Lasso Regression
    ├── ridge_ml.ipynb        # Ridge Regression
    ├── random_forest.ipynb   # Random Forest Regressor
    ├── xgboost.ipynb         # XGBoost Regressor
    ├── lightGBM.ipynb        # LightGBM Regressor
    ├── nn.ipynb              # Neural Network (TensorFlow + Keras Tuner)
    │
    ├── best_nn_spotify.h5    # Best neural network model (H5 format)
    ├── best_nn_spotify.keras # Best neural network model (Keras format)
    │
    └── nn_tuner_logs/        # Keras Tuner hyperparameter search logs
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository** (or download the project)
```bash
cd spotifyTrendAnalysis
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dependencies
```
seaborn              # Data visualization
matplotlib           # Plotting library
typing_extensions    # Type hints
scikit-learn         # Machine learning algorithms
optuna               # Hyperparameter optimization
scikit-optimize      # Bayesian optimization
xgboost             # Gradient boosting
lightgbm            # Light gradient boosting
tensorflow          # Deep learning
keras-tuner         # Neural network tuning
pre-commit          # Git hooks
nbstripout          # Clean notebook outputs
```

## 🚀 Usage

### 1. Exploratory Data Analysis

Open and run `eda_final.ipynb` to explore:
- Data distribution and statistics
- Missing value analysis
- Correlation analysis
- Artist popularity trends
- Genre distributions
- Feature relationships with popularity

```bash
jupyter notebook eda_final.ipynb
```

### 2. Feature Engineering

Run `feature_eng.ipynb` to create the engineered dataset:
- Processes playlist information
- Creates artist-level features
- Encodes subgenres
- Generates temporal features
- Outputs `final_data.csv`

```bash
jupyter notebook feature_eng.ipynb
```

### 3. Train Models

Navigate to the `models/` directory and run any model notebook:

```bash
cd models
jupyter notebook nn.ipynb  # Neural Network example
```

Each model notebook includes:
- Data loading and preprocessing
- Feature transformation using `FeatureEngineer` class
- Train/validation/test split
- Model training with hyperparameter tuning
- Performance evaluation
- Model saving

### 4. Using the Feature Engineer

The `FeatureEngineer` class can be used as a scikit-learn transformer:

```python
from EngineerFeature import FeatureEngineer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline
pipeline = Pipeline([
    ('feature_eng', FeatureEngineer(isMLP=False)),
    ('scaler', StandardScaler())
])

# Fit on training data
X_train_transformed = pipeline.fit_transform(X_train, y_train)

# Transform test data
X_test_transformed = pipeline.transform(X_test)
```

## 🔨 Feature Engineering

The custom `FeatureEngineer` class (`models/EngineerFeature.py`) implements sophisticated feature transformations:

### Artist-Level Features
1. **`artist_song_count`**: Number of songs by each artist in the dataset
2. **`artist_avg_popularity`**: Mean popularity of all artist's songs
3. **`artist_high_pop_ratio`**: Ratio of songs with popularity > 85th percentile
4. **`artist_song_count_bin`**: Binned artist productivity categories

### Playlist Features
- **`playlist_count`**: Number of playlists containing each track
- **One-hot encoded subgenres**: Binary features for each subgenre

### Temporal Features
- **`album_age`**: Years since album release (current_year - release_year)

### Subgenre Features (Tree-based models only)
- **`subgenre_avg_popularity`**: Average popularity for each subgenre

### Key Features
- **Data Leakage Prevention**: All statistics computed only on training data
- **Scikit-learn Compatible**: Implements `BaseEstimator` and `TransformerMixin`
- **Two Modes**: 
  - `isMLP=True`: For neural networks (one-hot subgenres only)
  - `isMLP=False`: For tree-based models (includes subgenre aggregations)

## 🤖 Models Implemented

| Model | Type | Hyperparameter Tuning | Notebook |
|-------|------|----------------------|----------|
| **Linear Regression** | Baseline | N/A | `lineer.ipynb` |
| **Lasso Regression** | Regularized Linear | Grid Search | `lasso_regression.ipynb` |
| **Ridge Regression** | Regularized Linear | Grid Search | `ridge_ml.ipynb` |
| **Random Forest** | Ensemble | Random Search | `random_forest.ipynb` |
| **XGBoost** | Gradient Boosting | Optuna | `xgboost.ipynb` |
| **LightGBM** | Gradient Boosting | Optuna | `lightGBM.ipynb` |
| **Neural Network** | Deep Learning | Keras Tuner | `nn.ipynb` |

### Neural Network Architecture
- **Framework**: TensorFlow/Keras
- **Layers**: 1-3 hidden layers with 16-64 units
- **Activation**: ReLU
- **Regularization**: Batch Normalization + Dropout (0.1-0.3)
- **Optimizer**: Adam (learning rate: 1e-4 to 1e-3)
- **Loss Function**: Huber Loss (robust to outliers)
- **Tuning**: Keras Tuner with 10 trials
- **Saved Models**: `best_nn_spotify.h5` and `best_nn_spotify.keras`

## 📈 Results

Each model is evaluated using:
- **MAE** (Mean Absolute Error) - Primary metric
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Coefficient of Determination)

### Evaluation Strategy
- **Train Split**: 60% (for training)
- **Validation Split**: 20% (for hyperparameter tuning)
- **Test Split**: 20% (for final evaluation)
- **Random State**: 24 (for reproducibility)

### Model Performance
The best performing model configurations and metrics can be found in the individual model notebooks. The neural network model with optimized hyperparameters is saved in the `models/` directory.

## 🛠️ Technologies Used

### Data Processing & Analysis
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning pipeline and preprocessing

### Visualization
- **Matplotlib**: Base plotting library
- **Seaborn**: Statistical data visualization

### Machine Learning
- **Scikit-learn**: Linear models, Random Forest, preprocessing
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Light gradient boosting machine
- **TensorFlow/Keras**: Deep learning framework

### Hyperparameter Tuning
- **Optuna**: Bayesian optimization framework
- **Scikit-Optimize**: Sequential model-based optimization
- **Keras Tuner**: Hyperparameter tuning for Keras models

### Development Tools
- **Jupyter Notebook**: Interactive development environment
- **Pre-commit**: Git hooks for code quality
- **Nbstripout**: Clean notebook outputs from version control

## 🤝 Contributing

Contributions are welcome! Here are some ways you can contribute:

1. **Bug Reports**: Open an issue describing the bug
2. **Feature Requests**: Suggest new features or improvements
3. **Model Improvements**: Implement new algorithms or optimization techniques
4. **Documentation**: Improve code documentation or README
5. **Data Analysis**: Add new visualizations or insights to EDA

### Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project uses data from [TidyTuesday](https://github.com/rfordatascience/tidytuesday) which is shared under the MIT License. The Spotify data is sourced via the [`spotifyr` package](https://www.rcharlie.com/spotifyr/).

## 🙏 Acknowledgments

- **Data Source**: [TidyTuesday 2020-01-21](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21)
- **Original Blog Post**: [Kaylin Pavlik - Classifying Songs by Genre](https://www.kaylinpavlik.com/classifying-songs-genres/)
- **spotifyr Package**: [Charlie Thompson](https://twitter.com/_RCharlie), [Josiah Parry](https://twitter.com/JosiahParry), Donal Phipps, Tom Wolff
- **TidyTuesday**: [Jon Harmon](https://github.com/rfordatascience/tidytuesday/issues/160) & [Neal Grantham](https://twitter.com/nsgrantham/status/1213190975113199616)

---

## 📖 Complete Data Dictionary

### `spotify_songs.csv`

| Variable | Type | Description |
|----------|------|-------------|
| `track_id` | character | Song unique ID |
| `track_name` | character | Song name |
| `track_artist` | character | Song artist |
| `track_popularity` | double | Song popularity (0-100) where higher is better |
| `track_album_id` | character | Album unique ID |
| `track_album_name` | character | Song album name |
| `track_album_release_date` | character | Date when album released |
| `playlist_name` | character | Name of playlist |
| `playlist_id` | character | Playlist ID |
| `playlist_genre` | character | Playlist genre |
| `playlist_subgenre` | character | Playlist subgenre |
| `danceability` | double | How suitable a track is for dancing based on tempo, rhythm stability, beat strength, and regularity (0.0-1.0) |
| `energy` | double | Perceptual measure of intensity and activity. Energetic tracks feel fast, loud, and noisy (0.0-1.0) |
| `key` | double | Estimated overall key using Pitch Class notation (0=C, 1=C♯/D♭, 2=D, etc., -1=no key detected) |
| `loudness` | double | Overall loudness in decibels (dB). Values typically range between -60 and 0 dB |
| `mode` | double | Modality of track: Major=1, Minor=0 |
| `speechiness` | double | Presence of spoken words. >0.66=speech (audiobook), 0.33-0.66=music+speech (rap), <0.33=music |
| `acousticness` | double | Confidence measure if track is acoustic (0.0-1.0) |
| `instrumentalness` | double | Predicts if track contains no vocals. >0.5=instrumental, closer to 1.0=higher confidence |
| `liveness` | double | Presence of audience in recording. >0.8=strong likelihood of live performance |
| `valence` | double | Musical positiveness. High valence=positive (happy, cheerful), low valence=negative (sad, angry) |
| `tempo` | double | Estimated tempo in beats per minute (BPM) |
| `duration_ms` | double | Duration of song in milliseconds |

---

**Happy Analyzing! 🎵📊**
