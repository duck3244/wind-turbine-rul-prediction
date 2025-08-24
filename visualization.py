"""
Visualization module for Wind Turbine Bearing RUL Prediction
풍력 터빈 베어링 RUL 예측을 위한 시각화 모듈 (새로 작성)

간결하고 안정적인 시각화 구현
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Tuple, Any
import warnings

# matplotlib 설정
plt.style.use('default')
try:
    import seaborn as sns

    sns.set_palette("husl")
except ImportError:
    pass

# 경고 필터링
warnings.filterwarnings('ignore')

try:
    from config import PLOTS_DIR, get_plot_path, get_timestamp
except ImportError:
    # config 모듈이 없는 경우 기본값 사용
    PLOTS_DIR = 'plots'


    def get_timestamp():
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")


    def get_plot_path(name):
        os.makedirs(PLOTS_DIR, exist_ok=True)
        return os.path.join(PLOTS_DIR, f"{name}_{get_timestamp()}.png")


class RULVisualizer:
    """간단하고 안정적인 RUL 시각화 클래스"""

    def __init__(self):
        # 색상 설정
        self.colors = {
            'true_rul': 'black',
            'exponential': 'blue',
            'lstm': 'red',
            'health_indicator': 'green',
            'threshold': 'red'
        }

        # 디렉토리 생성
        os.makedirs(PLOTS_DIR, exist_ok=True)

    def plot_health_indicator(self, health_indicator: np.ndarray,
                              dates: Optional[pd.DatetimeIndex] = None,
                              threshold: Optional[float] = None,
                              title: str = "Health Indicator Evolution") -> plt.Figure:
        """건강 지표 시각화"""

        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            # x축 설정
            if dates is not None and len(dates) >= len(health_indicator):
                x_axis = dates[:len(health_indicator)]
                ax.plot(x_axis, health_indicator, 'o-',
                        color=self.colors['health_indicator'],
                        linewidth=2, markersize=4)
                # 날짜 형식 설정
                fig.autofmt_xdate()
            else:
                x_axis = range(len(health_indicator))
                ax.plot(x_axis, health_indicator, 'o-',
                        color=self.colors['health_indicator'],
                        linewidth=2, markersize=4)

            # 임계값 표시
            if threshold is not None:
                ax.axhline(y=threshold, color=self.colors['threshold'],
                           linestyle='--', linewidth=2,
                           label=f'Threshold ({threshold:.3f})')
                ax.legend()

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Health Indicator Value')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"건강 지표 시각화 오류: {e}")
            # 빈 figure 반환
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, f'시각화 오류\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

    def plot_rul_predictions(self, true_rul: np.ndarray,
                             predictions: Dict[str, np.ndarray],
                             dates: Optional[pd.DatetimeIndex] = None,
                             title: str = "RUL Predictions Comparison") -> plt.Figure:
        """RUL 예측 결과 비교 시각화"""

        try:
            fig, ax = plt.subplots(figsize=(14, 8))

            # x축 설정
            max_len = len(true_rul)
            if dates is not None and len(dates) >= max_len:
                x_axis = dates[:max_len]
                fig.autofmt_xdate()
            else:
                x_axis = range(max_len)

            # 실제 RUL
            ax.plot(x_axis, true_rul,
                    color=self.colors['true_rul'], linewidth=3,
                    label='True RUL', zorder=10)

            # 예측 결과들
            colors = ['blue', 'red', 'green', 'orange', 'purple']

            for i, (model_name, pred_values) in enumerate(predictions.items()):
                if len(pred_values) == 0:
                    continue

                # 길이 맞추기
                aligned_len = min(len(pred_values), max_len)
                pred_aligned = pred_values[:aligned_len]
                x_aligned = x_axis[:aligned_len]

                color = self.colors.get(model_name.lower(), colors[i % len(colors)])

                ax.plot(x_aligned, pred_aligned,
                        '--', color=color, linewidth=2,
                        label=f'{model_name} Prediction', alpha=0.8)

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('RUL (days)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"RUL 예측 시각화 오류: {e}")
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.text(0.5, 0.5, f'시각화 오류\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

    def plot_performance_metrics(self, model_results: Dict[str, Dict[str, float]]) -> plt.Figure:
        """모델 성능 메트릭 시각화"""

        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            models = list(model_results.keys())

            # MAE 비교
            ax1 = axes[0]
            mae_values = []
            valid_models = []

            for model in models:
                if 'mae' in model_results[model]:
                    mae = model_results[model]['mae']
                    if np.isfinite(mae):
                        mae_values.append(mae)
                        valid_models.append(model)

            if len(mae_values) > 0:
                colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
                bars1 = ax1.bar(valid_models, mae_values,
                                color=colors[:len(valid_models)], alpha=0.7)
                ax1.set_title('Mean Absolute Error (MAE)', fontweight='bold')
                ax1.set_ylabel('MAE (days)')
                ax1.grid(True, axis='y', alpha=0.3)

                # 값 레이블
                for bar, value in zip(bars1, mae_values):
                    ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                             f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax1.text(0.5, 0.5, 'No valid MAE data', ha='center', va='center',
                         transform=ax1.transAxes)

            # RMSE 비교  
            ax2 = axes[1]
            rmse_values = []
            valid_models_rmse = []

            for model in models:
                if 'rmse' in model_results[model]:
                    rmse = model_results[model]['rmse']
                    if np.isfinite(rmse):
                        rmse_values.append(rmse)
                        valid_models_rmse.append(model)

            if len(rmse_values) > 0:
                bars2 = ax2.bar(valid_models_rmse, rmse_values,
                                color=colors[:len(valid_models_rmse)], alpha=0.7)
                ax2.set_title('Root Mean Square Error (RMSE)', fontweight='bold')
                ax2.set_ylabel('RMSE (days)')
                ax2.grid(True, axis='y', alpha=0.3)

                # 값 레이블
                for bar, value in zip(bars2, rmse_values):
                    ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                             f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No valid RMSE data', ha='center', va='center',
                         transform=ax2.transAxes)

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"성능 메트릭 시각화 오류: {e}")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.text(0.5, 0.5, f'시각화 오류\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

    def plot_residuals(self, true_rul: np.ndarray,
                       predictions: Dict[str, np.ndarray]) -> plt.Figure:
        """잔차 분석 시각화"""

        try:
            n_models = len(predictions)
            if n_models == 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, 'No prediction data available',
                        ha='center', va='center', transform=ax.transAxes)
                return fig

            fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models))
            if n_models == 1:
                axes = [axes]

            colors = ['blue', 'red', 'green', 'orange']

            for i, (model_name, pred_values) in enumerate(predictions.items()):
                ax = axes[i]

                if len(pred_values) > 0:
                    # 잔차 계산
                    min_len = min(len(pred_values), len(true_rul))
                    residuals = pred_values[:min_len] - true_rul[:min_len]

                    # 유효한 값들만 사용
                    valid_mask = np.isfinite(residuals)
                    if np.sum(valid_mask) > 0:
                        valid_residuals = residuals[valid_mask]
                        x_valid = np.arange(len(valid_residuals))

                        color = colors[i % len(colors)]
                        ax.plot(x_valid, valid_residuals, 'o-',
                                color=color, markersize=3, alpha=0.7)
                        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

                        # 통계 정보
                        mean_res = np.mean(valid_residuals)
                        std_res = np.std(valid_residuals)
                        ax.text(0.02, 0.98, f'Mean: {mean_res:.3f}\nStd: {std_res:.3f}',
                                transform=ax.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    else:
                        ax.text(0.5, 0.5, 'No valid residuals',
                                ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, 'No prediction data',
                            ha='center', va='center', transform=ax.transAxes)

                ax.set_title(f'{model_name} Model Residuals', fontweight='bold')
                ax.set_xlabel('Time')
                ax.set_ylabel('Prediction Error (days)')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"잔차 분석 시각화 오류: {e}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'시각화 오류\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

    def plot_feature_importance(self, feature_ranking: List[Tuple[str, float]],
                                top_n: int = 10) -> plt.Figure:
        """특징 중요도 시각화"""

        try:
            if len(feature_ranking) == 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, 'No feature ranking data',
                        ha='center', va='center', transform=ax.transAxes)
                return fig

            fig, ax = plt.subplots(figsize=(10, 6))

            # 상위 N개 특징
            top_features = feature_ranking[:min(top_n, len(feature_ranking))]
            features = [f[0] for f in top_features]
            scores = [f[1] for f in top_features]

            if len(features) > 0:
                # 수평 막대 그래프
                y_pos = np.arange(len(features))
                bars = ax.barh(y_pos, scores, color='steelblue', alpha=0.7)

                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.invert_yaxis()
                ax.set_xlabel('Importance Score')
                ax.set_title(f'Top {len(features)} Feature Importance', fontweight='bold')
                ax.grid(True, axis='x', alpha=0.3)

                # 값 레이블
                for i, (bar, score) in enumerate(zip(bars, scores)):
                    width = bar.get_width()
                    ax.text(width + max(scores) * 0.01, bar.get_y() + bar.get_height() / 2,
                            f'{score:.3f}', ha='left', va='center')
            else:
                ax.text(0.5, 0.5, 'No valid features',
                        ha='center', va='center', transform=ax.transAxes)

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"특징 중요도 시각화 오류: {e}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'시각화 오류\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

    def create_summary_dashboard(self,
                                 health_indicator: np.ndarray,
                                 true_rul: np.ndarray,
                                 predictions: Dict[str, np.ndarray],
                                 model_results: Dict[str, Dict[str, float]],
                                 dates: Optional[pd.DatetimeIndex] = None) -> plt.Figure:
        """요약 대시보드 생성"""

        try:
            fig = plt.figure(figsize=(16, 12))

            # 2x2 레이아웃
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # 1. 건강 지표 (좌상단)
            ax1 = fig.add_subplot(gs[0, 0])
            if len(health_indicator) > 0:
                if dates is not None and len(dates) >= len(health_indicator):
                    x_axis = dates[:len(health_indicator)]
                else:
                    x_axis = range(len(health_indicator))

                ax1.plot(x_axis, health_indicator, 'g-o', markersize=3, linewidth=2)
                threshold = health_indicator[-1]
                ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.7)
                ax1.set_title('Health Indicator', fontweight='bold')
                ax1.set_ylabel('Health Index')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No health indicator data',
                         ha='center', va='center', transform=ax1.transAxes)

            # 2. RUL 예측 (우상단)
            ax2 = fig.add_subplot(gs[0, 1])
            if len(true_rul) > 0 and len(predictions) > 0:
                max_len = len(true_rul)
                if dates is not None and len(dates) >= max_len:
                    x_axis = dates[:max_len]
                else:
                    x_axis = range(max_len)

                ax2.plot(x_axis, true_rul, 'k-', linewidth=3, label='True RUL')

                colors = ['blue', 'red', 'green']
                for i, (model_name, pred_values) in enumerate(predictions.items()):
                    if len(pred_values) > 0:
                        aligned_len = min(len(pred_values), max_len)
                        color = colors[i % len(colors)]
                        ax2.plot(x_axis[:aligned_len], pred_values[:aligned_len],
                                 '--', color=color, linewidth=2, label=model_name)

                ax2.set_title('RUL Predictions', fontweight='bold')
                ax2.set_ylabel('RUL (days)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No prediction data',
                         ha='center', va='center', transform=ax2.transAxes)

            # 3. 성능 메트릭 (좌하단)
            ax3 = fig.add_subplot(gs[1, 0])
            if len(model_results) > 0:
                models = list(model_results.keys())
                mae_values = []
                valid_models = []

                for model in models:
                    if 'mae' in model_results[model]:
                        mae = model_results[model]['mae']
                        if np.isfinite(mae):
                            mae_values.append(mae)
                            valid_models.append(model)

                if len(mae_values) > 0:
                    bars = ax3.bar(valid_models, mae_values,
                                   color=['skyblue', 'lightcoral', 'lightgreen'][:len(valid_models)])
                    ax3.set_title('Model Performance (MAE)', fontweight='bold')
                    ax3.set_ylabel('MAE (days)')

                    for bar, value in zip(bars, mae_values):
                        ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                                 f'{value:.2f}', ha='center', va='bottom')

                    ax3.grid(True, axis='y', alpha=0.3)
                else:
                    ax3.text(0.5, 0.5, 'No performance data',
                             ha='center', va='center', transform=ax3.transAxes)
            else:
                ax3.text(0.5, 0.5, 'No model results',
                         ha='center', va='center', transform=ax3.transAxes)

            # 4. 요약 정보 (우하단)
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis('off')

            summary_text = "Analysis Summary\n" + "=" * 20 + "\n\n"

            if len(health_indicator) > 0:
                summary_text += f"Data Points: {len(health_indicator)}\n"
                summary_text += f"HI Range: {health_indicator.min():.3f} ~ {health_indicator.max():.3f}\n\n"

            if len(model_results) > 0:
                summary_text += "Model Performance:\n"
                for model_name, results in model_results.items():
                    if 'mae' in results:
                        summary_text += f"  {model_name}: {results['mae']:.3f} days\n"

                # 최고 성능 모델
                valid_results = {k: v for k, v in model_results.items()
                                 if 'mae' in v and np.isfinite(v['mae'])}
                if valid_results:
                    best_model = min(valid_results.keys(), key=lambda k: valid_results[k]['mae'])
                    summary_text += f"\nBest: {best_model}\n"

            if dates is not None and len(dates) > 0:
                summary_text += f"\nDate Range:\n{dates[0].strftime('%Y-%m-%d')}\nto\n{dates[min(len(dates) - 1, len(health_indicator) - 1)].strftime('%Y-%m-%d')}"

            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            # 전체 제목
            fig.suptitle('Wind Turbine Bearing RUL Prediction - Analysis Dashboard',
                         fontsize=16, fontweight='bold', y=0.95)

            return fig

        except Exception as e:
            print(f"요약 대시보드 생성 오류: {e}")
            fig, ax = plt.subplots(figsize=(16, 12))
            ax.text(0.5, 0.5, f'대시보드 생성 오류\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig

    def save_plots(self, figures: Dict[str, plt.Figure]) -> List[str]:
        """그래프들을 파일로 저장"""

        # 디렉토리 생성
        os.makedirs(PLOTS_DIR, exist_ok=True)

        saved_paths = []

        for name, fig in figures.items():
            if fig is not None:
                try:
                    filepath = get_plot_path(name)
                    fig.savefig(filepath, dpi=300, bbox_inches='tight',
                                facecolor='white', edgecolor='none')
                    saved_paths.append(filepath)
                    print(f"  ✅ {name} 저장: {os.path.basename(filepath)}")

                except Exception as e:
                    print(f"  ❌ {name} 저장 실패: {e}")
                    continue

        return saved_paths


class DataExplorer:
    """데이터 탐색 시각화 클래스"""

    def __init__(self):
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))

    def plot_signal_overview(self, data: pd.DataFrame,
                             n_samples: int = 5) -> plt.Figure:
        """진동 신호 개요 시각화"""

        try:
            if len(data) == 0:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.text(0.5, 0.5, 'No data available',
                        ha='center', va='center', transform=ax.transAxes)
                return fig

            fig, axes = plt.subplots(2, 1, figsize=(14, 10))

            # 샘플 선택
            n_samples = min(n_samples, len(data))
            indices = np.linspace(0, len(data) - 1, n_samples, dtype=int)

            # 시간 영역 신호
            ax1 = axes[0]
            for i, idx in enumerate(indices):
                vibration = data.iloc[idx]['vibration']
                if vibration is not None and len(vibration) > 0:
                    # 다운샘플링
                    step = max(1, len(vibration) // 2000)
                    t = np.arange(0, len(vibration), step) / 97656  # 샘플링 주파수
                    v = vibration[::step]

                    date_str = data.iloc[idx]['Date'].strftime('%m-%d') if 'Date' in data.columns else f'Sample {idx}'
                    ax1.plot(t, v, color=self.colors[i % len(self.colors)],
                             alpha=0.7, label=f'Day {idx + 1} ({date_str})')

            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Acceleration (g)')
            ax1.set_title('Vibration Signals Over Time', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # RMS 추세
            ax2 = axes[1]
            rms_values = []
            dates = []

            for _, row in data.iterrows():
                vibration = row.get('vibration')
                if vibration is not None and len(vibration) > 0:
                    rms = np.sqrt(np.mean(vibration ** 2))
                    rms_values.append(rms)
                    if 'Date' in row:
                        dates.append(row['Date'])
                    else:
                        dates.append(len(dates))

            if len(rms_values) > 0:
                ax2.plot(dates, rms_values, 'ro-', markersize=4, linewidth=2)
                ax2.set_xlabel('Date' if isinstance(dates[0], pd.Timestamp) else 'Sample')
                ax2.set_ylabel('RMS (g)')
                ax2.set_title('RMS Trend Over Time', fontweight='bold')
                ax2.grid(True, alpha=0.3)

                if isinstance(dates[0], pd.Timestamp):
                    fig.autofmt_xdate()

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"신호 개요 시각화 오류: {e}")
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.text(0.5, 0.5, f'시각화 오류\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            return fig


def main():
    """시각화 모듈 테스트"""
    print("시각화 모듈 테스트")
    print("=" * 40)

    try:
        # 테스트 데이터 생성
        np.random.seed(42)
        n_points = 50

        # 건강 지표 (시간에 따라 증가)
        health_indicator = np.cumsum(np.random.normal(0.1, 0.05, n_points))

        # 실제 RUL (감소)
        true_rul = np.array([n_points - i for i in range(n_points)])

        # 모의 예측 결과
        predictions = {
            'Exponential': true_rul + np.random.normal(0, 2, n_points),
            'LSTM': true_rul + np.random.normal(0, 1.5, n_points)
        }

        # 모델 결과
        model_results = {
            'Exponential': {'mae': 2.5, 'rmse': 3.2},
            'LSTM': {'mae': 1.8, 'rmse': 2.4}
        }

        # 시각화 테스트
        visualizer = RULVisualizer()

        print("1. 건강 지표 시각화 테스트...")
        hi_fig = visualizer.plot_health_indicator(health_indicator)

        print("2. RUL 예측 시각화 테스트...")
        pred_fig = visualizer.plot_rul_predictions(true_rul, predictions)

        print("3. 성능 메트릭 시각화 테스트...")
        metrics_fig = visualizer.plot_performance_metrics(model_results)

        print("4. 요약 대시보드 테스트...")
        dashboard_fig = visualizer.create_summary_dashboard(
            health_indicator, true_rul, predictions, model_results
        )

        # 저장 테스트
        print("5. 파일 저장 테스트...")
        figures = {
            'health_indicator_test': hi_fig,
            'rul_predictions_test': pred_fig,
            'performance_metrics_test': metrics_fig,
            'dashboard_test': dashboard_fig
        }

        saved_paths = visualizer.save_plots(figures)
        print(f"✅ {len(saved_paths)}개 파일 저장 완료")

        print("\n시각화 모듈 테스트 완료!")

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()