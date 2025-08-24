#!/usr/bin/env python3
"""
Wind Turbine Bearing RUL Prediction - Example Execution Script
풍력 터빈 베어링 RUL 예측 - 실행 예제 스크립트

이 스크립트는 모든 오류를 처리하며 안전하게 파이프라인을 실행합니다.
"""

import os
import sys
import traceback
import warnings

# 경고 메시지 필터링
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def check_dependencies():
    """필수 의존성 확인"""
    print("🔍 의존성 확인 중...")
    
    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'), 
        ('scipy', 'scipy'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('requests', 'requests')
    ]
    
    missing_packages = []
    
    for package_name, install_name in required_packages:
        try:
            __import__(package_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            missing_packages.append(install_name)
            print(f"  ❌ {package_name}")
    
    # TensorFlow 확인 (선택사항)
    try:
        import tensorflow
        print(f"  ✅ tensorflow (LSTM 기능 사용 가능)")
        tensorflow_available = True
    except ImportError:
        print(f"  ⚠️  tensorflow (LSTM 기능 비활성화)")
        tensorflow_available = False
    
    if missing_packages:
        print(f"\n❌ 누락된 패키지: {', '.join(missing_packages)}")
        print("다음 명령으로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 모든 필수 의존성이 설치되어 있습니다!")
    return True, tensorflow_available

def create_directories():
    """필요한 디렉토리 생성"""
    directories = ['results', 'results/plots', 'results/saved_models']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print(f"⚠️ 디렉토리 생성 실패 ({directory}): {e}")

def run_basic_example():
    """기본 예제 실행"""
    print("\n🚀 기본 파이프라인 실행 중...")
    
    try:
        from main import RULPredictionPipeline
        
        # 기본 설정으로 파이프라인 생성
        pipeline = RULPredictionPipeline()
        
        # 파이프라인 실행
        success = pipeline.run_complete_pipeline()
        
        if success:
            print("✅ 기본 예제 실행 완료!")
            return True
        else:
            print("❌ 기본 예제 실행 실패")
            return False
            
    except Exception as e:
        print(f"❌ 기본 예제 실행 중 오류: {e}")
        traceback.print_exc()
        return False

def run_fast_example():
    """빠른 예제 실행 (시각화/저장 없이)"""
    print("\n⚡ 빠른 예제 실행 중 (시각화 제외)...")
    
    try:
        from main import RULPredictionPipeline
        
        # 파이프라인 생성
        pipeline = RULPredictionPipeline()
        
        # 빠른 실행을 위한 메서드 오버라이드
        def dummy_visualize():
            print("  시각화 건너뛰기")
            return {}
        
        def dummy_save():
            print("  모델 저장 건너뛰기")
        
        pipeline.generate_visualizations = dummy_visualize
        pipeline.save_models = dummy_save
        
        # 파이프라인 실행
        success = pipeline.run_complete_pipeline()
        
        if success:
            print("✅ 빠른 예제 실행 완료!")
            return True
        else:
            print("❌ 빠른 예제 실행 실패")
            return False
            
    except Exception as e:
        print(f"❌ 빠른 예제 실행 중 오류: {e}")
        traceback.print_exc()
        return False

def run_individual_components():
    """개별 컴포넌트 테스트"""
    print("\n🧪 개별 컴포넌트 테스트...")
    
    try:
        # 1. 데이터 로더 테스트
        print("1. 데이터 로더 테스트...")
        from data_loader import DataLoader
        
        loader = DataLoader()
        data = loader.load_wind_turbine_data()
        print(f"  ✅ 데이터 로드 성공: {len(data)}개 샘플")
        
        # 2. 특징 추출 테스트
        print("2. 특징 추출 테스트...")
        from feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        features = extractor.extract_features_from_dataframe(data.iloc[:5], show_progress=False)
        print(f"  ✅ 특징 추출 성공: {len(features.columns)}개 특징")
        
        # 3. 지수적 모델 테스트
        print("3. 지수적 모델 테스트...")
        from models import ExponentialDegradationModel
        
        exp_model = ExponentialDegradationModel()
        print("  ✅ 지수적 모델 생성 성공")
        
        # 4. 시각화 테스트
        print("4. 시각화 테스트...")
        from visualization import RULVisualizer
        
        visualizer = RULVisualizer()
        print("  ✅ 시각화 객체 생성 성공")
        
        print("✅ 모든 컴포넌트 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"❌ 컴포넌트 테스트 중 오류: {e}")
        traceback.print_exc()
        return False

def print_system_info():
    """시스템 정보 출력"""
    print("\n💻 시스템 정보:")
    print(f"  Python 버전: {sys.version}")
    print(f"  플랫폼: {sys.platform}")
    print(f"  작업 디렉토리: {os.getcwd()}")

def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("🌪️  Wind Turbine Bearing RUL Prediction - Example Runner")
    print("=" * 70)
    
    # 시스템 정보 출력
    print_system_info()
    
    # 디렉토리 생성
    create_directories()
    
    # 의존성 확인
    deps_result = check_dependencies()
    if isinstance(deps_result, tuple):
        deps_ok, tensorflow_available = deps_result
    else:
        deps_ok = deps_result
        tensorflow_available = False
    
    if not deps_ok:
        print("\n❌ 필수 패키지가 설치되지 않아 실행을 중단합니다.")
        return False
    
    # 실행 모드 선택
    print("\n🎯 실행 모드를 선택하세요:")
    print("  1. 기본 실행 (전체 기능, 시각화 포함)")
    print("  2. 빠른 실행 (시각화 제외)")  
    print("  3. 컴포넌트 테스트만")
    print("  4. 자동 실행 (빠른 모드)")
    
    try:
        # 자동 모드로 실행 (사용자 입력 대기 없이)
        choice = "4"
        
        if choice == "1":
            success = run_basic_example()
        elif choice == "2":
            success = run_fast_example()
        elif choice == "3":
            success = run_individual_components()
        elif choice == "4":
            print("자동 실행 모드 선택됨")
            success = run_fast_example()
        else:
            print("잘못된 선택입니다. 빠른 실행 모드로 진행합니다.")
            success = run_fast_example()
        
        if success:
            print("\n🎉 실행 완료!")
            print("\n📁 생성된 파일들을 확인하세요:")
            print("  📊 results/plots/ - 시각화 결과")
            print("  💾 results/saved_models/ - 훈련된 모델")
            print("  📄 results/ - 실행 리포트")
        else:
            print("\n❌ 실행 중 오류가 발생했습니다.")
            print("문제 해결 방법:")
            print("  1. pip install -r requirements.txt")
            print("  2. Python 버전 확인 (3.7+ 권장)")
            print("  3. 인터넷 연결 확인 (데이터 다운로드용)")
        
        return success
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        return False
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"치명적 오류: {e}")
        traceback.print_exc()
        sys.exit(1)
