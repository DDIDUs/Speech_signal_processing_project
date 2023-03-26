# 연구 계획: 다화자 음성합성 데이터를 이용한 음원분리
## 1. 프로젝트 개요
이 연구는 AIhub의 "다화자 음성합성 데이터"를 활용하여 음원분리 기술을 개발하고자 합니다. 음원분리는 오디오 신호에서 원하는 소리 성분을 추출하는 작업으로, 이를 통해 다양한 응용분야에서 활용할 수 있는 음원을 얻을 수 있습니다.

## 2. 연구 목적

이 연구의 목적은 다음과 같습니다:

도시 소리 데이터를 활용하여 현재 장소를 정확하게 예측하는 딥러닝 모델 개발
  - 복잡한 도시 환경에서 다양한 소리를 분석하여 현재 장소를 정확하게 인식하고 분류하는 능력을 가진 모델을 만들고자 합니다.

Transformer 아키텍처를 사용하여 음성 데이터를 처리하는 방법 탐구
  - Transformer 아키텍처는 자연어 처리 분야에서 뛰어난 성능을 보여주고 있으며, 이 연구에서는 음성 데이터를 처리하는 데 있어서도 효과적인지를 살펴보고자 합니다.

기존 음성 인식 기술과 비교하여 모델의 성능과 효율성 평가 
  - 이 연구에서 개발된 모델을 기존 음성 인식 기술과 비교하여 성능 및 효율성을 평가하고, 장단점을 분석하여 보다 효과적인 방법을 제안하고자 합니다.

## 3. 데이터셋

AIhub의 "도시 소리 데이터"는 다양한 도시 환경에서 녹음된 소리 데이터로 구성되어 있습니다. 이 데이터셋은 아래와 같은 특징을 가지고 있습니다:

다양한 도시 환경
  - 데이터셋에 포함된 소리는 거리, 공원, 시장, 지하철, 카페 등 다양한 도시 환경에서 녹음되었습니다. 이를 통해 모델은 다양한 환경에서 발생하는 소리를 학습할 수 있습니다.

레이블링
  - 데이터는 각 환경에 대한 레이블이 부여되어 있어, 지도 학습을 통해 모델을 학습시킬 수 있습니다. 이를 통해 모델은 각 환경에 특징적인 소리 패턴을 학습하게 됩니다.

데이터 양
 - 대규모 데이터셋으로 구성되어 있어, 딥러닝 모델을 학습시키기에 충분한 양의 데이터를 제공합니다. 이를 통해 모델의 일반화 능력을 향상시킬 수 있습니다.

데이터 전처리
 - 원본 음성 데이터는 일정한 샘플링 레이트와 길이로 전처리되어야 합니다. 이를 통해 모델에 입력될 데이터의 일관성을 유지할 수 있습니다.


## 4. 예상 결과 및 향후 연구 방향

AIhub의 "도시 소리 데이터" 데이터 세트를 활용하여 Transformer 모델을 학습시킴으로서 녹음된 환경을 예측할 수 있다.

