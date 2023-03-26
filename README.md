# 연구 계획: 도시 소리 데이터를 활용한 녹음 환경 추론
## 1. 프로젝트 개요

이 연구는 AIhub의 "도시 소리 데이터" 데이터 세트를 활용하여 음원이 어디서 녹음되었는지 추론하는 모델을 만들고자 합니다. 

이를 위해 자연어 처리에서 좋은 성능을 보여주고있는 Transformer 모델과, AIhub에서 제공하는 "도시 소리 데이터" 데이터 세트를 활용하여 학습을 진행합니다.

## 2. 연구 목적

이 연구의 목적은 다음과 같습니다:

도시 소리 데이터를 활용하여 현재 장소를 정확하게 예측하는 딥러닝 모델 개발
  - 복잡한 도시 환경에서 다양한 소리를 분석하여 현재 장소를 정확하게 인식하고 분류하는 능력을 가진 모델 개발.

Transformer 아키텍처를 사용하여 음성 데이터를 처리하는 방법 탐구
  - Transformer 아키텍처는 자연어 처리 분야에서 뛰어난 성능을 보여주고 있으며, 이 연구에서는 음성 데이터를 처리하는 데 있어서도 효과적인지 확인.

기존 음성 인식 기술과 비교하여 모델의 성능과 효율성 평가 
  - 이 연구에서 개발된 모델을 기존 음성 인식 기술과 비교하여 성능 및 효율성을 평가하고, 장단점을 분석하여 보다 효과적인 방법을 제안.

## 3. 데이터셋

AIhub의 "도시 소리 데이터"는 다양한 도시 환경에서 녹음된 소리 데이터로 구성되어 있습니다. 이 데이터셋은 아래와 같은 특징을 가지고 있습니다.

다양한 도시 환경
  - 데이터 세트의 음원은 다음과 같이 구성 됩니다.
      - 교통소음 : 자동차, 이륜 자동차, 항공기, 열차
      - 생활 소음 : 충격, 가전, 동물, 도구
      - 사업장 소음 : 공사장, 공장
  - 이를 통해 모델은 다양한 환경에서 발생하는 소리를 학습하게 됩니다.

레이블링
  - 데이터는 각 환경에 대한 레이블이 부여되어 있어, 지도 학습을 통해 모델을 학습시킬 수 있습니다.

데이터 전처리
 - 원본 음성 데이터는 일정한 샘플링 레이트와 길이로 전처리되어 학습에 적합합니다.


## 4. 예상 결과 및 향후 연구 방향

AIhub의 "도시 소리 데이터" 데이터 세트를 활용하여 Transformer 모델을 학습시킴으로서 녹음된 환경을 예측할 수 있습니다.

