## quick instructions on datasets
> ~~현재 `tfds.load("emnist")`가 작동하지 않음(사이트 리뉴얼로 인한 링크 corrupt로 추정)
> 그래서 이 [링크](https://www.kaggle.com/datasets/crawford/emnist/download?datasetVersionNumber=3)로 직접 dataset을 받은 후
> 해당 파일(archive.zip)을 `datasets/emnist`에 위치시켜야 함(압축 풀 필요 없음)~~
> 
> 해결됨. 그냥 notebook 돌리셈

TODO:
1. preprocessing
  - [ ] data augumentation: rotate은 그렇다 치는 데 crop, filp은 data integrity를 많이 헤칠 것 같음(추측)...
    [tf api 참고](https://www.tensorflow.org/tutorials/images/data_augmentation)
2. ResNet50
   - [x] `tf.resize()` -> `tf.grayscale_to_rgb()`를 이용해 (224,224,3) 혹은 (28,28,3) 으로 조절 후 train
    문제점: (224,224,3) 은 시간이 너무 걸림. 그리고 grayscale을 다시 rgb로 불리는 거는 뭔가 어불성설 같음
   - [ ] ResNet50을 베이스로 아예 새 model를 제작. 즉 기본적으로는 output이 224 -> 112 -> 56 -> ... -> 7 -> 1(fc) 식으로 3 x 3 mask로 반씩 줄어드는데, 동일한 residual Network로 28 -> 14 -> 7 -> 1(fc) 으로 단순화(연구중)

- [ ] ~~큰 dataset(`byclass`,`bymerge`)는 `read_csv()`로 처리 시 도중에 튕김~~
> `tf.load("mnist")`로 tensor를 load하면 됨
- [x] resNet-50 시운전
- [ ] 데이터 전처리 -> 증강?
- [ ] fine-tune resNet50
- [ ] pretrain model 사용과 아닌 model 사용 중 어느 것이 더 나은지 확인

## 공통
- model의 학습 과정을 tensorboard로 시각화 및 정리(교재 notebook 참고)
- model 저장 및 test evaluate 결과


### LeNet5
- batch_size, callbacks, activation functions 등을 변경해 가며 최적의 값을 도출
- 현재 emnist 이미지는 (28,28) 인데, LeNet5는 (32,32)를 요구함. 현재는 `tf.keras.resize()`를 이용해 (32,32)로 stretch한 상태인데 그 대신
  > `tf.pad` 을 이용해 끝부분에 padding 해여 (32,32)를 만들어 비교
  > 
  > model의 input_size를 (28,28,1)로 만들어 비교
  >

### ResNet50
- 모델에 imagenet 가중치를 두고 하는 방식과 없이 하는 방식의 비교(각자 분담)
  > feature extraction, fine-tuning
  >
  > 혹은 train from zero
  > 

* * *
## 2024 ANN Project Guide
### 주제: MNIST extended dataset을 이용한 CNN 모델 최적화 및 분석
### 프로젝트 일정

|      일정       | 기한                    |
|:-------------:|:----------------------|
|     팀 구성      | 4월 17-23일             |
| 프로젝트 수행계획서 제출 | 4월 30일(화) 23:59       |
|     중간 발표     | 23년 5월 22일(12주차 수업시간) |
|   최종 결과물 제출   | 23년 6월 4일 23:59      |
|     최종 발표     | 23년 6월 5일(14주차 수업시간) -> 일정 변동중 |

### 프로젝트 목표
MNIST extended dataset을 이용하여 직접 설계한 CNN 또는 pretrained CNN모델을 이
용하여 학습시키고 accuracy와 inference time의 적절한 조합을 찾는다.

### 프로젝트 수행 방법
1. MNIST extended dataset을 분석하고 LeNet-5와 ResNet-50를 이용하여 학습하고
결과를 비교한다. (baseline) Hyperparameter 변경을 통해 최적의 학습 결과를 얻는
다. (baseline은 연구에서 새로운 모델이나 방법의 성능을 객관적으로 평가하기 위한 기준점으로 사용되는 모델이나 알고리즘을 뜻함)
2. Accuracy와 학습시간, 추론시간을 고려하여 직접 설계한 네트워크와 Keras의 pretrained model을 이용한 CNN 모델을 선정하고 학습하여 결과를 분석한다. 이때
CNN 모델 선정 과정과 결과를 실험 및 분석 결과에 근거하여 제시한다.

**수행 과정**
1) MNIST extended dataset 개요 및 분석: (Get the data/Discover and visualize the data) - 여러 종류의
 dataset 중 하나를 선택: 근거 제시.
2) Dataset 분류: training/validation/test dataset으로 분배 (Prepare the data)
3) Baseline 학습 및 결과분석 (LeNet5/ResNet-50)
4) 학습에 사용할 모델 선택: 다수의 후보를 대상으로 학습 결과를 이용하여 최종 모델
선택. 근거와 분석결과 제시. (Select and train a model) 노트북 파일에 이러한 과정이 나타나야 함. (직접 설계한 모델과 pre-trained model 포함)
5) 모델 최적화 및 분석:
   - Model과 training hyperparameter의 최적화를 통해 최대 성능을 획득.
   - 최적화 과정 제시 및 결과 분석. 학습시간, 예측시간(inference time), 정확도 측면에서 분석 -> EMNIST dataset에 포함된 test dataset을 기반으로 한 성능 평가: 주어
   진 dataset에 대한 학습 최적화 결과 평가
   - Epoch에 따른 learning curve 제시.
   - 학습 dataset 선택을 포함 한 전반적인 CNN 개발 결과에 대한 평가(Fine tune the model)

### 요구 조건
1. 4인~5인 팀프로젝트 (42명 수업인원 이므로 4명씩 8팀, 5명씩 2팀이 가장 바람직)
2. 프로젝트 수행계획을 세우고 업무마다 역할 분담: 프로젝트 수행계획서 제출
3. 프로젝트 수행일지 작성
4. 제출자료: 수행계획서, 보고서, 발표자료, 실행결과 포함한 노트북 파일. 모든 파일은
pdf와 ipynb로 제출 (팀당 1개씩).
5. 보고서 및 발표자료 포함사항
   - 프로젝트 개요 및 목표
   - 수행계획 및 역할 분담 (수정한 계획 및 수행일지)
   - 수행과정 
   - 결론

### 평가 방식
- 보고서 및 시뮬레이션 결과 평가 (내용: 5, 팀웍:3, 동작 및 성능:3, 문제해결과정: 5)
- 문제의 분석과 solution 도출 과정이 나타나야 함(채점시 이 부분을 중점적으로 볼 예정) 
- 팀원별 역할과 기여 또한 평가에 반영 
  - 아이디어 및 분석 결과에 기여자 및 기여자를 수행계획서 및 보고서에 표시하고 팀원별 기여도를 총합이 100%가 되도록 자체적으로 평가하여 작성.
