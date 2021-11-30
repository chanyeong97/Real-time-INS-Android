# Real-time-INS-Android

관성센서를 활용한 AI기반의 정밀 궤적 위치추정 모델을 이용한 실시간 관성항법시스템입니다. 학습에 관련된 코드는 [이곳](https://github.com/chanyeong97/Real-time-INS-python)에서 확인하실 수 있습니다.

## 개발환경

- Android Studio @31.1.4
- Android 7.0(API 수준 24)

## 기능

<p align="center"><img src="images/RealtimeINS_App.gif" width="30%" alt="" /></p>

- 0.25초 간격으로 사용자의 속도를 추정하여 표시합니다
- 속도를 적분하여 현재 위치를 계산합니다
- 사용자의 이동 궤적을 표시합니다
