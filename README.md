# Future-IoT-Parking-System (미래형 IoT 주차 시스템)
[SSAFY 관통PJT]

## 1. Topic
라즈베리파이 Sensing Data를 이용한 IoT RC카 제작

## 2. Concept
### 1) Team Building
![image](https://user-images.githubusercontent.com/79623246/215255556-2369968d-10c9-4298-84de-5ed7c426d4d4.png)
### 2) Block Diagram of System
![image](https://user-images.githubusercontent.com/79623246/215258628-69a06cdb-a3c5-4ae5-acaf-85af56183f45.png)

## 3. Function
### 1) RC car
#### (1) Rear-Autonomous Emergency Braking(R-AEB, 후방 긴급자동제동)
![image](https://user-images.githubusercontent.com/79623246/215252751-742b2f77-db2f-4181-bc8e-30b92a87872a.png)
- OpenCV + Tensorflow lite 를 통한 객체 인식 -> 사람의 손을 사람으로 간주
#### (2) Infotainment (인포테인먼트)
![image](https://user-images.githubusercontent.com/79623246/215253598-379a9467-ad09-4ef4-b289-f10b2ecf2ca3.png)
- 버튼을 통해 차량 제어 가능
- Web browser 기능을 추가하여 주차장 상태 정보 확인 및 다양한 컨텐츠 이용 가능
- PyQT

### 2) Parking lot IoT System
![image](https://user-images.githubusercontent.com/79623246/215252943-200607f2-b192-4b06-a66b-e53094a40fce.png)
#### (1) Parking lot Lights system (주차장 상태 표시 점등 시스템)
- 초음파 센서를 통해 주차 여부 확인
- 주차 여부에 따라 LED 점등(주차 불가 - Red, 주차 가능 - Blue)
- 주차 상태를 MQTT 통신으로 라즈베리파이4에 전송
- 라즈베리파이4 에서 주차 상태를 DB에 저장
#### (2) Parking lot Status Web site (주차장 상태 표시 홈페이지)
![image](https://user-images.githubusercontent.com/79623246/215253211-ac6435fe-6f99-46fc-a1f2-055bed4ef605.png)
- 주차상 상황을 사용자 및 관리자가 쉽게 알 수 있도록 홈페이지 구축
- Node.js + Vue.js
- 라즈베리파이4가 저장한 DB 정보 사용 


## 4. Result
### 1) Hardware
![image](https://user-images.githubusercontent.com/79623246/215253478-22c2b576-46a6-4c95-bf2e-d0ec9e240be0.png)
### 2) Simulation
[![Simulation](https://img.youtube.com/vi/nmKbPrBvkBE/0.jpg)](https://youtu.be/nmKbPrBvkBE) 
> 이미지 클릭시 유튜브에서 영상 시청 가능합니다.
