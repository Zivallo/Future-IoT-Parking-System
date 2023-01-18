#include "EspMQTTClient.h"

EspMQTTClient client(
  "SSAFY_EMB_2",
  "1210@ssafy",
  "192.168.110.201",
  "MQTTUsername",
  "MQTTPassword",
  "hifaker1",
  1883        
);


uint32_t delayMS;

int trig = 15;
int echo = 2;

int led1 = 17;
int led2 = 16;


void setup() {
  pinMode(led1, OUTPUT);  
  pinMode(led2, OUTPUT);  

  pinMode(trig,OUTPUT);
  pinMode(echo,INPUT);

 // pinMode(led3, OUTPUT); 
  pinMode(A0, INPUT);
  Serial.begin(9600);

  client.enableDebuggingMessages();
  client.enableHTTPWebUpdater(); // Enable the web updater. User and password default to values of MQTTUsername and MQTTPassword. These can be overridded with enableHTTPWebUpdater("user", "password").
  client.enableOTA(); // Enable OTA (Over The Air) updates. Password defaults to MQTTPassword. Port is the default OTA port. Can be overridden with enableOTA("password", port).
  client.enableLastWillMessage("TestClient/lastwill", "I am going offline");  // You can activate the retain flag by setting the third parameter to true
}

void onConnectionEstablished()
{
  client.subscribe("myroom/led1", [](const String & payload) {
    int a = 1;
  });
}

void loop() {
  client.loop();

  long duration, distance;
  
  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);



  duration = pulseIn (echo, HIGH); //물체에 반사되어돌아온 초음파의 시간을 변수에 저장합니다.
  //34000*초음파가 물체로 부터 반사되어 돌아오는시간 
  ///1000000 / 2(왕복값이아니라 편도값이기때문에 나누기2를 해줍니다.)

  //초음파센서의 거리값이 위 계산값과 동일하게 Cm로 환산되는 계산공식 입니다. 수식이 간단해지도록 적용했습니다.

  distance = duration * 17 / 1000; 


  //PC모니터로 초음파 거리값을 확인 하는 코드 입니다.
  //Serial.println(duration ); //초음파가 반사되어 돌아오는 시간을 보여줍니다.
  Serial.print("\nDIstance1 : ");
  Serial.print(distance); //측정된 물체로부터 거리값(cm값)을 보여줍니다.
  Serial.println(" Cm");
  delay(500);

  if(distance < 10){
    digitalWrite(led2,HIGH);
    digitalWrite(led1,LOW);
    client.publish("parkinglot/place1", "1");
  }
  else{
    digitalWrite(led1,HIGH);
    digitalWrite(led2,LOW);
    client.publish("parkinglot/place1", "0");
  }

  
}