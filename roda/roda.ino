#define pwm1 5
#define pwm2 6
#define pwm3 9
#define pwm4 10

int i, isStop = 0;

void setup() {
  pinMode(pwm1, OUTPUT);
  pinMode(pwm2, OUTPUT);
  pinMode(pwm3, OUTPUT);
  pinMode(pwm4, OUTPUT);
  Serial.begin(9600);
  Serial.setTimeout(1);
}

int kiri = 0;
int kanan = 0;

int handleLeftWheel = 0;
int handleRightWheel = 0;

void adjustSpeed(int leftWheel, int rightWheel) {
  kiri = leftWheel;
  kanan = 0.84 * rightWheel;
};

void setSpeed(int inputSpeed){
    if (kiri < inputSpeed)
        {
            if (kiri >= inputSpeed-5)
            {
                adjustSpeed(inputSpeed, inputSpeed);
            }
            else
            {
                handleLeftWheel+=5;
                handleRightWheel+=5;
                adjustSpeed(handleLeftWheel, handleRightWheel);
            }
            analogWrite(pwm1, 0);
            analogWrite(pwm2, kiri);
            analogWrite(pwm3, 0);
            analogWrite(pwm4, kanan);
            delay(100); 
        }
    if (kiri > inputSpeed)
    {
        if (kiri <= inputSpeed+5)
        {
            adjustSpeed(inputSpeed, inputSpeed);
        }
        else
        {
            handleLeftWheel-=5;
            handleRightWheel-=5;
            adjustSpeed(handleLeftWheel, handleRightWheel);
        }
        analogWrite(pwm1, 0);
        analogWrite(pwm2, kiri);
        analogWrite(pwm3, 0);
        analogWrite(pwm4, kanan);
        delay(100); 
    }
};

void loop() {
  if (Serial.available() > 0)
  {
    int data;
    i = Serial.readString().toInt();
    if (i == 1){
      isStop = 1;
    } else {
      if (i == 0 && isStop == 0){
        setSpeed(50);
      }     
    }
    if (isStop == 1){
      setSpeed(0);
    };
    
    delay(100);

    Serial.println(String(kiri) + " - " + String(kanan));
  }
}