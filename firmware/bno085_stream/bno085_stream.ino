// BNO085 -> USB serial streamer for racer_imu (racer_imu/bno_driver.py parses this).
//
// Library: "Adafruit BNO08x" (Arduino Library Manager; pulls in Adafruit BusIO + Unified Sensor).
// Boards:  ESP32 / ESP32-S3, RP2040 (Pico), SAMD, Teensy. An Uno works only at low rates (RAM + 115200 baud).
//
// Output, one line per accelerometer report:
//   $B,<t_us>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>,<qw>,<qx>,<qy>,<qz>,<status>*HH\r\n
//   t_us  : sensor-hub timestamp of the accel report, lower 32 bits, microseconds
//   a     : SH2_ACCELEROMETER            [m/s^2], includes gravity
//   g     : SH2_GYROSCOPE_CALIBRATED     [rad/s]
//   q     : SH2_GAME_ROTATION_VECTOR     (w x y z), no magnetometer: motors don't disturb it
//   status: accel accuracy 0..3
//   HH    : XOR of every character between '$' and '*', two hex digits
// Lines starting with '#' are info/comments and are ignored by the host.
//
// Wiring (I2C, Adafruit breakout): VIN 3.3V, GND, SDA, SCL. Address 0x4A (0x4B if DI pulled high).
// For 400 Hz and cleanest timing prefer SPI (define USE_SPI and wire CS/INT/RST).

#include <Adafruit_BNO08x.h>

// ------------------------------------------------------------------ settings
#define BAUD            921600     // ignored by native-USB boards (ESP32-S3/RP2040/SAMD); matters for CP210x/CH340
#define REPORT_US       5000       // 5000 us = 200 Hz; 2500 = 400 Hz (use SPI for that)
// #define USE_SPI
#define BNO_CS          10
#define BNO_INT         9
#define BNO_RESET       -1         // pin wired to RST, or -1

Adafruit_BNO08x bno(BNO_RESET);
sh2_SensorValue_t v;

float gyr[3] = {NAN, NAN, NAN};
float quat[4] = {NAN, NAN, NAN, NAN};

// ------------------------------------------------------------------ tiny formatter (no printf float on AVR)
static char line[200];
static int len;

static void addChar(char c) { if (len < (int)sizeof(line) - 1) line[len++] = c; }
static void addStr(const char *s) { while (*s) addChar(*s++); }
static void addUInt(uint32_t x) {
  char t[11]; int n = 0;
  do { t[n++] = '0' + x % 10; x /= 10; } while (x);
  while (n) addChar(t[--n]);
}
static void addFloat(float x, uint8_t dec) {
  if (x != x) { addStr("nan"); return; }   // NaN (portable isnan)
  if (x < 0) { addChar('-'); x = -x; }
  uint32_t scale = 1;
  for (uint8_t i = 0; i < dec; i++) scale *= 10;
  uint32_t whole = (uint32_t)x;
  uint32_t frac = (uint32_t)((x - whole) * scale + 0.5f);
  if (frac >= scale) { whole++; frac -= scale; }
  addUInt(whole);
  addChar('.');
  char t[10];
  for (int i = dec - 1; i >= 0; i--) { t[i] = '0' + frac % 10; frac /= 10; }
  for (uint8_t i = 0; i < dec; i++) addChar(t[i]);
}

static void enableReports() {
  if (!bno.enableReport(SH2_ACCELEROMETER, REPORT_US)) Serial.println("# could not enable accelerometer");
  if (!bno.enableReport(SH2_GYROSCOPE_CALIBRATED, REPORT_US)) Serial.println("# could not enable gyro");
  if (!bno.enableReport(SH2_GAME_ROTATION_VECTOR, REPORT_US)) Serial.println("# could not enable game rotation vector");
}

void setup() {
  Serial.begin(BAUD);
  while (!Serial && millis() < 3000) delay(10);
#ifdef USE_SPI
  bool ok = bno.begin_SPI(BNO_CS, BNO_INT);
#else
  Wire.begin();
  Wire.setClock(400000);
  bool ok = bno.begin_I2C();
#endif
  if (!ok) {
    while (1) { Serial.println("# BNO08x not found, check wiring"); delay(1000); }
  }
  Serial.println("# bno085_stream ready");
  enableReports();
}

void loop() {
  if (bno.wasReset()) {
    Serial.println("# sensor was reset, re-enabling reports");
    enableReports();
  }
  if (!bno.getSensorEvent(&v)) return;

  switch (v.sensorId) {
    case SH2_GYROSCOPE_CALIBRATED:
      gyr[0] = v.un.gyroscope.x; gyr[1] = v.un.gyroscope.y; gyr[2] = v.un.gyroscope.z;
      break;
    case SH2_GAME_ROTATION_VECTOR:
      quat[0] = v.un.gameRotationVector.real; quat[1] = v.un.gameRotationVector.i;
      quat[2] = v.un.gameRotationVector.j;    quat[3] = v.un.gameRotationVector.k;
      break;
    case SH2_ACCELEROMETER: {
      // one output line per accel report, carrying the most recent gyro + quaternion
      len = 0;
      addStr("$B,");
      addUInt((uint32_t)v.timestamp);  addChar(',');
      addFloat(v.un.accelerometer.x, 4); addChar(',');
      addFloat(v.un.accelerometer.y, 4); addChar(',');
      addFloat(v.un.accelerometer.z, 4); addChar(',');
      for (int i = 0; i < 3; i++) { addFloat(gyr[i], 5); addChar(','); }
      for (int i = 0; i < 4; i++) { addFloat(quat[i], 5); addChar(','); }
      addUInt(v.status & 0x03);
      uint8_t cs = 0;
      for (int i = 1; i < len; i++) cs ^= (uint8_t)line[i];
      addChar('*');
      const char *hex = "0123456789ABCDEF";
      addChar(hex[cs >> 4]); addChar(hex[cs & 0x0F]);
      addStr("\r\n");
      Serial.write((const uint8_t *)line, len);
      break;
    }
  }
}
