/*
  EdgeGuard AI - Phase 1 IoT Simulation
  ESP32 + Wokwi virtual sensors -> MQTT (HiveMQ public broker)

  Sensors simulated:
    - Temperature   (DS18B20)            -> engine block / coolant
    - Vibration     (MPU6050 accel mag)  -> wheel hub / axle bearing
    - Oil Pressure  (potentiometer, 0-10 bar)
    - Hydraulic Pressure (potentiometer, 0-600 bar)
    - Suspension Pressure (potentiometer, 0-16 bar)
    - Battery Voltage (potentiometer, 0-25V)

  Failure scenario: AUTO-RAMP starting at T+60s.
  Simulates the documented degradation cascade:
    Temperature UP -> Vibration UP -> Oil Pressure DOWN -> failure window

  Topic schema (truck_id = "truck1" for hackathon demo):
    edgeguard/truck1/temperature
    edgeguard/truck1/vibration
    edgeguard/truck1/oil_pressure
    edgeguard/truck1/hydraulic_pressure
    edgeguard/truck1/suspension_pressure
    edgeguard/truck1/battery_voltage
    edgeguard/truck1/status        (heartbeat / online flag)

  JSON payload shape (same for every sensor topic):
    {
      "truck_id": "truck1",
      "sensor": "vibration",
      "value": 1.23,
      "unit": "g",
      "ts": 123456   // millis() since boot, swap for real epoch in Phase 2
    }
*/

#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ---------- WiFi (Wokwi simulated network) ----------
const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASS = "";

// ---------- MQTT (public HiveMQ broker) ----------
const char* MQTT_BROKER = "broker.hivemq.com";
const int   MQTT_PORT   = 1883;
const char* TRUCK_ID    = "truck1";
// Make this unique-ish so you don't collide with other teams testing on the
// same public broker during the hackathon.
const char* MQTT_CLIENT_ID = "edgeguard-truck1-sim-01";

WiFiClient espClient;
PubSubClient mqtt(espClient);

// ---------- Pins ----------
#define ONE_WIRE_PIN   4
#define OIL_PIN        34
#define HYD_PIN        35
#define SUSP_PIN       32
#define BATT_PIN       33
#define ALERT_LED_PIN  2

OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature tempSensor(&oneWire);
Adafruit_MPU6050 mpu;

// ---------- Timing ----------
unsigned long bootTime = 0;
unsigned long lastPublish = 0;
const unsigned long PUBLISH_INTERVAL_MS = 2000;   // publish every 2s
const unsigned long FAILURE_START_MS    = 60000;  // auto-ramp begins at T+60s
const unsigned long FAILURE_RAMP_MS     = 90000;  // ramps to full severity over 90s

void connectWiFi() {
  Serial.print("Connecting to WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println(" connected!");
  Serial.println(WiFi.localIP());
}

void connectMQTT() {
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  while (!mqtt.connected()) {
    Serial.print("Connecting to MQTT broker...");
    if (mqtt.connect(MQTT_CLIENT_ID)) {
      Serial.println(" connected!");
      String statusTopic = String("edgeguard/") + TRUCK_ID + "/status";
      mqtt.publish(statusTopic.c_str(), "{\"status\":\"online\"}");
    } else {
      Serial.print(" failed, rc=");
      Serial.print(mqtt.state());
      Serial.println(" retrying in 2s");
      delay(2000);
    }
  }
}

// Publishes one sensor reading as a JSON payload on its own topic.
void publishReading(const char* sensorName, float value, const char* unit) {
  String topic = String("edgeguard/") + TRUCK_ID + "/" + sensorName;

  String payload = "{";
  payload += "\"truck_id\":\"" + String(TRUCK_ID) + "\",";
  payload += "\"sensor\":\"" + String(sensorName) + "\",";
  payload += "\"value\":" + String(value, 2) + ",";
  payload += "\"unit\":\"" + String(unit) + "\",";
  payload += "\"ts\":" + String(millis());
  payload += "}";

  mqtt.publish(topic.c_str(), payload.c_str());
  Serial.print(topic);
  Serial.print(" -> ");
  Serial.println(payload);
}

// Returns 0.0 -> 1.0 representing how far into the failure ramp we are.
// 0 = healthy baseline, 1 = full failure severity reached.
float getFailureSeverity() {
  unsigned long elapsed = millis() - bootTime;
  if (elapsed < FAILURE_START_MS) return 0.0;
  float into = (float)(elapsed - FAILURE_START_MS);
  float severity = into / (float)FAILURE_RAMP_MS;
  if (severity > 1.0) severity = 1.0;
  return severity;
}

void setup() {
  Serial.begin(115200);
  pinMode(ALERT_LED_PIN, OUTPUT);
  digitalWrite(ALERT_LED_PIN, LOW);

  connectWiFi();
  connectMQTT();

  tempSensor.begin();

  if (!mpu.begin()) {
    Serial.println("MPU6050 not found - check wiring (continuing anyway)");
  } else {
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  }

  bootTime = millis();
  Serial.println("EdgeGuard AI simulation started. Auto-ramp failure at T+60s.");
}

void loop() {
  if (!mqtt.connected()) {
    connectMQTT();
  }
  mqtt.loop();

  unsigned long now = millis();
  if (now - lastPublish < PUBLISH_INTERVAL_MS) {
    return;
  }
  lastPublish = now;

  float severity = getFailureSeverity();

  // ---- Temperature: baseline ~65C, ramps up to ~110C under failure ----
  float baseTemp = 65.0;
  float temp = baseTemp + (severity * 45.0) + random(-10, 10) / 10.0;

  // ---- Vibration: read MPU6050 accel magnitude, add ramp + noise ----
  float vibBase = 0.3; // g, healthy baseline jitter
  float vib = vibBase + (severity * 2.2) + random(-5, 5) / 100.0;
  // If MPU is actually present, blend in a touch of real sensor noise
  if (mpu.getMotionInterruptStatus() || true) {
    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);
    float mag = sqrt(a.acceleration.x * a.acceleration.x +
                      a.acceleration.y * a.acceleration.y +
                      a.acceleration.z * a.acceleration.z) / 9.8;
    vib += (mag - 1.0) * 0.05; // small real-sensor influence, kept subtle
  }

  // ---- Oil pressure: baseline ~70 (of 0-10 bar scale *10), DROPS under failure ----
  int oilRaw = analogRead(OIL_PIN); // 0-4095
  float oilBase = map(oilRaw, 0, 4095, 30, 70) / 10.0; // 3.0 - 7.0 bar baseline
  float oilPressure = oilBase - (severity * 2.5) + random(-3, 3) / 100.0;
  if (oilPressure < 0.5) oilPressure = 0.5;

  // ---- Hydraulic pressure: baseline mid-range, dips/spikes under failure ----
  int hydRaw = analogRead(HYD_PIN);
  float hydBase = map(hydRaw, 0, 4095, 200, 400); // 200-400 bar baseline
  float hydPressure = hydBase - (severity * 90.0) + random(-5, 5);

  // ---- Suspension pressure: baseline steady, DROPS under failure (stress/wear) ----
  int suspRaw = analogRead(SUSP_PIN);
  float suspBase = map(suspRaw, 0, 4095, 60, 100) / 10.0; // 6.0-10.0 bar
  float suspPressure = suspBase - (severity * 3.0) + random(-2, 2) / 100.0;
  if (suspPressure < 1.0) suspPressure = 1.0;

  // ---- Battery voltage: mostly stable, slight dip under stress ----
  int battRaw = analogRead(BATT_PIN);
  float battVoltage = map(battRaw, 0, 4095, 220, 250) / 10.0; // 22.0-25.0V
  battVoltage -= (severity * 1.0);

  // ---- Publish all six readings ----
  publishReading("temperature", temp, "C");
  publishReading("vibration", vib, "g");
  publishReading("oil_pressure", oilPressure, "bar");
  publishReading("hydraulic_pressure", hydPressure, "bar");
  publishReading("suspension_pressure", suspPressure, "bar");
  publishReading("battery_voltage", battVoltage, "V");

  // ---- Local alert LED, mirrors what the dashboard should flag ----
  if (severity > 0.5) {
    digitalWrite(ALERT_LED_PIN, HIGH);
  } else {
    digitalWrite(ALERT_LED_PIN, LOW);
  }

  Serial.print("Failure severity: ");
  Serial.println(severity);
  Serial.println("---");
}
