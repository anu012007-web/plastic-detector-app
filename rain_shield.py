# rain_shield.py
"""
EcoScanIndia - Automatic Rain Shield Control Module
Handles automatic rain detection, manual overrides, status indication, and webhooks
on a Raspberry Pi 4.
"""

import time
import threading
import requests

# Try to import RPi.GPIO; if not available (e.g. running on local PC for testing),
# use a mock GPIO implementation to allow imports and simulations to succeed.
try:
    import RPi.GPIO as GPIO
except ImportError:
    class MockPWM:
        def __init__(self, pin, freq):
            self.pin = pin
            self.freq = freq
        def start(self, dc):
            print(f"[MOCK PWM] Pin {self.pin} started at {self.freq}Hz with {dc}% duty cycle")
        def ChangeDutyCycle(self, dc):
            pass
        def stop(self):
            print(f"[MOCK PWM] Pin {self.pin} stopped")

    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        IN = "IN"
        PUD_UP = "PUD_UP"
        FALLING = "FALLING"
        RISING = "RISING"
        BOTH = "BOTH"
        HIGH = 1
        LOW = 0
        
        def __init__(self):
            print("[WARNING] RPi.GPIO not found. Using Mock GPIO simulation.")
            self._pin_states = {}

        def setmode(self, mode):
            print(f"[MOCK GPIO] Pin mode set to {mode}")

        def setwarnings(self, flag):
            pass

        def setup(self, pin, direction, pull_up_down=None):
            print(f"[MOCK GPIO] Pin {pin} configured as {direction} (pull: {pull_up_down})")
            self._pin_states[pin] = 0

        def output(self, pin, state):
            self._pin_states[pin] = state
            print(f"[MOCK GPIO] Pin {pin} output set to {state}")

        def input(self, pin):
            # For rain sensor, simulation script can change this
            state = self._pin_states.get(pin, 0)
            return state

        def add_event_detect(self, pin, edge, callback, bouncetime=0):
            print(f"[MOCK GPIO] Interrupt registered on Pin {pin} (edge: {edge}, debounce: {bouncetime}ms)")

        def cleanup(self):
            print("[MOCK GPIO] Cleanup completed")

        def PWM(self, pin, freq):
            return MockPWM(pin, freq)

    GPIO = MockGPIO()


# Constants
SHIELD_RETRACTED = 0
SHIELD_DEPLOYED = 90

# Pin Definitions (Default config, can be overridden)
DEFAULT_RAIN_SENSOR_PIN = 17
DEFAULT_SERVO_PIN = 18
DEFAULT_LED_RED_PIN = 27
DEFAULT_LED_GREEN_PIN = 22
DEFAULT_BUTTON_PIN = 23


class ServoFailureException(Exception):
    """Raised when the servo fails to verify target position after retries."""
    pass


class RainShieldController:
    def __init__(self, 
                 rain_pin=DEFAULT_RAIN_SENSOR_PIN, 
                 servo_pin=DEFAULT_SERVO_PIN, 
                 led_red_pin=DEFAULT_LED_RED_PIN, 
                 led_green_pin=DEFAULT_LED_GREEN_PIN, 
                 button_pin=DEFAULT_BUTTON_PIN,
                 limit_retracted_pin=None,
                 limit_deployed_pin=None,
                 webhook_url=None,
                 active_high=True):
        """
        Initializes the RainShieldController with custom pin configuration.
        
        :param active_high: Set True if LM393 DO outputs HIGH when rain is detected (default). 
                            Set False if LM393 DO outputs LOW on rain (active-low hardware default).
        """
        self.rain_pin = rain_pin
        self.servo_pin = servo_pin
        self.led_red_pin = led_red_pin
        self.led_green_pin = led_green_pin
        self.button_pin = button_pin
        self.limit_retracted = limit_retracted_pin
        self.limit_deployed = limit_deployed_pin
        self.webhook_url = webhook_url
        self.active_high = active_high

        # State Machine Variables
        # States: "AUTO_RETRACTED", "AUTO_DEPLOYED", "OVERRIDE_RETRACTED", "OVERRIDE_DEPLOYED"
        self.state = "AUTO_RETRACTED"
        self.current_angle = SHIELD_RETRACTED
        self.override_end_time = None
        self.consecutive_dry_ticks = 0
        self.is_docked = False
        self.servo_error = False

        # Thread Safety & Control
        self.lock = threading.Lock()
        self.running = False
        self.monitor_thread = None
        self.blink_thread = None
        
        # Debouncing for Button
        self.last_button_press_time = 0

        # Integration Callbacks
        self.on_rain_start_callback = None
        self.on_rain_stop_callback = None
        self.on_dock_entered_callback = None

        # PWM setup
        self.pwm = None

    def setup_gpio(self):
        """Initializes and configures the GPIO pins."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Set up LEDs as outputs
        GPIO.setup(self.led_red_pin, GPIO.OUT)
        GPIO.setup(self.led_green_pin, GPIO.OUT)
        self._update_leds()

        # Set up Rain Sensor DO
        GPIO.setup(self.rain_pin, GPIO.IN)

        # Set up limit switches if configured
        if self.limit_retracted is not None:
            GPIO.setup(self.limit_retracted, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        if self.limit_deployed is not None:
            GPIO.setup(self.limit_deployed, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # Set up Servo Pin
        GPIO.setup(self.servo_pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.servo_pin, 50)  # 50 Hz PWM frequency
        self.pwm.start(0)  # Detached by default

        # Set up manual override button
        GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        # Add interrupt with hardware/software debouncing (300ms)
        GPIO.add_event_detect(self.button_pin, GPIO.FALLING, callback=self._button_isr, bouncetime=300)

    # Callback Registration
    def register_on_rain_start(self, callback):
        """Registers a callback to execute when rain starts detecting."""
        self.on_rain_start_callback = callback

    def register_on_rain_stop(self, callback):
        """Registers a callback to execute when rain stops."""
        self.on_rain_stop_callback = callback

    def register_on_dock_entered(self, callback):
        """Registers a callback for when the robot docks successfully."""
        self.on_dock_entered_callback = callback

    # External Trigger for Dock
    def on_dock_entered(self):
        """Call this from main robot code when the robot arrives at the dock."""
        with self.lock:
            self.is_docked = True
            print("[DOCK] Docked! Entering low-power mode (wheels stopped).")
            if self.on_dock_entered_callback:
                # Kicks off callback in a separate thread to prevent blocking
                threading.Thread(target=self.on_dock_entered_callback).start()
            self._send_status_webhook()

    # Core Action Methods
    def deploy_shield(self):
        """Deploys the shield from 0 to 90 degrees smoothly over 1 second."""
        print("[SERVO] Deploying shield...")
        self._move_servo_smoothly(SHIELD_DEPLOYED)
        self.current_angle = SHIELD_DEPLOYED

    def retract_shield(self):
        """Retracts the shield from 90 to 0 degrees smoothly over 1 second."""
        print("[SERVO] Retracting shield...")
        self._move_servo_smoothly(SHIELD_RETRACTED)
        self.current_angle = SHIELD_RETRACTED

    # Manual Override ISR
    def _button_isr(self, pin):
        # Additional software debounce check
        now = time.time()
        if now - self.last_button_press_time < 0.3:
            return
        self.last_button_press_time = now

        # Run toggle logic inside lock
        with self.lock:
            if self.servo_error:
                print("[WARNING] SERVO ERROR state is active. Button press ignored.")
                return
            
            print("[OVERRIDE] Manual override button pressed!")
            # Toggle state and trigger movement
            if self.state in ["AUTO_RETRACTED", "OVERRIDE_RETRACTED"]:
                self.deploy_shield()
                self.state = "OVERRIDE_DEPLOYED"
                self.override_end_time = time.time() + 600  # 10 minutes from now
                print(f"[OVERRIDE] Manual override: SHIELD DEPLOYED. Auto-mode disabled until {time.strftime('%H:%M:%S', time.localtime(self.override_end_time))}.")
            else:
                self.retract_shield()
                self.state = "OVERRIDE_RETRACTED"
                self.override_end_time = time.time() + 600  # 10 minutes from now
                print(f"[OVERRIDE] Manual override: SHIELD RETRACTED. Auto-mode disabled until {time.strftime('%H:%M:%S', time.localtime(self.override_end_time))}.")
            
            self._update_leds()
            self._send_status_webhook()

    # State Machine Loop
    def start_monitoring(self):
        """Starts background thread to monitor the rain sensor and check override timers."""
        if self.running:
            return
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, name="RainMonitor")
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("[START] Rain shield monitoring service started.")

    def stop_monitoring(self):
        """Stops background threads and cleans up GPIO."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        GPIO.cleanup()
        print("[STOP] Rain shield monitoring service stopped.")

    def _monitoring_loop(self):
        while self.running:
            try:
                with self.lock:
                    if self.servo_error:
                        # If servo is in error state, wait and check again.
                        # LED Blinking is handled by separate blink thread.
                        pass
                    else:
                        self._process_state()
            except Exception as e:
                print(f"[ERROR] Error in monitoring loop: {e}")
            time.sleep(5.0)

    def _process_state(self):
        # 1. Check if Manual Override has expired
        if self.override_end_time is not None:
            if time.time() > self.override_end_time:
                print("[OVERRIDE] Manual override period (10 minutes) expired. Resuming automatic mode.")
                self.override_end_time = None
                
                # Read Rain Sensor Digital State immediately on expiration
                sensor_val = GPIO.input(self.rain_pin)
                is_raining = (sensor_val == GPIO.HIGH) if self.active_high else (sensor_val == GPIO.LOW)
                
                if is_raining:
                    if self.current_angle == SHIELD_RETRACTED:
                        self.deploy_shield()
                    self.state = "AUTO_DEPLOYED"
                else:
                    if self.current_angle == SHIELD_DEPLOYED:
                        self.retract_shield()
                    self.state = "AUTO_RETRACTED"
                    
                self.consecutive_dry_ticks = 0
                self._update_leds()
                self._send_status_webhook()
                return  # Skip processing for this tick since we just aligned the state

        # Read Rain Sensor Digital State (for normal monitoring ticks)
        sensor_val = GPIO.input(self.rain_pin)
        is_raining = (sensor_val == GPIO.HIGH) if self.active_high else (sensor_val == GPIO.LOW)


        # 2. Process Auto States
        if self.state == "AUTO_RETRACTED":
            if is_raining:
                print("[RAIN] RAIN DETECTED! Shield deployed.")
                self.deploy_shield()
                self.state = "AUTO_DEPLOYED"
                self.consecutive_dry_ticks = 0
                self._update_leds()
                self._send_status_webhook()

                # Trigger robot to dock
                print("[DOCK] Returning to charging dock...")
                if self.on_rain_start_callback:
                    threading.Thread(target=self.on_rain_start_callback).start()

        elif self.state == "AUTO_DEPLOYED":
            if not is_raining:
                self.consecutive_dry_ticks += 1
                remaining_sec = 30 - (self.consecutive_dry_ticks * 5)
                if remaining_sec > 0:
                    print(f"[WAIT] Rain stopped. Checking stability: Retracting in {remaining_sec}s...")
                else:
                    print("[DRY] RAIN STOPPED. Shield retracted.")
                    self.retract_shield()
                    self.state = "AUTO_RETRACTED"
                    self.consecutive_dry_ticks = 0
                    self.is_docked = False  # Exit docked state
                    self._update_leds()
                    self._send_status_webhook()

                    # Trigger robot to resume patrol
                    if self.on_rain_stop_callback:
                        threading.Thread(target=self.on_rain_stop_callback).start()
            else:
                # Reset consecutive dry timer if rain continues
                self.consecutive_dry_ticks = 0

        # 3. For Override States, we just log status updates periodically without moving servo
        elif self.state in ["OVERRIDE_RETRACTED", "OVERRIDE_DEPLOYED"]:
            # Just verify sensor value but do not act.
            pass

    # Servo Helper Methods
    def _move_servo_smoothly(self, target_angle):
        """
        Sweeps the servo angle from current_angle to target_angle over 1 second.
        Includes retries and failure notification rules.
        """
        retries = 3
        success = False

        for attempt in range(1, retries + 1):
            try:
                start_angle = self.current_angle
                start_dc = 2.5 + (start_angle / 180.0) * 10.0
                target_dc = 2.5 + (target_angle / 180.0) * 10.0
                
                steps = 20
                delay = 1.0 / steps

                # Start PWM pulses
                self.pwm.ChangeDutyCycle(start_dc)
                time.sleep(0.1)

                # Gradual sweep
                for i in range(1, steps + 1):
                    dc = start_dc + (target_dc - start_dc) * (i / float(steps))
                    self.pwm.ChangeDutyCycle(dc)
                    time.sleep(delay)

                # Detach pulse to avoid jitter & save power
                self.pwm.ChangeDutyCycle(0)
                
                # Verify Position
                if self.verify_servo_position(target_angle):
                    success = True
                    break
                else:
                    print(f"[WARNING] Servo mechanical check failed. Attempt {attempt}/{retries}...")
            except Exception as e:
                print(f"[WARNING] Error moving servo: {e}. Attempt {attempt}/{retries}...")

        if not success:
            self._handle_servo_failure()

    def verify_servo_position(self, target_angle):
        """
        Hardware position verification logic.
        Uses limit switches if available, otherwise simulates verification check.
        """
        if target_angle == SHIELD_RETRACTED and self.limit_retracted is not None:
            # Active-low limit switches (closed connects GPIO to GND)
            return GPIO.input(self.limit_retracted) == GPIO.LOW
        
        if target_angle == SHIELD_DEPLOYED and self.limit_deployed is not None:
            return GPIO.input(self.limit_deployed) == GPIO.LOW

        # Default simulated return
        return True

    def _handle_servo_failure(self):
        """Triggers error blinks and raises exception to halt normal operations."""
        self.servo_error = True
        print("[ERROR] SERVO ERROR! Manual intervention required.")
        
        # Start background LED error blinking (both LEDs blinking)
        if not self.blink_thread or not self.blink_thread.is_alive():
            self.blink_thread = threading.Thread(target=self._led_error_blinker, name="LEDErrorBlink")
            self.blink_thread.daemon = True
            self.blink_thread.start()
            
        raise ServoFailureException("Servo failed to reach target angle after 3 attempts.")

    # Indication Helpers
    def _update_leds(self):
        """Updates indicator LEDs based on state (Red for deployed, Green for retracted)."""
        if self.servo_error:
            return # Blinking thread takes control

        if self.state in ["AUTO_DEPLOYED", "OVERRIDE_DEPLOYED"]:
            GPIO.output(self.led_red_pin, GPIO.HIGH)
            GPIO.output(self.led_green_pin, GPIO.LOW)
        else:
            GPIO.output(self.led_red_pin, GPIO.LOW)
            GPIO.output(self.led_green_pin, GPIO.HIGH)

    def _led_error_blinker(self):
        """Blinks both LEDs in sync to alert of hardware servo issue."""
        while self.servo_error and self.running:
            GPIO.output(self.led_red_pin, GPIO.HIGH)
            GPIO.output(self.led_green_pin, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(self.led_red_pin, GPIO.LOW)
            GPIO.output(self.led_green_pin, GPIO.LOW)
            time.sleep(0.5)

    # Webhook Dispatch
    def _send_status_webhook(self):
        """Helper to invoke the webhook posting. Retained for backwards compatibility."""
        self.send_status_to_webhook()

    def send_status_to_webhook(self):
        """Compiles the robot status fields and sends them in a background thread."""
        if not self.webhook_url:
            return
        
        # Read BCM Pin state for rain sensor
        sensor_val = GPIO.input(self.rain_pin)
        is_sensor_wet = (sensor_val == GPIO.HIGH) if self.active_high else (sensor_val == GPIO.LOW)
        
        battery = int(self.get_battery_percentage())
        shield_position = "DEPLOYED" if self.current_angle == SHIELD_DEPLOYED else "RETRACTED"
        rain_sensor = "WET" if is_sensor_wet else "DRY"
        is_raining = is_sensor_wet  # True if rain is physically hitting the sensor
        is_docked = self.is_docked
        mode = "MANUAL_OVERRIDE" if self.state in ["OVERRIDE_DEPLOYED", "OVERRIDE_RETRACTED"] else "AUTO"
        
        temperature = self.get_temperature()
        humidity = self.get_humidity()

        payload = {
            "battery": battery,
            "shield_position": shield_position,
            "rain_sensor": rain_sensor,
            "is_raining": is_raining,
            "is_docked": is_docked,
            "mode": mode,
            "temperature": temperature,
            "humidity": humidity
        }

        # Fire and forget status thread
        t = threading.Thread(
            target=self._post_webhook_thread, 
            args=(payload,),
            name="StatusWebhookSender"
        )
        t.daemon = True
        t.start()

    def _post_webhook_thread(self, payload):
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=3.0)
            if response.status_code == 200:
                print(f"[WEBHOOK] Status update sent: {payload}")
            else:
                print(f"[WARNING] Webhook returned status code {response.status_code}")
        except Exception as e:
            # Catch network errors silently on real-time loop, print only in test
            pass

    def get_battery_percentage(self):
        """
        Mock battery sensor check.
        Can be integrated with specific ADC or I2C sensor commands.
        """
        return 88  # Return integer percent

    def get_temperature(self):
        """
        Mock temperature sensor check.
        On Pi, this could read /sys/class/thermal/thermal_zone0/temp or a DHT22.
        """
        return 28.5

    def get_humidity(self):
        """
        Mock humidity sensor check.
        Could read DHT22.
        """
        return 45.0


# Command line testing utility
if __name__ == "__main__":
    print("Starting hardware simulation test...")
    controller = RainShieldController(webhook_url="http://localhost:8000/api/robot/status")
    controller.setup_gpio()
    controller.start_monitoring()

    try:
        # Simulate button presses and state transitions in console
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        controller.stop_monitoring()
        print("Cleaned up and exited.")

