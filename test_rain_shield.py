# test_rain_shield.py
"""
EcoScanIndia - Rain Shield Simulation Test Script
Simulates rain sensor inputs, button clicks, and servo actions to verify the
logic, state machine, and webhook reporting.
"""

import time
import unittest
from rain_shield import RainShieldController, GPIO, SHIELD_DEPLOYED, SHIELD_RETRACTED

class TestRainShieldStateMachine(unittest.TestCase):
    def setUp(self):
        # Initialize controller with a mock webhook URL
        self.controller = RainShieldController(
            rain_pin=17,
            servo_pin=18,
            led_red_pin=27,
            led_green_pin=22,
            button_pin=23,
            webhook_url="http://localhost:8000/api/robot/status",
            active_high=True
        )
        self.controller.setup_gpio()

        # Track callbacks
        self.rain_started_triggered = False
        self.rain_stopped_triggered = False
        self.docked_triggered = False

        self.controller.register_on_rain_start(self.on_rain_start)
        self.controller.register_on_rain_stop(self.on_rain_stop)
        self.controller.register_on_dock_entered(self.on_dock_entered)

    def on_rain_start(self):
        self.rain_started_triggered = True
        print("[TEST CALLBACK] Robot notified: Rain started! Navigation to dock initiated.")

    def on_rain_stop(self):
        self.rain_stopped_triggered = True
        print("[TEST CALLBACK] Robot notified: Rain stopped! Resuming patrol.")

    def on_dock_entered(self):
        self.docked_triggered = True
        print("[TEST CALLBACK] Robot notified: Dock reached. Entering low power.")

    def test_automatic_rain_cycle(self):
        print("\n--- TEST: Automatic Rain Cycle ---")
        self.assertEqual(self.controller.state, "AUTO_RETRACTED")
        self.assertEqual(self.controller.current_angle, SHIELD_RETRACTED)

        # 1. Simulate rain (Sensor goes HIGH)
        print("Simulating: Sensor DO goes HIGH (Rain detected)")
        GPIO._pin_states[self.controller.rain_pin] = GPIO.HIGH
        
        # Run state processing
        self.controller._process_state()
        
        self.assertEqual(self.controller.state, "AUTO_DEPLOYED")
        self.assertEqual(self.controller.current_angle, SHIELD_DEPLOYED)
        self.assertTrue(self.rain_started_triggered)

        # 2. Simulate rain stops (Sensor goes LOW)
        print("Simulating: Sensor DO goes LOW (Rain stopped)")
        GPIO._pin_states[self.controller.rain_pin] = GPIO.LOW
        
        # Process multiple times (need 6 ticks of 5s = 30s)
        for tick in range(1, 7):
            self.controller._process_state()
            if tick < 6:
                self.assertEqual(self.controller.state, "AUTO_DEPLOYED")
                self.assertFalse(self.rain_stopped_triggered)
            else:
                self.assertEqual(self.controller.state, "AUTO_RETRACTED")
                self.assertEqual(self.controller.current_angle, SHIELD_RETRACTED)
                self.assertTrue(self.rain_stopped_triggered)

    def test_manual_override(self):
        print("\n--- TEST: Manual Override ---")
        self.assertEqual(self.controller.state, "AUTO_RETRACTED")

        # Simulate button press interrupt
        print("Simulating: Manual Button Pressed while Retracted")
        self.controller._button_isr(self.controller.button_pin)

        self.assertEqual(self.controller.state, "OVERRIDE_DEPLOYED")
        self.assertEqual(self.controller.current_angle, SHIELD_DEPLOYED)
        self.assertIsNotNone(self.controller.override_end_time)

        # Confirm rain sensor is ignored in override state
        print("Simulating rain change during override (Sensor goes LOW, but should remain deployed)")
        GPIO._pin_states[self.controller.rain_pin] = GPIO.LOW
        self.controller._process_state()
        self.assertEqual(self.controller.state, "OVERRIDE_DEPLOYED")

        # Simulate override timeout (forwarding time)
        print("Simulating: Override timer expiration (10 minutes pass)")
        self.controller.override_end_time = time.time() - 1  # backdate expiration
        
        # Next tick should return state to automatic based on dry sensor
        self.controller._process_state()
        self.assertEqual(self.controller.state, "AUTO_RETRACTED")
        self.assertEqual(self.controller.current_angle, SHIELD_RETRACTED)

    def test_docking_state(self):
        print("\n--- TEST: Docking Behavior ---")
        
        # Dock entered
        self.controller.on_dock_entered()
        time.sleep(0.1)  # Allow threaded callback to fire
        self.assertTrue(self.controller.is_docked)
        self.assertTrue(self.docked_triggered)


if __name__ == "__main__":
    unittest.main()
