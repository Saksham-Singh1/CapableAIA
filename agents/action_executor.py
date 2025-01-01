import pyautogui
from pyautogui_actions import MouseAction, KeyboardAction, GeneralAction, ScreenshotAction

# Add a delay between all PyAutoGUI actions (e.g., 0.5 seconds)
pyautogui.PAUSE = 0.5  # Adjust as needed to slow down the actions

class ActionExecutor:
    """Executes actions based on ChatGPT instructions."""

    @staticmethod
    def execute(action, coordinates=None, text_to_type=None, keys=None, scroll_amount=None):
        """Execute the given action using pyautogui."""
        if isinstance(action, str):
            # Map string actions to enums if needed
            if action == "PressKey":
                action = KeyboardAction.PRESS_KEY
            elif action == "Click":
                action = MouseAction.CLICK
            elif action == "Type":
                action = KeyboardAction.TYPE
            # Add more mappings as needed

        if isinstance(action, MouseAction):
            ActionExecutor._handle_mouse_action(action, coordinates)
        elif isinstance(action, KeyboardAction):
            ActionExecutor._handle_keyboard_action(action, text_to_type, keys)
        elif action == GeneralAction.SCROLL:
            ActionExecutor._handle_scroll(scroll_amount)
        elif action == ScreenshotAction.CAPTURE_SCREEN:
            ActionExecutor._capture_screenshot()
        else:
            print(f"[ERROR] Unsupported action: {action}")



    @staticmethod
    def _handle_mouse_action(action, coordinates):
        """Handle mouse-related actions like click, move, etc."""
        if not coordinates:
            print("[ERROR] Coordinates are required for mouse actions.")
            return

        x, y = coordinates
        print(f"[INFO] Performing {action.name} at ({x}, {y})")

        # Move the mouse slowly to the target location before clicking
        pyautogui.moveTo(x, y, duration=1)  # 1 second to move to target
        if action == MouseAction.CLICK:
            pyautogui.click()
        elif action == MouseAction.DOUBLE_CLICK:
            pyautogui.doubleClick()
        elif action == MouseAction.RIGHT_CLICK:
            pyautogui.rightClick()


    @staticmethod
    def _handle_keyboard_action(action, text_to_type, keys):
        """Handle keyboard-related actions like typing or pressing keys."""
        if action == KeyboardAction.TYPE and text_to_type:
            print(f"[INFO] Typing: {text_to_type}")
            pyautogui.write(text_to_type)
        elif action == KeyboardAction.PRESS_KEY and keys:
            print(f"[INFO] Pressing keys: {keys}")
            for key in keys:
                if key.lower() == "enter":
                    pyautogui.press("enter")  # Special handling for "Enter"
                else:
                    pyautogui.press(key)
        elif action == KeyboardAction.HOTKEY and keys:
            print(f"[INFO] Pressing hotkey: {keys}")
            pyautogui.hotkey(*keys)


    @staticmethod
    def _handle_scroll(scroll_amount):
        """Handle scrolling action."""
        print(f"[INFO] Scrolling by {scroll_amount}")
        pyautogui.scroll(scroll_amount)

    @staticmethod
    def _capture_screenshot():
        """Capture and save a screenshot."""
        screenshot_path = "captured_screenshot.png"
        print(f"[INFO] Capturing screenshot and saving as '{screenshot_path}'")
        screenshot = pyautogui.screenshot()
        screenshot.save(screenshot_path)
