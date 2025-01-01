from enum import Enum

class MouseAction(Enum):
    CLICK = "Click"
    DOUBLE_CLICK = "DoubleClick"
    RIGHT_CLICK = "RightClick"
    MOVE = "Move"

class KeyboardAction(Enum):
    TYPE = "Type"
    PRESS_KEY = "PressKey"
    HOTKEY = "Hotkey"

class GeneralAction(Enum):
    SCROLL = "Scroll"
    WAIT = "Wait"
    EXIT = "Exit"

class ScreenshotAction(Enum):
    CAPTURE_SCREEN = "CaptureScreen"
    CAPTURE_REGION = "CaptureRegion"
