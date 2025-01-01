from typing import Optional, List, Dict, Type, Union
from enum import Enum

# Assuming these are defined in your `pyautogui_actions` module
from pyautogui_actions import MouseAction, KeyboardAction, GeneralAction, ScreenshotAction

class WhenDoneType(Enum):
    CONTINUE = "Continue"
    END = "End"

class ChatGPTResponse:
    def __init__(
        self,
        immediate_task: str,
        action: Optional[Union[MouseAction, KeyboardAction, GeneralAction, ScreenshotAction]] = None,
        coordinates: Optional[List[int]] = None,
        text_to_type: Optional[str] = None,
        keys: Optional[List[str]] = None,
        scroll_amount: Optional[int] = None,
        when_done: str = WhenDoneType.CONTINUE.value,
        response_id: str = ""
    ):
        self.immediate_task = immediate_task
        self.action = action
        self.coordinates = coordinates
        self.text_to_type = text_to_type
        self.keys = keys
        self.scroll_amount = scroll_amount
        self.when_done = when_done
        self.response_id = response_id

    @classmethod
    def from_dict(cls, data: Dict) -> "ChatGPTResponse":
        """Creates a ChatGPTResponse object from a dictionary."""
        action_enum = cls._get_action_enum(data.get("Action"))
        return cls(
            immediate_task=data.get("ImmediateTaskInNaturalLanguage", ""),
            action=action_enum,
            coordinates=data.get("Coordinates", []),
            text_to_type=data.get("WhatToType", ""),
            keys=data.get("Keys", []),
            scroll_amount=data.get("ScrollAmount", 0),
            when_done=data.get("WhenDone", WhenDoneType.CONTINUE.value),
            response_id=data.get("Id", "")
        )

    def to_dict(self) -> Dict:
        """Converts the ChatGPTResponse object to a dictionary for serialization."""
        return {
            "ImmediateTaskInNaturalLanguage": self.immediate_task,
            "Action": self.action.name if self.action else None,
            "Coordinates": self.coordinates,
            "WhatToType": self.text_to_type,
            "Keys": self.keys,
            "ScrollAmount": self.scroll_amount,
            "WhenDone": self.when_done,
            "Id": self.response_id
        }

    @staticmethod
    def _get_action_enum(action_name):
        """Maps a string to the appropriate action enum."""
        all_actions = [MouseAction, KeyboardAction, GeneralAction, ScreenshotAction]

        print(f"[DEBUG] Matching action: '{action_name}'")

        # Convert the action name to uppercase for case-insensitive matching
        action_name_upper = action_name.upper()

        for enum_class in all_actions:
            # Match the action name against the uppercase version of the members
            for member_name, member in enum_class.__members__.items():
                if member_name.upper() == action_name_upper:
                    print(f"[DEBUG] Found matching action: {member}")
                    return member

        print(f"[ERROR] No matching enum found for action: {action_name}")
        return None
