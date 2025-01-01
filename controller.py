import json
import time
import base64
import io
import os
from PIL import ImageGrab
import pyautogui
import openai
from agents.action_executor import ActionExecutor
from services.image_service import ImageService
from response_model import ChatGPTResponse, WhenDoneType
from PIL import Image
import uuid
from services.matcher_service import MatcherService

# Set your OpenAI API key
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

class Controller:
    def __init__(self, scaling_factors):
        """Initialize the Controller with necessary services and scaling factors."""
        self.image_service = ImageService(cell_size=200, scaling_factors=scaling_factors)
        self.matcher_service = MatcherService()        
        self.output_dir = "output"  # Ensure you have this directory

    def optimize_instruction(self, instruction):
        """Optimize the initial user instruction using ChatGPT and provide necessary context."""

        # Capture and encode the screenshot
        screenshot_full = pyautogui.screenshot().convert("RGB")
        screenshot_resized = screenshot_full.resize((800, 600))
        buffered = io.BytesIO()
        screenshot_resized.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Call ChatGPT to optimize the instruction
        message_content = f"""
        The user has given the following instruction: "{instruction}". However, the instruction might be vague or incomplete. 
        We want the instruction to be as detailed as possible because it will be used by an AI agent to control a computer and achieve the goal.
        You are provided with a screenshot to give you context about the current state of the user's computer.
        
        Screenshot: data:image/jpeg;base64,{base64_image}
        
        Please suggest a more complete and specific version of the instruction, considering all the necessary details such as 
        specifying fields, actions, subjects, or any additional information required to perform the task efficiently.
        The task will be later broken down by an AI agent so it knows where to click and do actions.
        Do not use placeholders, and feel free to make up any content that is missing from the original instruction, while maintaining the end goal.
        Just give the instruction directly as if you were giving it to the AI agent right away. there are no intermediaries between you and him, and the user who submitted the request is unreachable until the task is completely done, so this is a communication between two AI bots.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": message_content}
            ]
        )

        try:
            # Extract the optimized instruction from the response
            optimized_instruction = response.choices[0].message["content"].strip()
            print("[INFO] Optimized Instruction:", optimized_instruction)
            return optimized_instruction
        except (KeyError, IndexError) as e:
            print(f"[ERROR] Failed to get optimized instruction: {e}")
            return instruction  # Fallback to the original instruction

    def run(self, endGoalInstruction, context=None):
        """Main flow of the application with iterative AI loop."""
        if context is None:
            context = {
                "completed_steps": [],
                "last_elements": [],
                "mouse_position": None
            }

        verifier_feedback = ""  # Initialize verifier feedback as empty

        while True:
            # Step 1: Extract visible text from the browser
            visible_browser_text = ""

            # Step 2: Capture the current mouse position
            mouse_x, mouse_y = pyautogui.position()
            context["mouse_position"] = (mouse_x, mouse_y)

            # Step 3: Capture and encode the screenshot
            screenshot_full = pyautogui.screenshot().convert("RGB")
            screenshot_resized = screenshot_full.resize((800, 600))
            buffered = io.BytesIO()
            screenshot_resized.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")


            # Step 4: Save the screenshot to a temporary file and extract elements
            temp_image_path = "temp_screenshot.jpg"
            screenshot_full.save(temp_image_path)
            elements = self.extract_elements_with_adjusted_coordinates(temp_image_path)

            # Step 5: Send extracted elements, context, text, and goal instruction to ChatGPT
            response = self.call_chatgpt_with_elements(
                elements, endGoalInstruction, context, base64_image, visible_browser_text, verifier_feedback
            )

            # Step 6: Handle ChatGPT's response
            if response:
                actions = self.handle_response(response, context)
            else:
                print("[ERROR] Failed to process the ChatGPT response.")
                break

            # Step 7: Execute actions
            if actions:
                self.execute_actions(actions, context)

                # Check if any action has WhenDone == "End"
                should_end = any(action.when_done == WhenDoneType.END.value for action in actions)
                if should_end:
                    print("[INFO] ChatGPT indicates to end the task. Exiting...")
                    break
            else:
                print("[ERROR] No actions to execute.")
                continue


            # Step 8: After executing actions, capture a new screenshot and call verifier AI
            screenshot_full_after_actions = pyautogui.screenshot().convert("RGB")
            screenshot_resized_after_actions = screenshot_full_after_actions.resize((800, 600))
            buffered_after_actions = io.BytesIO()
            screenshot_resized_after_actions.save(buffered_after_actions, format="JPEG")
            base64_image_after_actions = base64.b64encode(buffered_after_actions.getvalue()).decode("utf-8")

            verifier_feedback = self.call_verifierAI(actions, base64_image_after_actions)


    def extract_elements_with_adjusted_coordinates(self, image_path):
        """Extract elements and adjust center coordinates for ChatGPT, while keeping full bounding boxes for internal use."""
        # Step 1: Extract elements and bounding boxes from the full-resolution image
        elements = self.image_service.extract_text_from_elements(image_path)
        self.image_service.save_json_to_file(elements, prefix="extracted_elements")

        # Step 2: Adjust coordinates using scaling factors for ChatGPT
        adjusted_elements = []
        for element in elements:
            x, y, w, h = element["coordinates"]
            # Calculate the center of the bounding box
            center_x, center_y = self.image_service.get_center_of_box(x, y, w, h)
            # Adjust the center coordinates for ChatGPT
            adjusted_x, adjusted_y = self.image_service.adjust_coordinates_for_scaling(center_x, center_y)

            # Only add the adjusted center coordinates (where ChatGPT should click)
            adjusted_elements.append({
                "text": element["text"],
                "coordinates": [adjusted_x, adjusted_y]  # Only the center coordinates
            })

        # Step 3: Use browser text as a reference, but not as the main source
        browser_text = ""

        # Step 4: Match and combine texts with the adjusted coordinates
        matched_elements = self.matcher_service.match_texts_with_coordinates(browser_text, adjusted_elements)

        # Step 5: Save matched elements as JSON
        self.image_service.save_json_to_file(matched_elements, prefix="matched_elements")

        print("Matched Elements successfully")
        return matched_elements

    def call_chatgpt_with_elements(
        self, elements, instruction, context, base64_image, visible_browser_text, verifier_feedback
    ):
        """Send elements, context, goal, and image to ChatGPT."""
        message_content = f"""
        You are in full control of the user's computer. Your task is to assist in achieving the goal provided.

        **Goal:** "{instruction}"

        **Current State:**

        - **Mouse Position:** {context["mouse_position"]}
        - **Context (Completed Steps):** {json.dumps(context['completed_steps'], indent=4)}
        - **UI Elements:** {json.dumps(elements, indent=4)}
        - **Screenshot:** data:image/jpeg;base64,{base64_image}

        **Responsibilities:**

        - Think and act step-by-step, performing actions as a human would when physically using the computer.
        - Ensure each action leads toward completing the goal without errors.
        - Always verify actions using the screenshot and current mouse position.
        - Respond **only** in pure JSON format (see format below).
        """

        # Include verifier feedback if it's not empty
        if verifier_feedback:
            message_content += f"""
        The verifier AI has provided the following feedback on your previous actions:
        {verifier_feedback}

        Please consider this feedback when determining the next actions. It is very important that you consider this to know what to do next.
        Listen to him in what he says.
        """

        message_content += """
        **Guidelines:**

        1. **Step-by-Step Actions:**
        - Send **only one** action at a time.
        - Follow a logical sequence of actions without skipping steps.
        - Always interact with the correct elements before typing or pressing keys.

        2. **Avoid Duplicate Actions:**
        - Do not repeat actions already completed successfully (see Context).
        - Each action should progress toward the goal.

        3. **No Blind Clicks:**
        - Provide click actions **only** if you have precise coordinates.
        - Do not suggest clicks without exact coordinates.

        4. **Use the Screenshot as the Source of Truth:**
        - Analyze the screenshot carefully to determine the next action.
        - Verify if previous actions succeeded or need correction.

        5. **Response Format:**
        - Return actions in pure JSON format as specified below.
        - Do not include explanations or comments.

        6. **Completion Criteria:**
        - Use `"WhenDone": "Continue"` if more steps are needed.
        - Use `"WhenDone": "End"` only when the goal is fully achieved and verified via the screenshot.

        **JSON Format Example:**

        ```json
        [
            {
                "ImmediateTaskInNaturalLanguage": "Describe the next step",
                "Action": "Click" | "DoubleClick" | "RightClick" | "Move" | "Type" | "PressKey" | "Hotkey" | "Scroll" | "Wait" | "Exit" | "CaptureScreen" | "CaptureRegion",
                "Coordinates": [x, y],  // Required for click actions
                "WhatToType": "Text to type, if applicable",
                "Keys": ["key1", "key2"],  // For key presses, if applicable
                "WhenDone": "Continue" | "End",
                "Id": "response-12345"
            }
        ]

        """

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": message_content}
            ]
        )

        try:
            # Remove any backticks and strip out comments (lines with "//")
            content = response.choices[0].message["content"].strip("```json").strip("```")
            # Remove lines that contain comments
            content = "\n".join([line for line in content.splitlines() if "//" not in line])

            print("[INFO] ChatGPT Response Content:", content)
            # Parse the JSON content
            parsed_content = json.loads(content)

            # If parsed_content is a list, convert each item to a ChatGPTResponse; otherwise, convert the single object
            if isinstance(parsed_content, list):
                return [ChatGPTResponse.from_dict(action) for action in parsed_content]
            else:
                return [ChatGPTResponse.from_dict(parsed_content)]

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"[ERROR] Failed to parse ChatGPT response: {e}")
            return None


    def handle_response(self, chatgpt_response, context):
        """Handle ChatGPT's response, prepare actions to execute."""
        
        # Check if response is a list of actions or a single action
        actions = chatgpt_response if isinstance(chatgpt_response, list) else [chatgpt_response]
        valid_actions = []
        for action in actions:
            print(f"[INFO] Immediate Task: {action.immediate_task}")

            # Avoid redundant tasks by checking if the task is already completed
            if action.immediate_task in context["completed_steps"]:
                print("[INFO] Task already completed. Skipping...")
                continue

            # Add the immediate task to the completed steps
            context["completed_steps"].append(action.immediate_task)

            # Check if action is valid
            if action.action:
                valid_actions.append(action)
            else:
                print("[ERROR] No valid action found in the response.")
        
        return valid_actions

    def execute_actions(self, actions, context):
        """Execute the given actions."""
        for action in actions:
            # Execute the action
            if action.coordinates:
                x, y = action.coordinates
                print(f"[INFO] Moving mouse to ({x}, {y})...")
                pyautogui.moveTo(x, y, duration=1)

            try:
                ActionExecutor.execute(
                    action=action.action,
                    coordinates=action.coordinates,
                    text_to_type=action.text_to_type,
                    keys=action.keys,
                    scroll_amount=action.scroll_amount,
                )
                print("[INFO] Action executed successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to execute action: {e}")
                # Handle exception or decide whether to continue

    def call_verifierAI(self, previous_actions, base64_image):
        """Send the screenshot and previous actions to the verifier AI."""

        screenshot = ImageGrab.grab().convert("RGB")

        # Save screenshot as base64-encoded string
        buffered = io.BytesIO()
        screenshot.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Send base64-encoded image to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What’s in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        # Print the model's response
        print(response.choices[0]["message"]["content"])
        return(response.choices[0]["message"]["content"])



    def save_text_to_file(self, text, prefix="extracted_text"):
        """Save the extracted text to a file with a unique name in the output directory."""
        self.create_output_dir()  # Ensure the output directory exists
        unique_id = uuid.uuid4()
        filename = f"{prefix}_{unique_id}.txt"
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w") as file:
            file.write(text)
        print(f"Text saved to {output_path}")

    def create_output_dir(self):
        """Create the output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

