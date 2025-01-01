
import cv2
import os
import numpy as np
import pytesseract
from PIL import Image
import pyautogui
import uuid
import io
import json

class ImageService:
    def __init__(self, cell_size, scaling_factors, min_box_size=30):
        """Initialize ImageService with cell size, scaling factors, and minimum box size."""
        self.cell_size = cell_size
        self.scaling_factor_x, self.scaling_factor_y = scaling_factors
        self.min_box_size = min_box_size
        self.custom_config = r'--oem 3 --psm 6'



    def compute_and_cache_scaling_factors(self):
        """Compute and cache the scaling factors if they haven't been calculated yet."""
        if self.scaling_factor_x is None or self.scaling_factor_y is None:
            screen_width, screen_height = pyautogui.size()
            screenshot = pyautogui.screenshot()
            image_width, image_height = screenshot.size

            self.scaling_factor_x = image_width / screen_width
            self.scaling_factor_y = image_height / screen_height

    def get_scaling_factors(self):
        """Return the cached scaling factors, computing them if necessary."""
        self.compute_and_cache_scaling_factors()
        return self.scaling_factor_x, self.scaling_factor_y

    def adjust_coordinates_for_scaling(self, x, y):
        """Adjust the given coordinates based on the cached scaling factors."""
        adjusted_x = int(x / self.scaling_factor_x)
        adjusted_y = int(y / self.scaling_factor_y)
        # print(f"[INFO] Adjusted coordinates: ({adjusted_x}, {adjusted_y})")
        return adjusted_x, adjusted_y

    
    def create_output_dir(self):
        """Ensure the output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)

    def load_image(self, image_path):
        """Load an image from the given path."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Error: {image_path} not found!")
        return image

    def save_image(self, image, filename):
        """Save the processed image to the output directory."""
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, image)
        print(f"Image saved to {output_path}")

    def detect_elements(self, image):
        """Detect contours (elements) in the image and return valid bounding boxes."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
        edges = cv2.Canny(blurred, 5, 5)  # Lower thresholds to capture finer details


        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [
            cv2.boundingRect(c) for c in contours
            if self.is_valid_box(c)
        ]  # Filter boxes by size
        
        return bounding_boxes

    def is_valid_box(self, contour):
        """Check if the bounding box is larger than the minimum size."""
        x, y, w, h = cv2.boundingRect(contour)
        return w >= self.min_box_size and h >= self.min_box_size
    
    def draw_uniform_grid(self, image, bounding_boxes, line_color=(0, 0, 255)):
        """
        Draw a uniform grid with square cells, ensuring no grid line crosses a bounding box.
        """
        height, width, _ = image.shape
        cell_size = self.cell_size

        # Track all adjusted grid lines (vertical and horizontal)
        vertical_lines = self.get_adjusted_lines(bounding_boxes, axis='x', max_value=width)
        horizontal_lines = self.get_adjusted_lines(bounding_boxes, axis='y', max_value=height)

        # Draw the adjusted vertical lines
        for x in vertical_lines:
            cv2.line(image, (x, 0), (x, height), line_color, 1)

        # Draw the adjusted horizontal lines
        for y in horizontal_lines:
            cv2.line(image, (0, y), (width, y), line_color, 1)

        return image

    def get_adjusted_lines(self, bounding_boxes, axis='x', max_value=1000):
        """
        Generate uniform grid lines, adjusted to avoid crossing bounding boxes.
        """
        cell_size = self.cell_size
        lines = []

        # Start placing lines from the first position, moving by cell size
        pos = 0
        while pos < max_value:
            # Check if the line intersects any bounding box
            if not self.line_intersects_boxes(pos, bounding_boxes, axis):
                lines.append(pos)  # Add the line if it's valid
            else:
                # Adjust the line to just outside the conflicting bounding box
                pos = self.get_next_valid_position(pos, bounding_boxes, axis)

            pos += cell_size  # Move to the next cell position

        return lines

    def line_intersects_boxes(self, pos, bounding_boxes, axis):
        """
        Check if a given line position intersects any bounding box.
        """
        for (x, y, w, h) in bounding_boxes:
            if axis == 'x' and x <= pos <= x + w:
                return True  # Line intersects horizontally
            if axis == 'y' and y <= pos <= y + h:
                return True  # Line intersects vertically
        return False

    def get_next_valid_position(self, pos, bounding_boxes, axis):
        """
        Find the next valid line position that doesn't intersect any bounding box.
        """
        for (x, y, w, h) in bounding_boxes:
            if axis == 'x' and x <= pos <= x + w:
                return x + w  # Move to the right edge of the box
            if axis == 'y' and y <= pos <= y + h:
                return y + h  # Move to the bottom edge of the box
        return pos  # If no conflicts, return the original position


    def process_image_with_uniform_grid(self, image_path, title="Uniform Grid Image"):
        """
        Process an image to draw bounding boxes and a smart uniform grid with fixed-size square cells.
        """
        image = self.load_image(image_path)
        self.save_image(image, prefix="original_image")
        bounding_boxes = self.detect_elements(image)
        image_with_boxes = self.draw_bounding_boxes(image, bounding_boxes)
        self.save_image(image_with_boxes, prefix="image_with_boxes")
        final_image = self.draw_uniform_grid(image_with_boxes, bounding_boxes)
        self.save_image(final_image, prefix="final_grid_image") 

        # Add a title to the image
        final_image = self.add_title(final_image, title)

        # Save the final image
        filename = f"smart_uniform_grid_{os.path.basename(image_path)}"
        self.save_image(final_image, filename)

        return final_image

    def draw_bounding_boxes(self, image, bounding_boxes):
        """Draw bounding boxes with labels on the image."""
        for idx, (x, y, w, h) in enumerate(bounding_boxes):
            # Draw a red rectangle around the element
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box
        return image

    def add_title(self, image, title):
        """Add a title bar at the top of the image."""
        title_bar = np.zeros((50, image.shape[1], 3), dtype=np.uint8)  # Black bar
        cv2.putText(title_bar, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # Red text
        return np.vstack((title_bar, image))

    def get_grid_lines(self, bounding_boxes, axis='x', max_value=1000):
        """
        Compute grid lines based on bounding boxes along the specified axis.
        Avoid crossing any bounding box along the axis and ensure minimum cell size.
        """
        edges = set()
        for (x, y, w, h) in bounding_boxes:
            if axis == 'x':
                edges.add(x)
                edges.add(x + w)
            elif axis == 'y':
                edges.add(y)
                edges.add(y + h)

        sorted_edges = sorted(edges)
        grid_lines = []

        # Use cell_size instead of min_cell_size
        prev_line = 0
        for edge in sorted_edges:
            if edge - prev_line >= self.cell_size:  # Corrected reference
                grid_lines.append(edge)
                prev_line = edge

        if sorted_edges[-1] < max_value:
            while prev_line + self.cell_size < max_value:
                prev_line += self.cell_size
                grid_lines.append(prev_line)

        return grid_lines


    def draw_smart_grid(self, image, bounding_boxes, line_color=(0, 255, 0)):
        """
        Draw a smart grid on the image, avoiding intersections with bounding boxes
        and respecting the minimum cell size.
        """
        height, width, _ = image.shape

        # Get the smart grid lines based on bounding boxes and minimum cell size
        vertical_lines = self.get_grid_lines(bounding_boxes, axis='x', max_value=width)
        horizontal_lines = self.get_grid_lines(bounding_boxes, axis='y', max_value=height)

        # Draw the vertical lines
        for x in vertical_lines:
            cv2.line(image, (x, 0), (x, height), line_color, 1)

        # Draw the horizontal lines
        for y in horizontal_lines:
            cv2.line(image, (0, y), (width, y), line_color, 1)

        return image
    
    def extract_text_from_elements(self, image_path):
        """
        Extract text from individual bounding boxes by cropping each element
        and applying OCR on the cropped image.
        """
        # Step 1: Load the original image without resizing or scaling
        image = self.load_image(image_path)
        
        # Step 2: Detect bounding boxes on the original image
        bounding_boxes = self.detect_elements(image)

        extracted_texts = []

        print(f"Total bounding boxes detected: {len(bounding_boxes)}")  # Debugging print

        # Step 3: Process each bounding box separately for OCR
        for idx, (x, y, w, h) in enumerate(bounding_boxes):
            # Crop the image to the bounding box
            cropped_image = image[y:y + h, x:x + w]

            # Convert to PIL image for Tesseract
            pil_image = Image.fromarray(cropped_image)

            # Extract text from the cropped image using Tesseract
            # Using '--psm 7' for single-line text, adjust as needed
            text = pytesseract.image_to_string(pil_image, config='--psm 7').strip()

            # Debugging: Output the extracted text for each bounding box
            print(f"Extracted Text from Bounding Box {idx + 1}: {repr(text)}")

            # Append the text and coordinates to the list
            extracted_texts.append({
                "text": text,
                "coordinates": (x, y, w, h)
            })

        # Step 4: Output the final extracted texts and coordinates
        print("\nFinal Extracted Texts with Coordinates:")
        for item in extracted_texts:
            print(item)

        return extracted_texts


    def resize_image_with_size_limit(self, image, max_size_bytes):
        scaling_factor = 0.95  # Start with 95% of original size
        while scaling_factor > 0.5:  # Don't reduce below 50% of original size
            new_width = int(image.width * scaling_factor)
            new_height = int(image.height * scaling_factor)
            resized_image = image.resize((new_width, new_height), resample=Image.LANCZOS)
            buffered = io.BytesIO()
            resized_image.save(buffered, format="JPEG", quality=85, optimize=True)
            size_in_bytes = buffered.tell()
            if size_in_bytes <= max_size_bytes:
                print(f"[INFO] Resized image to {new_width}x{new_height}, size: {size_in_bytes} bytes")
                return resized_image, buffered.getvalue()
            scaling_factor -= 0.02  # Reduce by 2% and try again
        # If size limit not met, return the smallest possible image
        print("[WARNING] Could not resize image within size limit. Returning smallest possible image.")
        return resized_image, buffered.getvalue()







    def process_image_with_smart_grid(self, image_path, title="Smart Grid Image"):
        """
        Process an image to draw bounding boxes and a smart grid that adapts to elements.
        """
        image = self.load_image(image_path)
        bounding_boxes = self.detect_elements(image)
        image_with_boxes = self.draw_bounding_boxes(image, bounding_boxes)
        final_image = self.draw_smart_grid(image_with_boxes, bounding_boxes)

        # Add a title to the image
        final_image = self.add_title(final_image, title)

        # Save the final image
        filename = f"smart_grid_{os.path.basename(image_path)}"
        self.save_image(final_image, filename)

        return final_image
    
    def get_center_of_box(self, x, y, w, h):
        """Calculate the center (x, y) of a bounding box."""
        center_x = x + w // 2
        center_y = y + h // 2
        return (center_x, center_y)
    
    def encode_image(image_path, target_size=(512, 512), quality=50):
        """Resize and compress the image, then encode it to base64."""
        with Image.open(image_path) as img:
            # Resize the image to target size
            img = img.resize(target_size)

            # Save the image to a byte stream with reduced quality
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)

            # Encode the byte stream to base64
            encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
            return encoded_image
    
    def process_image_with_coordinates(self, image_path):
        """
        Process an image to detect elements and return their adjusted center coordinates.
        """
        image = self.load_image(image_path)
        bounding_boxes = self.detect_elements(image)
        image_with_boxes = self.draw_bounding_boxes(image, bounding_boxes)

        # Extract and adjust center coordinates for each bounding box
        element_coordinates = [
            self.adjust_coordinates_for_scaling(
                *self.get_center_of_box(x, y, w, h)
            ) for (x, y, w, h) in bounding_boxes
        ]

        # Save the processed image with bounding boxes
        filename = f"processed_with_coordinates_{os.path.basename(image_path)}"
        self.save_image(image_with_boxes, filename)

        return element_coordinates

    def get_center_of_box(self, x, y, w, h):
        """Calculate the center (x, y) of a bounding box."""
        center_x = x + w // 2
        center_y = y + h // 2
        return (center_x, center_y)

    def preprocess_image_for_ocr(self, image):
        """Preprocess the image to improve OCR accuracy."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to smooth out noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding to create a binary image
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Optionally, resize the image to make the text clearer
        scale_percent = 150  # Increase size by 150%
        width = int(thresh.shape[1] * scale_percent / 100)
        height = int(thresh.shape[0] * scale_percent / 100)
        resized_image = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Optionally, sharpen the image to make text clearer
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(resized_image, -1, kernel)
        
        return sharpened






    # HELPER METHODS: TODO: Must be moved to utils
    def generate_unique_filename(self, prefix="image"):
        """Generate a unique filename using a random UUID."""
        unique_id = uuid.uuid4()
        return f"{prefix}_{unique_id}.jpg"

    def create_output_dir(self):
        """Ensure the output directory 'image_service_work' exists."""
        self.output_dir = "image_service_work"
        os.makedirs(self.output_dir, exist_ok=True)

    def save_image(self, image, prefix="output"):
        """Save the processed image with a unique filename in the output directory."""
        self.create_output_dir()  # Ensure the directory exists
        filename = self.generate_unique_filename(prefix)
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, image)
        print(f"Image saved to {output_path}")

    def save_text_to_file(self, text, prefix="extracted_text"):
        """Save the extracted text to a file with a unique name in the output directory."""
        self.create_output_dir()  # Ensure the output directory exists
        unique_id = uuid.uuid4()
        filename = f"{prefix}_{unique_id}.txt"
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w") as file:
            file.write(text)
        print(f"Text saved to {output_path}")
    
    def save_json_to_file(self, data, prefix="processed_elements"):
        """Save the processed elements data to a JSON file with a unique name in the output directory."""
        self.create_output_dir()  # Ensure the output directory exists
        unique_id = uuid.uuid4()
        filename = f"{prefix}_{unique_id}.json"
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w") as file:
            json.dump(data, file, indent=4)
        print(f"JSON data saved to {output_path}")