from difflib import SequenceMatcher

class MatcherService:

    def match_texts_with_coordinates(self, browser_text, extracted_texts):
        """
        Match text from the browser with coordinates from the extracted texts, ensuring that only
        texts viewable in the screenshot are considered as matches. If no reasonable match is found,
        the element is still added with a match ratio of 0.
        
        Args:
        - browser_text (str): The text content extracted from the browser.
        - extracted_texts (list of dict): The text content with coordinates extracted from the screenshot.
        
        Returns:
        - list of dict: Matched elements with text, coordinates, and match ratio.
        """
        matched_elements = []

        # Use the extracted texts from the screenshot as the source of truth
        for extracted in extracted_texts:
            best_match = None
            highest_ratio = 0

            for browser_line in browser_text.splitlines():
                # Compare each line of the browser text with the current extracted text
                ratio = SequenceMatcher(None, browser_line, extracted["text"]).ratio()
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_match = browser_line

            # Always add the element, whether matched well or not
            matched_elements.append({
                "text": extracted["text"],
                "coordinates": extracted["coordinates"],
                "match_ratio": highest_ratio  # This will be 0 if no good match is found
            })

        return matched_elements
