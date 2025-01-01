from services.image_service import ImageService  # Ensure correct path

def main():
    # Step 1: Create an instance of the ImageService with a minimum box size
    image_service = ImageService(cell_size=200)  # Create 100x100 square grid cells

    # Step 2: Ensure the output directory exists
    image_service.create_output_dir()

    # Step 3: Define the input image path
    input_image_path = "googleScreenshot.png"

    try:
        # Step 4: Process the image with bounding boxes
        processed_image = image_service.process_image_with_uniform_grid(input_image_path)

        print("Image processing complete!")
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
