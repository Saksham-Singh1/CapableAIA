from controller import Controller
from services.image_service import ImageService  # Assuming you have your ImageService in a separate file

def main():
    """Main entry point for the application."""
    # Create an instance of ImageService to compute scaling factors
    cell_size = 200
    temp_image_service = ImageService(cell_size, (None, None))
    scaling_factors = temp_image_service.get_scaling_factors()
    print("Computed Scaling Factors:", scaling_factors)

    # Now create a new ImageService with the computed scaling factors
    image_service = ImageService(cell_size, scaling_factors)

    print("Calling controller now")
    # Initialize the Controller with the computed scaling factors
    controller = Controller(scaling_factors=scaling_factors)

    # Get user input for the instruction
    instruction = input("Enter the instruction for the system (e.g., 'Sign in to Google'): ")

    # Optimize the instruction using ChatGPT
    optimized_instruction = controller.optimize_instruction(instruction)

    # Start the program by calling the run method with the optimized instruction
    controller.run(endGoalInstruction=optimized_instruction)

if __name__ == "__main__":
    main()
