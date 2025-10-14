import random
import logging
import os

from .agent import RealEstateAgent

logger = logging.getLogger(__name__)

def setup_logging():
    """Set up logging for the CLI application."""
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def run_cli_interface():
    """Run a simple CLI loop for the Real Estate Agent chatbot."""
    # Set up logging
    setup_logging()
    
    # Get API key from environment variable or use default for demo
    gemini_api_key = os.environ.get(
        "GEMINI_API_KEY", 
        "AIzaSyCHTlehGQFmbl9Cq9HAPKEGiON7CC8HILY"  # Default key for demo
    )
    
    # Get knowledge folder path from environment variable or use default for demo
    knowledge_folder = os.environ.get("REAL_ESTATE_KNOWLEDGE_FOLDER", "knowledge_folder")
    
    # Check if the knowledge folder exists
    if not os.path.exists(knowledge_folder):
        logger.warning(f"Knowledge folder not found at {knowledge_folder}. Using sample data instead.")
        # Optionally, create a sample folder with minimal data for demo if folder doesn't exist
        # create_sample_folder(knowledge_folder)
    
    # Initialize the agent
    agent = RealEstateAgent(gemini_api_key, csv_path)
    print("Real Estate Assistant initialized. Type 'exit' to quit.")
    print(random.choice(agent.greetings))
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ('exit', 'quit', 'bye'):
                print("Assistant: Thank you for using the Real Estate Assistant. Goodbye!")
                break
            
            # Simulate the typing effect in the response
            agent.demo_typing_response(user_input)
        except KeyboardInterrupt:
            print("\nAssistant: Session terminated. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in conversation: {e}")
            print("\nAssistant: I encountered an error. Let's try again.")

if __name__ == "__main__":
    run_cli_interface()