# import os
# from real_estate_agent.agent import RealEstateAgent

# def main():
#     # Initialize the real estate agent
#     gemini_api_key = os.getenv("GEMINI_API_KEY")
#     csv_path = "realestatedata.csv"
#     agent = RealEstateAgent(gemini_api_key, csv_path)
    
#     # Initial greeting
#     print("Real Estate Assistant initialized. Type 'exit' to quit.")
#     initial_greeting = "Hello! I'm your personal real estate assistant. How can I help you find your perfect property today?"
#     print(f"Assistant: {initial_greeting}")
    
#     while True:
#         try:
#             # Get user input
#             user_input = input("\nYou: ").strip()
            
#             if user_input.lower() in ('exit', 'quit', 'bye'):
#                 farewell = "Thank you for using our service. Have a great day!"
#                 print(f"Assistant: {farewell}")
#                 break
            
#             # Process the input and generate response with typing effect
#             agent.demo_typing_response(user_input)
            
#         except KeyboardInterrupt:
#             print("\nSession terminated.")
#             break
#         except Exception as e:
#             print(f"\nError: {e}")
#             continue

# if __name__ == "__main__":
#     main() 