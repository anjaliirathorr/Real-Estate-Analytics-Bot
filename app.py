from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import logging
import pandas as pd
from dotenv import load_dotenv
from real_estate_agent import RealEstateAgent

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
KNOWLEDGE_FOLDER = os.getenv('KNOWLEDGE_FOLDER', 'knowledge')  # Path to the knowledge folder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load all CSV files from the knowledge folder
def load_knowledge_data(folder_path):
    """Load and combine all CSV files from the specified folder."""
    if not os.path.exists(folder_path):
        logger.error(f"Knowledge folder not found: {folder_path}")
        return pd.DataFrame()  # Return an empty DataFrame if the folder is missing

    combined_data = pd.DataFrame()
    try:
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                logger.info(f"Loading data from {file_path}")
                data = pd.read_csv(file_path)
                combined_data = pd.concat([combined_data, data], ignore_index=True)
        if combined_data.empty:
            logger.warning("No data found in the knowledge folder.")
        else:
            logger.info("All CSV files loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading knowledge data: {e}")
    return combined_data

# Load the knowledge data
knowledge_data = load_knowledge_data(KNOWLEDGE_FOLDER)

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Initialize the Real Estate Agent
try:
    if knowledge_data.empty:
        logger.warning("Knowledge data is empty. The agent may not function as expected.")
    agent = RealEstateAgent(GEMINI_API_KEY, knowledge_data)
    logger.info("Real Estate Agent initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Real Estate Agent: {e}")
    agent = None

@app.route('/')
def home():
    """Serve the main chat interface."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat messages and return agent responses."""
    if not agent:
        return jsonify({
            'success': False,
            'message': 'Agent initialization failed. Check server logs.',
            'acknowledgment': 'Sorry, I encountered an error.'
        }), 500

    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({
            'success': False,
            'message': 'No message provided',
            'acknowledgment': 'I need a message to respond to.'
        }), 400

    try:
        response_data = agent.respond_with_typing(user_message)
        return jsonify({
            'success': True,
            'acknowledgment': response_data['acknowledgment'],
            'message': response_data['full_response']
        })
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'acknowledgment': 'Sorry, I encountered an error processing your request.'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Return statistics about the real estate data."""
    if not agent:
        return jsonify({'success': False, 'message': 'Agent initialization failed'}), 500
    try:
        return jsonify({'success': True, 'stats': agent.stats})
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)