import os
import re
import sys
import time
import random
import logging
import pandas as pd
import numpy as np
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import helper functions from our submodules
from .filtering import semantic_search
from .insights import get_property_insights, format_response_with_typing_effect, get_dataset_overview
from .message_processor import process_user_message_with_typing, respond_with_typing, demo_typing_response

logger = logging.getLogger(__name__)

class RealEstateAgent:
    def __init__(self, gemini_api_key, csv_path):
        """Initialize the Real Estate Agent with API key, load CSV, and set up conversation state."""
        # Hardcode the API key and CSV path for reliability
        self.gemini_api_key = "AIzaSyCHTlehGQFmbl9Cq9HAPKEGiON7CC8HILY"
        self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
        self.conversation_state = "initial"
        self.current_transaction = None  # 'buy' or 'rent'
        self.user_requirements = {
            'transaction_type': None,
            'property_type': None,
            'layout': None,
            'budget': None,
            'down_payment': None,
            'location': None,
            'term': None,  # for rental: 'short' or 'long'
            'ready_status': None  # 'ready' or 'off-plan'
        }
        self.user_preferences = {}
        self.conversation_history = []
        self.current_query = ""
        self.questions_asked = set()  # for tracking asked questions

        # Define greetings and follow-up templates
        self.greetings = [
            "Hello! I'm your personal real estate assistant. How can I help you find your perfect property today?",
            "Hi there! I'm here to help you find the ideal property. What are you looking for?",
            "Welcome! I'm your real estate expert. Tell me what you're looking for in your next home.",
            "Good day! I'm ready to assist with your property search. What kind of property are you interested in?"
        ]

        self.conversation_flow = {
            'initial': {
                'question': "What do you want to do? (Buy/Rent)",
                'next': 'transaction_type'
            },
            'transaction_type': {
                'buy': {
                    'question': "What type of user best describes you?",
                    'next': 'ready_status'
                },
                'rent': {
                    'question': "Are you looking to rent short term or long term?",
                    'next': 'property_type'
                }
            },
            'ready_status': {
                'question': "Do you want ready or Off-plan?",
                'next': 'property_type'
            },
            'property_type': {
                'question': "What is your target unit?",
                'next': 'layout'
            },
            'layout': {
                'question': "What is your target layout (bedrooms)?",
                'next': 'budget'
            },
            'budget': {
                'question': "What is your budget?",
                'next': 'down_payment'
            },
            'down_payment': {
                'question': "What is your down payment budget?",
                'next': 'location'
            },
            'location': {
                'question': "Preferred Location?",
                'next': 'summary'
            }
        }

        # Define the knowledge folder path
        self.knowledge_folder = csv_path  # Use the csv_path argument as the knowledge folder path

        # Load property data from the knowledge folder
        self.knowledge_base = self.load_properties_from_folder(self.knowledge_folder)

        # If the knowledge base is empty, create sample data
        if self.knowledge_base.empty:
            logger.warning("Could not load any CSV files, creating sample data")
            self.knowledge_base = self._create_sample_data()

        # Preprocess the data for better searching
        self.knowledge_base = self._preprocess_data(self.knowledge_base)

        # Analyze the preprocessed knowledge base
        self.stats = self.analyze_knowledge_base()
        
    def load_properties_from_folder(self, folder_path):
        """Load and combine all CSV files from the specified folder into a single DataFrame."""
        combined_data = pd.DataFrame()
        try:
            if not os.path.exists(folder_path):
                logger.error(f"Knowledge folder not found: {folder_path}")
                return combined_data

            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(folder_path, file)
                    logger.info(f"Loading data from {file_path}")
                    data = pd.read_csv(file_path)
                    combined_data = pd.concat([combined_data, data], ignore_index=True)

            if combined_data.empty:
                logger.warning("No data found in the knowledge folder.")
            else:
                logger.info(f"Successfully loaded {len(combined_data)} rows from the knowledge folder.")

        except Exception as e:
            logger.error(f"Error loading knowledge data: {e}")

        return combined_data

    def _preprocess_data(self, df):
        """Preprocess the data for better search and matching."""
        try:
            # Convert column names to standard format
            df.columns = [col.strip() for col in df.columns]
            
            # Ensure essential columns exist
            essential_columns = ['Price', 'Location', 'Type', 'Bedrooms', 'Area', 'Description', 'Title', 'Purpose', 'Furnishing']
            for col in essential_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Create processed columns for better text matching
            if 'Location' in df.columns:
                df['Processed_Location'] = df['Location'].astype(str).apply(lambda x: x.lower())
                
            if 'Type' in df.columns:
                df['Processed_Type'] = df['Type'].astype(str).apply(lambda x: x.lower())
                
            if 'Description' in df.columns:
                df['Processed_Description'] = df['Description'].astype(str).apply(lambda x: x.lower())
            
            # Convert price to numeric - handle various price formats
            if 'Price' in df.columns:
                # First, clean the price column
                df['Price'] = df['Price'].astype(str).apply(lambda x: re.sub(r'[^\d.]', '', x))
                # Convert to numeric, replacing empty strings and invalid values with NaN
                df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
                # Convert NaN to None for better handling
                df['Price'] = df['Price'].replace({np.nan: None})
                
            # Convert bedrooms to numeric
            if 'Bedrooms' in df.columns:
                # If bedrooms is string, extract numbers
                if df['Bedrooms'].dtype == 'object':
                    df['Bedrooms'] = df['Bedrooms'].astype(str).apply(
                        lambda x: re.search(r'(\d+)', x).group(1) if re.search(r'(\d+)', x) else None
                    )
                df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors='coerce')
                # Convert NaN to None for better handling
                df['Bedrooms'] = df['Bedrooms'].replace({np.nan: None})
            
            logger.info(f"Preprocessed data: {len(df)} rows")
            logger.info(f"Price column sample: {df['Price'].head()}")
            return df
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return df

    def load_properties_from_csv(self, csv_path):
        """Load property data from a CSV file into a pandas DataFrame."""
        try:
            if not os.path.exists(csv_path):
                logger.error(f"CSV file not found: {csv_path}")
                return pd.DataFrame(columns=['Price', 'Location', 'Type', 'Bedrooms', 'Area', 'Description', 'Title', 'Purpose', 'Furnishing'])
            logger.info(f"Loading CSV file from: {csv_path}")
            df = pd.read_csv(csv_path)
            logger.info(f"Successfully loaded {len(df)} properties")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            return pd.DataFrame(columns=['Price', 'Location', 'Type', 'Bedrooms', 'Area', 'Description', 'Title', 'Purpose', 'Furnishing'])

    def analyze_knowledge_base(self):
        """Perform analysis on the knowledge base and store basic statistics."""
        df = self.knowledge_base
        stats = {}
        if 'Purpose' in df.columns:
            stats['purposes'] = df['Purpose'].value_counts().to_dict()
        if 'Type' in df.columns:
            stats['types'] = df['Type'].value_counts().to_dict()
        if 'Price' in df.columns:
            stats['price_min'] = df['Price'].min()
            stats['price_max'] = df['Price'].max()
        if 'Location' in df.columns:
            stats['top_locations'] = df['Location'].value_counts().head(10).to_dict()
        if 'Bedrooms' in df.columns:
            stats['bedrooms'] = df['Bedrooms'].value_counts().to_dict()
        logger.info("Knowledge base analysis complete")
        return stats

    # ----- Methods that use helper functions -----
    def _is_greeting(self, text: str) -> bool:
        """Check if the text is a greeting."""
        greeting_keywords = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        text_lower = text.lower().strip()
        return any(text_lower.startswith(keyword) for keyword in greeting_keywords)

    def _is_basic_question(self, text: str) -> bool:
        """Check if the text contains a basic question about the assistant."""
        basic_questions = [
            'who are you', 'what can you do', 'how can you help', 'what do you do',
            'assist me', 'tell me about yourself'
        ]
        text_lower = text.lower()
        return any(question in text_lower for question in basic_questions)

    def _is_data_question(self, text: str) -> bool:
        """Check if the text is asking for data or statistics about the properties."""
        data_questions = [
            'property types', 'available properties', 'property data', 'show me the data',
            'data overview', 'price ranges', 'top locations', 'property statistics'
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in data_questions)

    def _semantic_search(self, filtered_df, user_query, top_k=5):
        """Perform semantic search on filtered properties using the helper."""
        return semantic_search(filtered_df, user_query, top_k=top_k)

    def get_property_insights(self, properties):
        """Return formatted property details using the helper."""
        return get_property_insights(properties)

    def _format_response_with_typing_effect(self, properties, user_query, query_details):
        """Format response text using typing effect style using the helper."""
        return format_response_with_typing_effect(properties, user_query, query_details)

    def _get_dataset_overview(self):
        """Get an overview of the dataset using the helper, passing in the stored stats."""
        return get_dataset_overview(self.stats)

    def _process_user_message_with_typing(self, message):
        """Process a user message. Delegates to the helper function."""
        # This helper may use methods such as _is_greeting, _is_basic_question, etc.
        # For simplicity, assume these helper checks remain methods of this class.
        return process_user_message_with_typing(self, message)

    def respond_with_typing(self, user_message):
        """Return full assistant response using helper function."""
        return respond_with_typing(self, user_message)

    def demo_typing_response(self, user_message):
        """Demonstrate the full response with a simulated typing effect using helper."""
        return demo_typing_response(self, user_message)

    def _filter_properties(self, properties, required_property_count=5, **filters):
        """
        Filter the properties based on the provided filters.
        
        Args:
            properties (pandas.DataFrame): The properties dataset to filter
            required_property_count (int): Number of properties to return
            **filters: Keyword arguments for filtering
                - property_type: Type of property (apartment, villa, etc.)
                - bedrooms: Number of bedrooms
                - location: Location of the property
                - min_price: Minimum price
                - max_price: Maximum price
                - purpose: Sale or Rent
        
        Returns:
            pandas.DataFrame: Filtered properties
        """
        logger.info(f"Filtering properties with filters: {filters}")
        
        # Check if properties is empty or not a DataFrame
        if properties is None or not isinstance(properties, pd.DataFrame) or len(properties) == 0:
            logger.warning("No properties available to filter")
            return pd.DataFrame()
        
        filtered_df = properties.copy()
        
        # Filter by purpose (Sale/Rent)
        if 'purpose' in filters and filters['purpose']:
            filtered_df = filtered_df[filtered_df['Purpose'] == filters['purpose']]
            logger.info(f"After purpose filter: {len(filtered_df)} properties")
        
        # Filter by number of bedrooms
        if 'bedrooms' in filters and filters['bedrooms'] is not None:
            try:
                # Convert bedrooms value to integer if it's a string
                bedrooms_value = filters['bedrooms']
                if isinstance(bedrooms_value, str):
                    bedrooms_match = re.search(r'(\d+)', bedrooms_value)
                    if bedrooms_match:
                        bedrooms_value = int(bedrooms_match.group(1))
                    else:
                        logger.warning(f"Could not extract bedroom count from: {bedrooms_value}")
                        bedrooms_value = None
                
                if bedrooms_value is not None:
                    # Create a numeric version of the Bedrooms column for comparison
                    if 'Bedrooms_Numeric' not in filtered_df.columns:
                        if filtered_df['Bedrooms'].dtype == 'object':
                            filtered_df['Bedrooms_Numeric'] = filtered_df['Bedrooms'].apply(
                                lambda x: int(re.search(r'(\d+)', str(x)).group(1)) 
                                if pd.notna(x) and re.search(r'(\d+)', str(x)) 
                                else None
                            )
                        else:
                            filtered_df['Bedrooms_Numeric'] = pd.to_numeric(filtered_df['Bedrooms'], errors='coerce')
                    
                    # Filter by exact bedroom count - ensure it's an exact match
                    filtered_df = filtered_df[filtered_df['Bedrooms_Numeric'] == bedrooms_value]
                    logger.info(f"After bedrooms filter ({bedrooms_value}): {len(filtered_df)} properties")
            except Exception as e:
                logger.error(f"Error filtering by bedrooms: {e}")
        
        # Filter by price range
        if ('min_price' in filters or 'max_price' in filters) and 'Price' in filtered_df.columns:
            try:
                # Convert price values to numeric for comparison
                if 'Price_Numeric' not in filtered_df.columns:
                    # Extract and convert prices from string format with AED and commas to numeric
                    filtered_df['Price_Numeric'] = filtered_df['Price'].apply(
                        lambda x: float(re.sub(r'[^\d.]', '', str(x))) if pd.notna(x) else None
                    )
                
                # Apply min price filter
                if 'min_price' in filters and filters['min_price'] is not None:
                    min_price = filters['min_price']
                    # Convert string prices like "2M" or "2 million AED" to numeric
                    if isinstance(min_price, str):
                        min_price = self._convert_price_to_numeric(min_price)
                    try:
                        min_price = float(min_price)
                        filtered_df = filtered_df[filtered_df['Price_Numeric'] >= min_price]
                        logger.info(f"After min price filter ({min_price}): {len(filtered_df)} properties")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Invalid min_price value: {min_price}, error: {e}")
                
                # Apply max price filter
                if 'max_price' in filters and filters['max_price'] is not None:
                    max_price = filters['max_price']
                    # Convert string prices like "3M" or "3 million AED" to numeric
                    if isinstance(max_price, str):
                        max_price = self._convert_price_to_numeric(max_price)
                    try:
                        max_price = float(max_price)
                        filtered_df = filtered_df[filtered_df['Price_Numeric'] <= max_price]
                        logger.info(f"After max price filter ({max_price}): {len(filtered_df)} properties")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Invalid max_price value: {max_price}, error: {e}")
                
            except Exception as e:
                logger.error(f"Error filtering by price: {e}")
        
        # Filter by location
        if 'location' in filters and filters['location']:
            try:
                location = filters['location'].lower()
                # Case-insensitive substring matching
                location_mask = filtered_df['Location'].fillna('').str.lower().str.contains(location, case=False)
                filtered_df = filtered_df[location_mask]
                logger.info(f"After location filter ({location}): {len(filtered_df)} properties")
            except Exception as e:
                logger.error(f"Error filtering by location: {e}")
        
        # Filter by property type
        if 'property_type' in filters and filters['property_type']:
            try:
                property_type = filters['property_type'].lower()
                # Case-insensitive substring matching
                type_mask = filtered_df['Type'].fillna('').str.lower().str.contains(property_type, case=False)
                filtered_df = filtered_df[type_mask]
                logger.info(f"After property type filter ({property_type}): {len(filtered_df)} properties")
            except Exception as e:
                logger.error(f"Error filtering by property type: {e}")
        
        # If we don't have enough properties, log a warning
        if len(filtered_df) < required_property_count:
            logger.warning(f"Found only {len(filtered_df)} properties, less than the required {required_property_count}")
            
            # Fallback: If we didn't find any properties matching all filters, try with just the bedrooms and price filters
            if len(filtered_df) == 0:
                logger.info("No properties matched all filters, trying with main filters only")
                filtered_df = properties.copy()
                
                # Apply only bedroom filter if specified
                if 'bedrooms' in filters and filters['bedrooms'] is not None:
                    try:
                        bedrooms_value = filters['bedrooms']
                        if isinstance(bedrooms_value, str):
                            bedrooms_match = re.search(r'(\d+)', bedrooms_value)
                            if bedrooms_match:
                                bedrooms_value = int(bedrooms_match.group(1))
                            else:
                                bedrooms_value = None
                        
                        if bedrooms_value is not None:
                            if 'Bedrooms_Numeric' not in filtered_df.columns:
                                if filtered_df['Bedrooms'].dtype == 'object':
                                    filtered_df['Bedrooms_Numeric'] = filtered_df['Bedrooms'].apply(
                                        lambda x: int(re.search(r'(\d+)', str(x)).group(1)) 
                                        if pd.notna(x) and re.search(r'(\d+)', str(x)) 
                                        else None
                                    )
                                else:
                                    filtered_df['Bedrooms_Numeric'] = pd.to_numeric(filtered_df['Bedrooms'], errors='coerce')
                            
                            filtered_df = filtered_df[filtered_df['Bedrooms_Numeric'] == bedrooms_value]
                            logger.info(f"Fallback with bedrooms filter only: {len(filtered_df)} properties")
                    except Exception as e:
                        logger.error(f"Error in fallback bedroom filtering: {e}")
                
                # If still no results or too few, return some properties from the original dataset
                if len(filtered_df) < required_property_count:
                    logger.warning(f"Fallback filtering returned {len(filtered_df)} properties, returning from original dataset")
                    return properties.head(required_property_count)
        
        # Return the filtered properties, limited to the required count
        result = filtered_df.head(required_property_count) if len(filtered_df) > required_property_count else filtered_df
        logger.info(f"Returning {len(result)} filtered properties")
        return result
    
    def _convert_price_to_numeric(self, price_str):
        """
        Convert price strings like "2M", "2 million", "2.5 million AED" to numeric values.
        
        Args:
            price_str (str): Price string to convert
            
        Returns:
            float: Numeric price value
        """
        if not price_str:
            return None
            
        # Remove currency symbols, commas, and other non-essential characters
        price_str = str(price_str).lower().strip()
        price_str = re.sub(r'[^\d.km]', '', price_str)
        
        # Handle "k" (thousands) and "m" (millions) suffixes
        multiplier = 1
        if 'k' in price_str:
            price_str = price_str.replace('k', '')
            multiplier = 1000
        elif 'm' in price_str:
            price_str = price_str.replace('m', '')
            multiplier = 1000000
            
        try:
            return float(price_str) * multiplier
        except ValueError:
            logger.error(f"Could not convert price string to numeric: {price_str}")
            return None

    def _call_gemini_api(self, prompt: str) -> dict:
        """
        Call the Gemini API to extract structured details from the query.
        The prompt instructs the model to return raw JSON without any markdown formatting.
        """
        try:
            logger.info(f"Calling Gemini API with prompt: {prompt[:100]}...")
            # Append additional instruction to get raw JSON output
            refined_prompt = prompt + "\n\nPlease return only raw JSON without any markdown formatting, code fences, or additional text."
            payload = {
                "contents": [{"parts": [{"text": refined_prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1024}
            }
            headers = {'Content-Type': 'application/json'}
            response = requests.post(self.gemini_url, headers=headers, data=json.dumps(payload))
        
            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code}, {response.text}")
                return self._fallback_query_parser(prompt)
        
            response_json = response.json()
            candidate = response_json.get("candidates", [{}])[0]
        
            # Extract text from candidate; try both possible keys
            if "content" in candidate and "parts" in candidate["content"]:
                generated_text = candidate["content"]["parts"][0].get("text", "")
            elif "output" in candidate:
                generated_text = candidate["output"]
            else:
                logger.error(f"Unexpected Gemini API response format: {candidate}")
                return self._fallback_query_parser(prompt)
        
            # Remove markdown code fences if present
            if "```" in generated_text:
                json_text = re.search(r'```(?:json)?(.*?)```', generated_text, re.DOTALL)
                generated_text = json_text.group(1) if json_text else generated_text.strip('`')
        
            # Extra cleaning: try to extract a JSON substring if extra text is included
            try:
                query_details = json.loads(generated_text)
                return query_details
            except json.JSONDecodeError:
                # Attempt a rough extraction of JSON by finding the first '{' and the last '}'
                start = generated_text.find('{')
                end = generated_text.rfind('}') + 1
                if start != -1 and end != -1:
                    json_text = generated_text[start:end]
                    try:
                        query_details = json.loads(json_text)
                        return query_details
                    except json.JSONDecodeError as json_err:
                        logger.error(f"Extraction attempt failed: {json_err}")
                logger.error(f"Error parsing JSON from Gemini API response. Raw text: {generated_text}")
                return self._fallback_query_parser(prompt)
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return self._fallback_query_parser(prompt)

    def _fallback_query_parser(self, prompt: str) -> dict:
        """
        Fallback query parser when Gemini API fails.
        Uses simple keyword matching to extract information.
        """
        query = prompt.lower()
        result = {
            "location": None,
            "bedrooms": None,
            "property_type": None,
            "max_price": None,
            "min_price": None,
            "price_qualifier": None,
            "amenities": [],
            "purpose": None,
            "furnishing": None
        }
        
        # Extract location (simplified)
        location_match = re.search(r'in\s+([a-zA-Z\s]+?)(?:with|\.|$)', query)
        if location_match:
            result["location"] = location_match.group(1).strip()
            
        # Extract bedrooms
        bedroom_match = re.search(r'(\d+)\s+bed', query)
        if bedroom_match:
            result["bedrooms"] = int(bedroom_match.group(1))
            
        # Extract property type
        property_types = ['house', 'apartment', 'villa', 'studio', 'townhouse']
        for p_type in property_types:
            if p_type in query:
                result["property_type"] = p_type
                break
                
        # Extract price
        price_match = re.search(r'(\d+)[k]?\s*(?:to|-)?\s*(\d+)[k]?', query)
        if price_match:
            result["min_price"] = float(price_match.group(1))
            result["max_price"] = float(price_match.group(2))
        else:
            under_match = re.search(r'under\s*(\d+)[k]?', query)
            if under_match:
                result["max_price"] = float(under_match.group(1))
                
        # Extract price qualifier
        price_qualifiers = ['cheap', 'affordable', 'luxury', 'budget', 'expensive']
        for qualifier in price_qualifiers:
            if qualifier in query:
                result["price_qualifier"] = qualifier
                break
                
        # Extract purpose
        if any(word in query for word in ['rent', 'lease']):
            result["purpose"] = "Rent"
        elif any(word in query for word in ['buy', 'purchase', 'sale']):
            result["purpose"] = "Sale"
            
        # Extract furnishing
        if 'furnished' in query:
            result["furnishing"] = "Furnished"
        elif 'unfurnished' in query:
            result["furnishing"] = "Unfurnished"
            
        logger.info(f"Fallback parser extracted: {result}")
        return result

    def _process_user_message(self, message):
        """Internal method to process and respond to a user message."""
        return self._process_user_message_with_typing(message)
    
    def _create_gemini_query_prompt(self, user_query: str) -> str:
        """
        Create a prompt for the Gemini API that extracts structured details from the query.
        The prompt instructs Gemini to output a JSON with keys like location, bedrooms, property_type, max_price, etc.
        """
        # Include previous context in the query
        context = ""
        if len(self.conversation_history) > 0:
            last_messages = self.conversation_history[-4:] if len(self.conversation_history) >= 4 else self.conversation_history
            context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in last_messages])
        
        prompt = f"""
    Analyze the following real estate query and extract structured details.
    
    Previous conversation context:
    {context}
    
    Current Query: "{user_query}"

    Please provide a JSON response with the following keys:
    - location: Extracted city or region (string or null)
    - bedrooms: Number of bedrooms (integer or null)
    - property_type: Type of property (string or null, options: 'house', 'apartment', 'villa', 'studio', 'townhouse')
    - max_price: Maximum budget (float or null)
    - min_price: Minimum budget (float or null)
    - price_qualifier: If a qualitative price descriptor is used (string or null, examples: 'cheap', 'affordable', 'luxury', 'budget', 'expensive')
    - amenities: List of desired amenities (list of strings or empty list)
    - purpose: Whether the user wants to rent or buy (string or null, options: 'Rent', 'Sale')
    - furnishing: Furnishing preferences (string or null, options: 'Furnished', 'Unfurnished')

    IMPORTANT GUIDELINES:
    1. For "bedrooms", extract a numeric value. For example, if the user says "2 bedrooms" or "two bedrooms", the value should be 2.
    2. For "price", extract numeric values and ignore currency symbols.
    3. For "property_type", use one of the options provided, normalizing terms (e.g., "flat" should be "apartment").
    4. For "purpose", look for keywords related to renting (e.g., "rent", "lease") or buying (e.g., "buy", "purchase") to determine the value.
    5. Look for specific phrases like "2 million AED" or "2M AED" and convert them to numbers (2000000).
    6. Always use the most specific information available, considering both the current query and any relevant context from previous messages.

    Provide null or empty values if not specified in the query or context.
    """
        return prompt.strip()

    def _create_sample_data(self):
        """Create sample property data for demo purposes when the CSV file cannot be loaded."""
        logger.info("Creating sample property data")
        import pandas as pd
        import numpy as np
        
        # Create a sample DataFrame with necessary columns
        data = {
            'Serial No.': range(1, 31),
            'Title': [f"Beautiful Property {i}" for i in range(1, 31)],
            'Purpose': ['Sale'] * 25 + ['Rent'] * 5,
            'Furnishing': np.random.choice(['Furnished', 'Unfurnished'], 30),
            'Type': np.random.choice(['Apartment', 'Villa', 'House', 'Studio', 'Townhouse'], 30),
            'Price(Currency)': ['AED'] * 30,
            'Price': [
                # First 5: 3BR apartments in Dubai in 2-3M range (EXACT MATCH)
                2500000, 2800000, 2200000, 2900000, 2100000,
                # Next 5: 3BR apartments in Dubai but above price range
                3500000, 4200000, 3800000, 4800000, 5200000,
                # Next 5: 2BR apartments in Dubai in the 2-3M range
                2200000, 2500000, 2800000, 2100000, 2900000,
                # Next 5: 5BR villas in Dubai in the 2-3M range
                2500000, 2200000, 2700000, 2900000, 2100000,
                # Next 5: 3BR properties not in Dubai in the 2-3M range
                2200000, 2500000, 2800000, 2100000, 2900000,
                # Rental properties with various bedrooms
                120000, 150000, 180000, 90000, 210000
            ],
            'Price(Frequency)': ['Yearly'] * 30,
            'Bedrooms': [
                # First 5 are 3BR apartments in Dubai in 2-3M range (EXACT MATCH)
                3, 3, 3, 3, 3,
                # Next 5 are 3BR apartments in Dubai but above price range
                3, 3, 3, 3, 3,
                # Next 5 are 2BR apartments in Dubai in the 2-3M range
                2, 2, 2, 2, 2,
                # Next 5 are 5BR villas in Dubai in the 2-3M range
                5, 5, 5, 5, 5,
                # Next 5 are 3BR properties not in Dubai in the 2-3M range
                3, 3, 3, 3, 3,
                # Rental properties with various bedrooms
                2, 3, 4, 1, 5
            ],
            'Bathrooms': [
                3, 3, 4, 4, 3,
                3, 4, 4, 5, 5,
                2, 2, 3, 2, 3,
                5, 6, 5, 7, 8,
                3, 4, 3, 3, 4,
                2, 3, 4, 1, 5
            ],
            'Area': [
                1500, 1800, 1600, 2000, 1400,
                2000, 2200, 1900, 2300, 2500,
                1200, 1300, 1400, 1100, 1500,
                3500, 4500, 3800, 5000, 6000,
                1600, 1700, 1500, 1800, 1900,
                1200, 1600, 2500, 800, 3000
            ],
            'Location': [
                # First 5 are 3BR apartments in Dubai in 2-3M range (EXACT MATCH)
                'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai',
                # Next 5 are 3BR apartments in Dubai but above price range
                'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai',
                # Next 5 are 2BR apartments in Dubai in the 2-3M range
                'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai',
                # Next 5 are 5BR villas in Dubai in the 2-3M range
                'Dubai', 'Dubai', 'Dubai', 'Dubai', 'Dubai',
                # Next 5 are 3BR properties not in Dubai in the 2-3M range
                'Abu Dhabi', 'Sharjah', 'Ajman', 'Ras Al Khaimah', 'Fujairah',
                # Rental properties with various locations
                'Dubai', 'Abu Dhabi', 'Dubai', 'Sharjah', 'Dubai'
            ],
            'Description': [
                f"Beautiful {bed}BR {prop_type} in {loc} with amazing views and premium finishes."
                for bed, prop_type, loc in zip(
                    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 2, 3, 4, 1, 5],
                    ['Apartment']*10 + ['Apartment']*5 + ['Villa']*5 + ['Apartment']*5 + ['Apartment']*5,
                    ['Dubai']*20 + ['Abu Dhabi', 'Sharjah', 'Ajman', 'Ras Al Khaimah', 'Fujairah'] + ['Dubai', 'Abu Dhabi', 'Dubai', 'Sharjah', 'Dubai']
                )
            ],
            'Amenities': [
                'Swimming Pool, Gym, Security, Parking',
                'Swimming Pool, Gym, Garden, Parking',
                'Swimming Pool, Gym, Security, Balcony',
                'Swimming Pool, Spa, Security, Parking',
                'Gym, Security, Parking, Children Area',
                'Swimming Pool, Gym, Security, Parking, Sauna',
                'Swimming Pool, Gym, Garden, Parking, Jacuzzi',
                'Swimming Pool, Gym, Security, Balcony, BBQ Area',
                'Swimming Pool, Spa, Security, Parking, Tennis Court',
                'Gym, Security, Parking, Children Area, Squash Court',
                'Swimming Pool, Gym, Parking',
                'Gym, Security, Parking',
                'Swimming Pool, Security, Parking',
                'Swimming Pool, Gym, Security',
                'Gym, Security, Parking, Balcony',
                'Swimming Pool, Gym, Garden, Security, Parking, Maid Room',
                'Swimming Pool, Gym, Garden, Security, Parking, Maid Room, Private Pool',
                'Swimming Pool, Gym, Garden, Security, Parking, Maid Room',
                'Swimming Pool, Gym, Garden, Security, Parking, Maid Room, Private Pool, BBQ Area',
                'Swimming Pool, Gym, Garden, Security, Parking, Maid Room, Private Pool, BBQ Area, Cinema',
                'Swimming Pool, Gym, Security, Parking',
                'Swimming Pool, Gym, Garden, Parking',
                'Swimming Pool, Gym, Garden, Security, Parking, Maid Room',
                'Gym, Security, Parking',
                'Swimming Pool, Gym, Garden, Security, Parking, Maid Room, Private Pool',
                'Swimming Pool, Gym, Security, Parking',
                'Swimming Pool, Gym, Garden, Parking',
                'Swimming Pool, Gym, Garden, Security, Parking, Maid Room',
                'Gym, Security, Parking',
                'Swimming Pool, Gym, Garden, Security, Parking, Maid Room, Private Pool'
            ]
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Created sample data with {len(df)} properties")
        return df

# If running in command-line mode, use the CLI helper
if __name__ == "__main__":
    from .cli import run_cli_interface
    run_cli_interface()