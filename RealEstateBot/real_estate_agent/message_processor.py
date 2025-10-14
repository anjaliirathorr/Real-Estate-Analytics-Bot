import sys
import time
import random
import logging
import pandas as pd
import re

logger = logging.getLogger(__name__)

def process_user_message_with_typing(agent, message: str) -> dict:
    """Process a user's message following the conversation flow."""
    agent.current_query = message
    agent.conversation_history.append({"role": "user", "content": message})
    
    # Handle general property search requests
    property_request_patterns = [
        r'(?:show|give|find|get)\s+(?:me\s+)?(?:some\s+)?(?:other\s+)?(?:\d+\s+)?(?:properties|houses|apartments|villas)',
        r'(?:other|more|additional|different)\s+properties',
        r'properties\s+(?:with|having|that\s+have)',
        r'see\s+(?:more|other|some)\s+properties'
    ]
    
    is_property_request = any(re.search(pattern, message.lower()) for pattern in property_request_patterns)
    
    if is_property_request:
        # Extract requirements using Gemini API
        gemini_prompt = agent._create_gemini_query_prompt(message)
        query_details = agent._call_gemini_api(gemini_prompt)
        
        if query_details:
            # Update user requirements with extracted details
            if query_details.get('property_type'):
                agent.user_requirements['property_type'] = query_details['property_type']
            if query_details.get('bedrooms'):
                agent.user_requirements['layout'] = str(query_details['bedrooms'])
            if query_details.get('min_price'):
                agent.user_requirements['budget'] = str(query_details['min_price'])
            if query_details.get('max_price') and query_details.get('min_price'):
                agent.user_requirements['budget'] = f"{query_details['min_price']} to {query_details['max_price']}"
            if query_details.get('location'):
                agent.user_requirements['location'] = query_details['location']
            if query_details.get('purpose'):
                agent.current_transaction = 'buy' if query_details['purpose'] == 'Sale' else 'rent'
                agent.user_requirements['transaction_type'] = agent.current_transaction
            
            logger.info(f"Updated user requirements: {agent.user_requirements}")
        
        # Search for properties with the requirements
        response = search_matching_properties(agent)
        agent.conversation_history.append({"role": "assistant", "content": response})
        return {"query_details": None, "response": response}
    
    # Handle initial greeting
    if agent._is_greeting(message):
        agent.conversation_state = "initial"
        greeting = "Hi! 👋 Let me help you find your perfect property.\n\nWhat do you want to do? (Buy/Rent)"
        agent.conversation_history.append({"role": "assistant", "content": greeting})
        return {"query_details": None, "response": greeting}
    
    # Handle property detail requests with more flexible pattern matching
    property_patterns = [
        r'(?:show|give|tell).*(?:details|more|info).*property\s*(\d+)',
        r'property\s*(\d+)\s*(?:details|info)',
        r'more\s*(?:about|on)\s*property\s*(\d+)'
    ]
    
    for pattern in property_patterns:
        match = re.search(pattern, message.lower())
        if match:
            try:
                property_num = int(match.group(1))
                if hasattr(agent, 'last_shown_properties') and agent.last_shown_properties is not None:
                    detailed_info = get_detailed_property_info(agent, property_num)
                    agent.conversation_history.append({"role": "assistant", "content": detailed_info})
                    return {"query_details": None, "response": detailed_info}
                else:
                    return {"query_details": None, "response": "I don't have any properties in context. Let's start your search first."}
            except (AttributeError, ValueError) as e:
                logger.error(f"Error processing property request: {e}")
                return {"query_details": None, "response": "Could you specify which property number you're interested in?"}
    
    # Check if user is responding to missing information request
    if agent.conversation_history and len(agent.conversation_history) >= 2:
        last_bot_msg = agent.conversation_history[-2]['content'] if agent.conversation_history[-2]['role'] == 'assistant' else ""
        if "I need a bit more information" in last_bot_msg and any(x in message.lower() for x in ["looking for", "bedroom", "budget", "location"]):
            # Extract requirements using Gemini API
            gemini_prompt = agent._create_gemini_query_prompt(message)
            query_details = agent._call_gemini_api(gemini_prompt)
            
            if query_details:
                # Update user requirements with extracted details
                if query_details.get('property_type'):
                    agent.user_requirements['property_type'] = query_details['property_type']
                if query_details.get('bedrooms'):
                    agent.user_requirements['layout'] = str(query_details['bedrooms'])
                if query_details.get('min_price'):
                    agent.user_requirements['budget'] = str(query_details['min_price'])
                if query_details.get('max_price') and query_details.get('min_price'):
                    agent.user_requirements['budget'] = f"{query_details['min_price']} to {query_details['max_price']}"
                if query_details.get('location'):
                    agent.user_requirements['location'] = query_details['location']
                if query_details.get('purpose'):
                    agent.current_transaction = 'buy' if query_details['purpose'] == 'Sale' else 'rent'
                    agent.user_requirements['transaction_type'] = agent.current_transaction
                
                logger.info(f"Updated user requirements: {agent.user_requirements}")
                # Search for properties with these requirements
                response = search_matching_properties(agent)
                agent.conversation_history.append({"role": "assistant", "content": response})
                return {"query_details": None, "response": response}
    
    # Handle property follow-up actions if we have a current property
    if hasattr(agent, 'current_property') and agent.current_property is not None:
        follow_up_response = handle_property_follow_up(agent, message)
        if follow_up_response:
            agent.conversation_history.append({"role": "assistant", "content": follow_up_response})
            return {"query_details": None, "response": follow_up_response}
    
    # Process based on conversation state
    if agent.conversation_state == "initial":
        if "buy" in message.lower():
            agent.current_transaction = "buy"
            agent.user_requirements['transaction_type'] = 'buy'
            agent.conversation_state = "transaction_type"
            response = "What type of user best describes you? (End User/Investor)"
        elif "rent" in message.lower():
            agent.current_transaction = "rent"
            agent.user_requirements['transaction_type'] = 'rent'
            agent.conversation_state = "transaction_type"
            response = "Are you looking to rent short term or long term?"
        else:
            # Try to extract requirements using Gemini API
            gemini_prompt = agent._create_gemini_query_prompt(message)
            query_details = agent._call_gemini_api(gemini_prompt)
            
            if query_details and query_details.get('purpose'):
                if query_details['purpose'] == 'Sale':
                    agent.current_transaction = "buy"
                    agent.user_requirements['transaction_type'] = 'buy'
                    agent.conversation_state = "transaction_type"
                    response = "What type of user best describes you? (End User/Investor)"
                else:
                    agent.current_transaction = "rent"
                    agent.user_requirements['transaction_type'] = 'rent'
                    agent.conversation_state = "transaction_type"
                    response = "Are you looking to rent short term or long term?"
            else:
                response = "Please specify if you want to Buy or Rent."
        
        agent.conversation_history.append({"role": "assistant", "content": response})
        return {"query_details": None, "response": response}
    
    # Handle other states in the conversation flow
    current_flow = agent.conversation_flow.get(agent.conversation_state)
    if current_flow:
        # Update user requirements based on the response
        agent.user_requirements[agent.conversation_state] = message
        
        # Move to next state
        next_state = current_flow.get('next')
        if next_state:
            agent.conversation_state = next_state
            next_question = agent.conversation_flow[next_state]['question']
            
            # Format the accumulated requirements
            requirements_summary = format_requirements_summary(agent.user_requirements)
            response = f"{requirements_summary}\n\n{next_question}"
        else:
            # We've reached the end of the flow, show matching properties
            response = search_matching_properties(agent)
        
        agent.conversation_history.append({"role": "assistant", "content": response})
        return {"query_details": None, "response": response}
    
    # For any message, try using Gemini API to extract structured data
    gemini_prompt = agent._create_gemini_query_prompt(message)
    query_details = agent._call_gemini_api(gemini_prompt)
    
    if query_details:
        # Update user preferences and requirements on the agent instance
        for key, value in query_details.items():
            if value is not None and value != []:
                agent.user_preferences[key] = value
                
                # Also update user_requirements with the extracted data
                if key == 'property_type':
                    agent.user_requirements['property_type'] = value
                elif key == 'bedrooms':
                    agent.user_requirements['layout'] = str(value)
                elif key == 'min_price':
                    if key == 'max_price' and query_details.get('max_price'):
                        agent.user_requirements['budget'] = f"{value} to {query_details['max_price']}"
                    else:
                        agent.user_requirements['budget'] = str(value)
                elif key == 'location':
                    agent.user_requirements['location'] = value
                elif key == 'purpose':
                    agent.current_transaction = 'buy' if value == 'Sale' else 'rent'
                    agent.user_requirements['transaction_type'] = agent.current_transaction
        
        logger.info(f"Extracted query details: {query_details}")
        logger.info(f"Updated user requirements: {agent.user_requirements}")
        
        # Now, try to search for properties with the updated requirements
        if any(agent.user_requirements.values()):
            response = search_matching_properties(agent)
            agent.conversation_history.append({"role": "assistant", "content": response})
            return {"query_details": query_details, "response": response}
        
        return {"query_details": query_details, "response": None}
    else:
        fallback = ("I'm not quite sure what you're looking for. Could you provide more details about the type of property, "
                    "location, or price range you're interested in?")
        agent.conversation_history.append({"role": "assistant", "content": fallback})
        return {"query_details": None, "response": fallback}

def respond_with_typing(agent, user_message: str) -> dict:
    """Determine an initial acknowledgment and then process the full message using typing effect."""
    # Choose an acknowledgment based on keywords in the message.
    lower_msg = user_message.lower()
    if agent._is_greeting(user_message):
        # acknowledgment = "Hi there! 👋"
        acknowledgment = ""
    elif "help" in lower_msg or "looking for" in lower_msg:
        acknowledgment = "I'll help you find that! 🔍"
    elif "available" in lower_msg or "show me" in lower_msg:
        acknowledgment = "Let me check what's available..."
    elif "price" in lower_msg or "cost" in lower_msg:
        acknowledgment = "Checking prices for you..."
    elif "location" in lower_msg or "area" in lower_msg:
        acknowledgment = "Looking at locations for you..."
    else:
        acknowledgment = "Processing your request..."
    
    # Process the user's message and retrieve query details
    result = process_user_message_with_typing(agent, user_message)
    query_details = result.get("query_details")
    
    # If we already have a response from the processing, use it directly
    if result.get("response"):
        full_response = result.get("response")
    # If query details were extracted, we then filter properties and produce the full formatted response
    elif query_details is not None:
        filtered_properties = agent._filter_properties(query_details)
        
        # Extract number of properties requested from user message
        num_properties = None
        match = re.search(r'(\d+)\s+(?:properties|houses|apartments|villas)', user_message.lower())
        if match:
            num_properties = int(match.group(1))
            
        relevant_properties = agent._semantic_search(filtered_properties, user_message, top_k=num_properties if num_properties else None)
        full_response = agent._format_response_with_typing_effect(relevant_properties, user_message, query_details)
        
        # Store the properties for future reference
        agent.last_shown_properties = relevant_properties
        agent.property_display_numbers = {i+1: idx for i, idx in enumerate(relevant_properties.index)}
    else:
        full_response = "I'm still learning how to understand your requirements. Could you please provide more specific details about what you're looking for?"
    
    # Add to conversation history and return both acknowledgment and full response.
    agent.conversation_history.append({"role": "assistant", "content": full_response})
    return {"acknowledgment": acknowledgment, "full_response": full_response}

def demo_typing_response(agent, user_message: str) -> str:
    """Simulate a typing effect with more natural pauses and brief responses."""
    response_data = respond_with_typing(agent, user_message)
    
    print(f"\nAssistant: ", end='', flush=True)
    
    full_response = response_data['full_response']
    sentences = full_response.split('\n')
    
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            print(word, end=' ', flush=True)
            delay = min(len(word) * 0.05 + random.uniform(0.05, 0.15), 0.3)
            time.sleep(delay)
        print()  # New line after each sentence
        time.sleep(0.5)  # Pause between sentences
    
    return full_response

def format_requirements_summary(requirements):
    """Format the accumulated requirements into a natural sentence."""
    summary = "I am a client looking to "
    
    if requirements.get('transaction_type') == 'buy':
        summary += "buy "
    else:
        summary += "rent "
    
    if requirements.get('ready_status'):
        summary += f"{requirements['ready_status']} "
    
    if requirements.get('property_type'):
        summary += f"{requirements['property_type']} "
    
    if requirements.get('layout'):
        summary += f"with {requirements['layout']} layout "
    
    if requirements.get('budget'):
        summary += f"minimum budget of {requirements['budget']} AED "
    
    if requirements.get('location'):
        summary += f"in {requirements['location']}"
    
    return summary.strip()

def search_matching_properties(agent):
    """Search for properties matching the user's requirements."""
    logger.info(f"Searching for properties with requirements: {agent.user_requirements}")
    logger.info(f"Current transaction type: {agent.current_transaction}")
    
    # Check if we have a specific direct search request
    specific_request = {}
    message = agent.current_query.lower()
    
    # Extract specific requests for property types
    property_types = ['apartment', 'villa', 'townhouse', 'penthouse']
    for prop_type in property_types:
        if prop_type in message:
            specific_request['property_type'] = prop_type
            logger.info(f"Found specific property type request: {prop_type}")
    
    # Extract bedroom requests
    bedroom_match = re.search(r'(\d+)\s*(?:bed|bedroom|br)', message)
    if bedroom_match:
        specific_request['bedrooms'] = bedroom_match.group(1)
        logger.info(f"Found specific bedroom request: {specific_request['bedrooms']}")
    
    # Extract budget/price range
    price_range_match = re.search(r'(?:budget|price|cost|between|range).*?(\d[\d,.]*k?m?)(?:\s*(?:to|-|and)\s*(\d[\d,.]*k?m?))?', message, re.IGNORECASE)
    if price_range_match:
        # Extract the matched prices
        min_price_str = price_range_match.group(1)
        max_price_str = price_range_match.group(2) if price_range_match.group(2) else None
        
        # Convert to numeric values (handle k, m suffixes)
        def convert_price(price_str):
            if not price_str:
                return None
            
            # Remove commas and other non-alphanumeric characters except 'k' and 'm'
            price_str = re.sub(r'[^\d.km]', '', price_str.lower())
            
            # Handle 'k' (thousands) and 'm' (millions) suffixes
            if 'k' in price_str:
                price_str = price_str.replace('k', '')
                multiplier = 1000
            elif 'm' in price_str:
                price_str = price_str.replace('m', '')
                multiplier = 1000000
            else:
                multiplier = 1
            
            try:
                return float(price_str) * multiplier
            except ValueError:
                logger.error(f"Could not convert price: {price_str}")
                return None
        
        min_price = convert_price(min_price_str)
        max_price = convert_price(max_price_str)
        
        # If we only have one price, assume it's the max price
        if min_price is not None and max_price is None:
            # Check for "under", "less than", "up to" phrases
            if any(phrase in message for phrase in ["under", "less than", "up to", "below", "not more than"]):
                max_price = min_price
                min_price = None
                logger.info(f"Interpreted as max price: {max_price}")
            # Check for "over", "more than", "at least" phrases
            elif any(phrase in message for phrase in ["over", "more than", "at least", "above", "not less than"]):
                # min_price already set correctly
                logger.info(f"Interpreted as min price: {min_price}")
            else:
                # By default, assume it's the max price
                max_price = min_price
                min_price = None
                logger.info(f"Defaulting to max price: {max_price}")
        
        if min_price is not None:
            specific_request['min_price'] = min_price
            logger.info(f"Found min price: {min_price}")
        if max_price is not None:
            specific_request['max_price'] = max_price
            logger.info(f"Found max price: {max_price}")
    
    # Extract location
    locations = ["dubai", "abu dhabi", "sharjah", "ajman", "palm jumeirah", "downtown dubai", "dubai marina", 
                "business bay", "jumeirah", "jumeirah beach residence", "jbr", "deira", "al barsha",
                "al quoz", "arabian ranches", "motor city", "jumeirah village circle", "jvc",
                "dubai sports city", "international city", "dubai silicon oasis", "dubai hills",
                "dubai land", "emirates hills", "the springs", "the greens", "al nahda", "mirdif"]
    
    for location in locations:
        if location in message.lower():
            specific_request['location'] = location
            logger.info(f"Found specific location: {location}")
            break
    
    # Extract purpose (buy/rent)
    if 'buy' in message or 'purchase' in message:
        specific_request['purpose'] = 'Sale'
    elif 'rent' in message or 'lease' in message:
        specific_request['purpose'] = 'Rent'
    else:
        # Use transaction type from agent state
        specific_request['purpose'] = 'Sale' if agent.current_transaction == 'buy' else 'Rent' if agent.current_transaction == 'rent' else None
    
    # Flag to track if user is requesting all properties (override filters)
    request_all = False
    if agent.current_query:
        if any(phrase in agent.current_query.lower() for phrase in ["all properties", "any property", "all of them", "any properties", "give me property"]):
            request_all = True
            logger.info("User requested all properties, skipping detailed filtering")
            
    # Extract number of properties requested from user message or specific request
    num_properties = None
    if 'count' in specific_request:
        num_properties = specific_request['count']
    else:
        message = agent.current_query.lower()
        match = re.search(r'(\d+)\s+(?:properties|houses|apartments|villas)', message)
        if match:
            num_properties = int(match.group(1))

    # Convert user requirements to filters format
    filters = {}
    
    # Use specific request details if available, otherwise use agent's stored user requirements
    if specific_request.get('property_type'):
        filters['property_type'] = specific_request['property_type']
    elif agent.user_requirements.get('property_type'):
        filters['property_type'] = agent.user_requirements['property_type']
    
    if specific_request.get('bedrooms'):
        filters['bedrooms'] = specific_request['bedrooms']
    elif agent.user_requirements.get('layout'):
        filters['bedrooms'] = agent.user_requirements['layout']
    
    if specific_request.get('location'):
        filters['location'] = specific_request['location']
    elif agent.user_requirements.get('location'):
        filters['location'] = agent.user_requirements['location']
    
    if specific_request.get('min_price'):
        filters['min_price'] = specific_request['min_price']
    elif agent.user_requirements.get('budget_min'):
        filters['min_price'] = agent.user_requirements['budget_min']
    
    if specific_request.get('max_price'):
        filters['max_price'] = specific_request['max_price']
    elif agent.user_requirements.get('budget'):
        filters['max_price'] = agent.user_requirements['budget']
    
    # Set the required property count
    required_count = num_properties if num_properties else 5  # Default to 5 properties if not specified

    # Log the filtering criteria
    logger.info(f"Filtering with criteria: {filters}, requested count: {required_count}")
    
    # If user requested all properties, don't apply filters
    if request_all:
        filtered_properties = agent.knowledge_base.copy()
    else:
        # Apply filtering with user requirements
        filtered_properties = agent._filter_properties(agent.knowledge_base, required_property_count=required_count, **filters)
    
    # Check if we found any properties
    if filtered_properties is None or len(filtered_properties) == 0:
        logger.warning("No properties found after primary filtering")
        # Fallback to returning some random properties
        filtered_properties = agent.knowledge_base.sample(min(10, len(agent.knowledge_base)))
        logger.info(f"Using random fallback, returning {len(filtered_properties)} properties")

    # Reset property display numbers
    agent.property_display_numbers = {}
    agent.last_shown_properties = filtered_properties
    
    # Create display numbers for filtered properties
    display_numbers = {}
    for i, (index, property_data) in enumerate(filtered_properties.iterrows(), 1):
        display_numbers[i] = index
    
    agent.property_display_numbers = display_numbers
    logger.info(f"Set property display numbers: {display_numbers}")
    
    # Present the properties to the user
    response = format_property_list(agent, filtered_properties)
    
    # If no properties were found at all, apologize to the user
    if len(filtered_properties) == 0:
        response = (
            "I apologize, but I couldn't find any properties matching your criteria. "
            "Would you like to broaden your search by adjusting your requirements?"
        )
    
    # Add some context and follow-up options
    if len(filtered_properties) > 0:
        response += (
            "\n\nYou can ask for more details about any property by saying 'Tell me more about property #X'."
            "\nOr you can refine your search by specifying more criteria."
        )
    
    logger.info(f"Returning {len(filtered_properties)} properties to user")
    return response

def get_detailed_property_info(agent, property_num):
    """Get detailed information for a specific property."""
    try:
        # Validate property number
        if not hasattr(agent, 'property_display_numbers') or property_num not in agent.property_display_numbers:
            return f"I couldn't find property {property_num}. Please specify a valid property number from the list above."

        # Get the actual property data
        actual_index = agent.property_display_numbers[property_num]
        property_data = agent.last_shown_properties.loc[actual_index]
        
        # Convert to dictionary for easier handling
        property_dict = property_data.to_dict() if hasattr(property_data, 'to_dict') else property_data
        
        # Format price properly
        price = property_dict.get('Price')
        price_formatted = "Price not specified"
        
        if price is not None and price != '':
            try:
                if isinstance(price, str):
                    # Try to extract numeric value from string
                    price_str = re.sub(r'[^\d.]', '', price)
                    if price_str:
                        price = float(price_str)
                    else:
                        price = None
                
                if price is not None:
                    # Check if the property is for sale
                    is_for_sale = False
                    if 'Purpose' in property_dict and property_dict['Purpose'] is not None:
                        is_for_sale = str(property_dict['Purpose']).lower() in ['sale', 'buy', 'purchase']
                    
                    # Format based on magnitude and property type
                    if price >= 1000000:
                        price_formatted = f"AED {price/1000000:.2f}M"
                    elif price >= 100000:
                        if is_for_sale:
                            price_formatted = f"AED {price/1000:.1f}K"
                        else:
                            price_formatted = f"AED {price/1000:.1f}K/year"
                    else:
                        price_formatted = f"AED {price:,.2f}"
            except (ValueError, TypeError) as e:
                logger.error(f"Error formatting price: {e}")
                price_formatted = "Price not specified"
        
        # Format bedrooms
        bedrooms = property_dict.get('Bedrooms', 'N/A')
        bedrooms_formatted = "N/A"
        
        if bedrooms is not None and bedrooms != 'N/A' and bedrooms != '':
            try:
                if isinstance(bedrooms, str):
                    # Try to extract numeric value from string
                    bedroom_match = re.search(r'(\d+)', bedrooms)
                    if bedroom_match:
                        bedrooms = int(bedroom_match.group(1))
                    else:
                        bedrooms = None
                
                if bedrooms is not None:
                    bedrooms_formatted = f"{int(bedrooms)} bedroom{'s' if bedrooms != 1 else ''}"
            except (ValueError, TypeError) as e:
                logger.error(f"Error formatting bedrooms: {e}")
                bedrooms_formatted = "N/A"

        # Format bathrooms
        bathrooms = property_dict.get('Bathrooms', 'N/A')
        bathrooms_formatted = "N/A"
        
        if bathrooms is not None and bathrooms != 'N/A' and bathrooms != '':
            try:
                if isinstance(bathrooms, str):
                    bathroom_match = re.search(r'(\d+)', bathrooms)
                    if bathroom_match:
                        bathrooms = int(bathroom_match.group(1))
                    else:
                        bathrooms = None
                
                if bathrooms is not None:
                    bathrooms_formatted = f"{int(bathrooms)} bathroom{'s' if bathrooms != 1 else ''}"
            except (ValueError, TypeError) as e:
                logger.error(f"Error formatting bathrooms: {e}")
                bathrooms_formatted = "N/A"
                
        # Format area
        area = property_dict.get('Area', 'N/A')
        area_formatted = "N/A"
        
        if area is not None and area != 'N/A' and area != '':
            try:
                if isinstance(area, str):
                    area_match = re.search(r'(\d+(?:\.\d+)?)', area)
                    if area_match:
                        area = float(area_match.group(1))
                    else:
                        area = None
                
                if area is not None:
                    area_formatted = f"{area:,.2f} sqft"
            except (ValueError, TypeError) as e:
                logger.error(f"Error formatting area: {e}")
                area_formatted = "N/A"

        # Format detailed response
        details = [
            f"📍 Detailed information for Property {property_num}:",
            "",
            f"Location: {property_dict.get('Location', 'N/A')}",
            f"Type: {property_dict.get('Type', 'N/A')}",
            f"Price: {price_formatted}",
            f"Bedrooms: {bedrooms_formatted}",
            f"Bathrooms: {bathrooms_formatted}",
            f"Area: {area_formatted}",
            ""
        ]

        # Add purpose/transaction type if available
        purpose = property_dict.get('Purpose', 'N/A')
        if purpose != 'N/A':
            details.insert(3, f"Purpose: {purpose}")

        # Add description
        description = property_dict.get('Description', 'No description available')
        if description != 'No description available':
            details.append(f"Description: {description}")
            details.append("")
        
        # Format amenities properly
        amenities = property_dict.get('Amenities', 'N/A')
        if pd.notna(amenities) and amenities != 'N/A':
            if isinstance(amenities, str):
                # Split by comma and clean up
                amenity_list = [a.strip() for a in amenities.split(',')]
                details.append("Amenities:")
                for amenity in amenity_list:
                    details.append(f"- {amenity}")
            else:
                details.append(f"Amenities: {amenities}")
        else:
            details.append("Amenities: Not specified")

        details.extend([
            "",
            "Would you like to:",
            "1. Schedule a viewing",
            "2. Ask more questions about this property",
            "3. See other similar properties"
        ])
        
        # Store the current property data for follow-up actions
        agent.current_property = property_data
        
        return "\n".join(details)
    except Exception as e:
        logger.error(f"Error getting detailed property info: {e}")
        logger.exception("Stack trace:")
        return "I encountered an error while retrieving the property details. Please try again."

def handle_property_follow_up(agent, message: str) -> str:
    """Handle follow-up actions for a property."""
    if not hasattr(agent, 'current_property') or agent.current_property is None:
        return "I don't have any property details in context. Please select a property first."

    message = message.lower()
    
    # Handle schedule viewing request
    if any(word in message for word in ['schedule', 'viewing', 'visit', 'tour']):
        whatsapp = agent.current_property.get('Agent Whatsapp No.', 'N/A')
        if pd.notna(whatsapp) and whatsapp != 'N/A':
            return f"I'd be happy to schedule a viewing for you! You can contact the property agent directly at +{whatsapp} to arrange a viewing time that works for you."
        else:
            return "I apologize, but I don't have the agent's contact information for this property. Would you like to see other similar properties instead?"

    # Handle questions about the property
    elif any(word in message for word in ['question', 'ask', 'tell me', 'what about', 'how about']):
        # Extract the question from the message
        question = message.replace('question', '').replace('ask', '').replace('tell me', '').replace('what about', '').replace('how about', '').strip()
        
        # Try to find relevant information in the property data
        if any(word in question for word in ['price', 'cost']):
            price = agent.current_property.get('Price', 'N/A')
            return f"The price of this property is AED {price:,}."
        elif any(word in question for word in ['location', 'where', 'address']):
            location = agent.current_property.get('Location', 'N/A')
            return f"This property is located in {location}."
        elif any(word in question for word in ['bedroom', 'room']):
            bedrooms = agent.current_property.get('Bedrooms', 'N/A')
            return f"This property has {bedrooms} bedrooms."
        elif any(word in question for word in ['bathroom', 'bath']):
            bathrooms = agent.current_property.get('Bathrooms', 'N/A')
            return f"This property has {bathrooms} bathrooms."
        elif any(word in question for word in ['area', 'size', 'square']):
            area = agent.current_property.get('Area', 'N/A')
            return f"The area of this property is {area}."
        elif any(word in question for word in ['amenity', 'feature', 'facility']):
            amenities = agent.current_property.get('Amenities', 'N/A')
            if pd.notna(amenities) and amenities != 'N/A':
                return f"This property has the following amenities: {amenities}"
            else:
                return "No specific amenities are listed for this property."
        else:
            return "I'm not sure I understand your question about this property. Could you please be more specific?"

    # Handle similar properties request
    elif any(word in message for word in ['similar', 'other', 'more', 'like this']):
        # Get current property details for similarity search
        current_type = agent.current_property.get('Type', '')
        current_location = agent.current_property.get('Location', '')
        current_bedrooms = agent.current_property.get('Bedrooms', '')
        
        # Create a query for similar properties
        query = f"{current_type} in {current_location} with {current_bedrooms} bedrooms"
        
        # Filter and search for similar properties
        filtered_properties = agent._filter_properties({
            'property_type': current_type,
            'location': current_location,
            'bedrooms': current_bedrooms
        })
        
        # Exclude the current property
        filtered_properties = filtered_properties[filtered_properties.index != agent.current_property.name]
        
        if len(filtered_properties) > 0:
            # Get top 3 similar properties
            similar_properties = agent._semantic_search(filtered_properties, query, top_k=3)
            
            # Update the last shown properties
            agent.last_shown_properties = similar_properties
            agent.property_display_numbers = {i+1: idx for i, idx in enumerate(similar_properties.index)}
            
            response = [
                "Here are some similar properties you might be interested in:",
                ""
            ]
            
            for i, (idx, prop) in enumerate(similar_properties.iterrows(), 1):
                price = f"{prop.get('Price', 'N/A'):,}" if pd.notna(prop.get('Price')) else 'N/A'
                response.extend([
                    f"🏠 Property {i}:",
                    f"- {prop.get('Type', 'N/A')} in {prop.get('Location', 'N/A')}",
                    f"- Price: AED {price}",
                    f"- {prop.get('Bedrooms', 'N/A')} bedrooms",
                    ""
                ])
            
            response.append("To see more details about any of these properties, just say 'show me details of property X' (where X is the property number).")
            return "\n".join(response)
        else:
            return "I couldn't find any similar properties at the moment. Would you like to adjust your search criteria?"

    return "I'm not sure what you'd like to do with this property. You can schedule a viewing, ask questions about it, or see similar properties."

def format_property_list(agent, properties):
    """Format a list of properties for display in a consistent way."""
    if properties.empty:
        return "No properties found."
    
    formatted_properties = []
    
    # Add property summaries with consistent numbering
    for i, (idx, prop) in enumerate(properties.iterrows(), 1):
        agent.property_display_numbers[i] = idx
        
        # Ensure the property has all required fields
        prop_dict = prop.to_dict()
        for field in ['Type', 'Location', 'Price', 'Bedrooms']:
            if field not in prop_dict or pd.isna(prop_dict[field]):
                prop_dict[field] = 'N/A'
        
        # Format price properly
        price = prop_dict.get('Price')
        price_formatted = "Price not specified"
        
        if price is not None and price != 'N/A' and price != '':
            try:
                if isinstance(price, str):
                    # Try to extract numeric value from string
                    price_str = re.sub(r'[^\d.]', '', price)
                    if price_str:
                        price = float(price_str)
                    else:
                        price = None
                
                if price is not None:
                    # Format based on magnitude (checking if this is a real estate price in AED)
                    # For Dubai property prices in AED, values like 185,000 are likely in AED per year (rent),
                    # while values like 2,500,000 are purchase prices. 
                    
                    # Check if the property is for sale
                    is_for_sale = False
                    if 'Purpose' in prop_dict and prop_dict['Purpose'] is not None:
                        is_for_sale = str(prop_dict['Purpose']).lower() in ['sale', 'buy', 'purchase']
                    
                    # For properties with prices between 80K-500K that are for rent
                    if price < 500000 and not is_for_sale and price > 80000:
                        price_formatted = f"AED {price/1000:.1f}K/year"
                    # For prices over 1M (typical purchase prices)
                    elif price >= 1000000:
                        price_formatted = f"AED {price/1000000:.2f}M"
                    # For prices below 1M but over 100K
                    elif price >= 100000:
                        price_formatted = f"AED {price/1000:.1f}K"
                    else:
                        price_formatted = f"AED {price:,.2f}"
            except (ValueError, TypeError) as e:
                logger.error(f"Error formatting price {price}: {e}")
                price_formatted = "Price not specified"
        
        # Format bedrooms
        bedrooms = prop_dict.get('Bedrooms')
        bedrooms_formatted = "N/A"
        
        if bedrooms is not None and bedrooms != 'N/A' and bedrooms != '':
            try:
                if isinstance(bedrooms, str):
                    # Try to extract numeric value from string
                    bedroom_match = re.search(r'(\d+)', bedrooms)
                    if bedroom_match:
                        bedrooms = int(bedroom_match.group(1))
                    else:
                        bedrooms = None
                
                if bedrooms is not None:
                    bedrooms_formatted = f"{int(bedrooms)} bedroom{'s' if bedrooms != 1 else ''}"
            except (ValueError, TypeError) as e:
                logger.error(f"Error formatting bedrooms {bedrooms}: {e}")
                bedrooms_formatted = "N/A bedrooms"
        
        # Get type and location
        prop_type = prop_dict.get('Type', 'N/A')
        location = prop_dict.get('Location', 'N/A')
        
        # Format a summary
        summary = [
            f"🏠 Property {i}:",
            f"- {prop_type} in {location}",
            f"- Price: {price_formatted}",
            f"- {bedrooms_formatted}",
            ""
        ]
        formatted_properties.extend(summary)
    
    # Add navigation help
    formatted_properties.append("To see more details about a specific property, just say 'show me details of property X' (where X is the property number).")
    formatted_properties.append("Or we can refine your search if these don't match your preferences.")
    
    return "\n".join(formatted_properties)
