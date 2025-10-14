import re
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_property_insights(properties: pd.DataFrame) -> list:
    """Format property details for presentation.
    
    Extracts and formats key fields (e.g., price, bedrooms, area, location) and returns
    a list of dictionaries with details.
    """
    insights = []
    for _, prop in properties.iterrows():
        try:
            price = float(prop['Price']) if pd.notna(prop.get('Price')) else np.nan
            price_formatted = f"AED {price:,.2f}" if pd.notna(price) else "N/A"
            
            # Extract number of bedrooms from string
            bedrooms = prop.get('Bedrooms', 'N/A')
            if isinstance(bedrooms, str):
                match = re.search(r'(\d+)', bedrooms)
                bedrooms = int(match.group(1)) if match else "N/A"
            elif pd.isna(bedrooms):
                bedrooms = "N/A"
                
            # Format area (try column "Area(Sqft)" first, fallback to "Area")
            area = prop.get('Area(Sqft)', prop.get('Area', 'N/A'))
            if isinstance(area, str):
                area_clean = re.sub(r'[^\d.]', '', area)
                area_formatted = f"{float(area_clean):,.2f} sqft" if area_clean else "N/A"
            elif pd.notna(area) and isinstance(area, (int, float)):
                area_formatted = f"{area:,.2f} sqft"
            else:
                area_formatted = "N/A"
            
            # Format location (concatenate location, city, country if available)
            location = prop.get('Location', 'N/A')
            city = prop.get('city', '')
            country = prop.get('country', '')
            full_location = location
            if city and pd.notna(city):
                full_location += f", {city}"
            if country and pd.notna(country):
                full_location += f", {country}"
            
            property_insight = {
                'title': prop.get('Title', 'Property'),
                'price': price_formatted,
                'purpose': prop.get('Purpose', 'N/A'),
                'type': prop.get('Type', 'N/A'),
                'bedrooms': bedrooms,
                'bathrooms': prop.get('Bathrooms', 'N/A'),
                'area': area_formatted,
                'location': full_location,
                'furnishing': prop.get('Furnishing', 'N/A'),
                'description': prop.get('Description', 'No description available'),
                'amenities': prop.get('Amenities', 'N/A')
            }
            insights.append(property_insight)
        except Exception as e:
            logger.error(f"Error processing property: {e}")
    return insights

def format_response_with_typing_effect(properties, user_query, query_details):
    """Format response text with a more natural, thinking-while-typing style."""
    response_parts = []
    
    # Start with an analysis of what was found
    if properties.empty:
        return "I couldn't find any properties matching your criteria. Could you please try with different preferences?"
    
    # Add introduction based on number of properties found
    num_properties = len(properties)
    if num_properties == 1:
        response_parts.append("I found 1 property that matches your criteria.")
    else:
        response_parts.append(f"I found {num_properties} properties that match your criteria.")
    
    # Add search criteria summary
    criteria_parts = []
    if query_details.get('location'):
        criteria_parts.append(f"in {query_details['location']}")
    if query_details.get('property_type'):
        criteria_parts.append(f"of type {query_details['property_type']}")
    if query_details.get('bedrooms'):
        criteria_parts.append(f"with {query_details['bedrooms']} bedrooms")
    if query_details.get('max_price'):
        criteria_parts.append(f"under {query_details['max_price']:,}")
    
    if criteria_parts:
        response_parts.append("You're looking for properties " + ", ".join(criteria_parts) + ".")
    
    # Add property details one by one
    response_parts.append("\nHere are the details:")
    
    for idx, prop in properties.iterrows():
        prop_details = []
        prop_details.append(f"\n🏠 Property {idx + 1}:")
        prop_details.append(f"- Type: {prop.get('Type', 'N/A')}")
        prop_details.append(f"- Location: {prop.get('Location', 'N/A')}")
        prop_details.append(f"- Price: {prop.get('Price', 'N/A'):,}")
        if 'Bedrooms' in prop:
            prop_details.append(f"- Bedrooms: {prop['Bedrooms']}")
        if 'Area' in prop:
            prop_details.append(f"- Area: {prop['Area']}")
        if 'Description' in prop:
            prop_details.append(f"- Description: {prop['Description']}")
        
        response_parts.append("\n".join(prop_details))
    
    # Add a follow-up question
    response_parts.append("\nWould you like more details about any of these properties?")
    
    # Join all parts with appropriate pauses
    return "\n".join(response_parts)

def get_dataset_overview(stats: dict) -> str:
    """Generate an overview of the dataset based on provided statistics."""
    try:
        overview = "📊 **Real Estate Dataset Overview:**\n\n"
        if 'purposes' in stats:
            overview += "**Properties by Purpose:**\n"
            for purpose, count in stats['purposes'].items():
                overview += f"- {purpose}: {count} properties\n"
            overview += "\n"
        if 'types' in stats:
            overview += "**Property Types Available:**\n"
            for prop_type, count in stats['types'].items():
                overview += f"- {prop_type}: {count} properties\n"
            overview += "\n"
        if 'price_min' in stats and 'price_max' in stats:
            overview += "**Price Ranges:**\n"
            overview += f"- Overall: AED {stats['price_min']:,.2f} to AED {stats['price_max']:,.2f}\n"
            overview += "\n"
        if 'top_locations' in stats:
            overview += "**Top Locations:**\n"
            for location, count in list(stats['top_locations'].items())[:5]:
                overview += f"- {location}: {count} properties\n"
            overview += "\n"
        if 'bedrooms' in stats:
            overview += "**Bedroom Distribution:**\n"
            for bedrooms, count in stats['bedrooms'].items():
                if pd.notna(bedrooms):
                    overview += f"- {bedrooms} bedrooms: {count} properties\n"
            overview += "\n"
        overview += "Would you like to search for a specific type of property or learn more about properties in a particular area?"
        return overview
    except Exception as e:
        logger.error(f"Error generating dataset overview: {e}")
        return ("I have information on various properties for rent and sale. "
                "What type of property are you looking for?")