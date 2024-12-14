import os
from dotenv import load_dotenv

# Ensure you call load_dotenv to load the variables
load_dotenv()

# Retrieve the API key from the environment
api_key = os.getenv('API_KEY')
map_key = os.getenv('MAP_KEY')

# Print to confirm it's loaded correctly
if api_key:
    print(f"API Key Loaded")
else:
    print("Error: API Key not found. Ensure .env file is configured correctly.")

if map_key:
    print(f"MAP Key Loaded")
else:
    print("Error: MAP Key not found. Ensure .env file is configured correctly.")