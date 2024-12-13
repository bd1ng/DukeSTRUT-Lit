# %% Imports
import streamlit as st
import google.generativeai as genai
import googlemaps
import os
from dotenv import load_dotenv
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import json
import networkx as nx
from networkx.readwrite import json_graph
from pyproj import Transformer 
import rasterio
import rasterio
from pyproj import Transformer
import numpy as np
from shapely.affinity import translate
from shapely.geometry import Point
from shapely.geometry import LineString, Polygon, MultiLineString
from shapely.geometry import Point
from shapely.ops import unary_union
import pvlib
from datetime import datetime
import pytz
import math
import requests
from sklearn.linear_model import LinearRegression
import joblib

# %% Load Necessary Files
veg_path = "Campus_Vegetation.gdb"
feat_path = "Campus_Features.gdb"
lin_model_reg = joblib.load('linear_regression_model.pkl')
with rasterio.open("lidar_dsm.tif") as dsm_src, rasterio.open("dem_resampled.tif") as dem_src:
    dsm_data = dsm_src.read(1)  # Read DSM
    dem_data = dem_src.read(1)  # Read DEM
    dsm_transform = dsm_src.transform
    dem_transform = dem_src.transform
with open("graph.json", "r") as f:
    graph_data = json.load(f)
graph = json_graph.node_link_graph(graph_data)

# %% GEO Setup
base = gpd.read_file(feat_path, layer="BaseFeatures") 
base = base[base['BASE_SUBTYPE'].isin([2,4,6, 9, 10,11, 13])]
paths = gpd.read_file(feat_path, layer="Street_centerline")  # Paths for routing
paths = paths[paths.geometry.type.isin(["LineString", "MultiLineString"])]
base = base[base.geometry.type.isin(["LineString", "MultiLineString"])]
paths = paths.explode(index_parts=False).reset_index(drop=True)
base = base.explode(index_parts=False).reset_index(drop=True)
common_crs = "EPSG:32617" 
layers = [paths, base]

for idx, layer in enumerate(layers):
    if layer.crs != common_crs:
        layers[idx] = layer.to_crs(common_crs)

paths, base = layers

paths = gpd.GeoDataFrame(pd.concat([paths, base], ignore_index=True), crs=common_crs)
paths = paths[~paths.geometry.is_empty].reset_index(drop=True)

paths['buffered_geometry'] = paths.geometry.buffer(5)
paths.set_geometry('buffered_geometry', inplace=True)
paths.head()

try:
    pos = {node: (node[0], node[1]) for node in graph.nodes}
except (KeyError, TypeError):
    pos = nx.spring_layout(graph)

edges = [[pos[u], pos[v]] for u, v in graph.edges]
nodes = list(graph.nodes)
node_positions = [pos[node] for node in nodes]

buildings = gpd.read_file(feat_path, layer="Buildings") 
con_trees = gpd.read_file(veg_path, layer="ConTree") 
dec_trees = gpd.read_file(veg_path, layer="DecTree") 

if buildings.crs != "EPSG:32617":
    buildings = buildings.to_crs("EPSG:32617")

def extract_elevation(geom, raster_data, transform):
    if geom.is_empty or not isinstance(geom, Point):
        return np.nan

    x, y = geom.x, geom.y
    
    row, col = ~transform * (x, y)  # Transform to row/col indices
    row, col = int(row), int(col)   # Ensure integer indices
    
    try:
        return raster_data[row, col]  # Get the elevation value
    except IndexError:
        return np.nan  # If out of bounds, return NaN

buildings["centroid"] = buildings.geometry.centroid
buildings = buildings[~buildings["centroid"].isna()]

buildings["dsm_elevation"] = buildings["centroid"].apply(lambda geom: extract_elevation(geom, dsm_data, dsm_transform))
buildings["dem_elevation"] = buildings["centroid"].apply(lambda geom: extract_elevation(geom, dem_data, dem_transform))

buildings["net_height"] = buildings["dsm_elevation"] - buildings["dem_elevation"]
buildings = buildings.drop(columns=["centroid"])

def extract_elevation(geom, raster_data, transform):
    if geom.is_empty or not isinstance(geom, Point):
        return np.nan
    x, y = geom.x, geom.y
    row, col = ~transform * (x, y) 
    row, col = int(row), int(col) 
    
    try:
        return raster_data[row, col]
    except IndexError:
        return np.nan 

if con_trees.crs != "EPSG:32617":
    con_trees = con_trees.to_crs("EPSG:32617")
if dec_trees.crs != "EPSG:32617":
    dec_trees = dec_trees.to_crs("EPSG:32617")

con_trees["dsm_elevation"] = con_trees.geometry.apply(lambda geom: extract_elevation(geom, dsm_data, dsm_transform))
con_trees["dem_elevation"] = con_trees.geometry.apply(lambda geom: extract_elevation(geom, dem_data, dem_transform))
con_trees["net_height"] = con_trees["dsm_elevation"] - con_trees["dem_elevation"]
dec_trees["dsm_elevation"] = dec_trees.geometry.apply(lambda geom: extract_elevation(geom, dsm_data, dsm_transform))
dec_trees["dem_elevation"] = dec_trees.geometry.apply(lambda geom: extract_elevation(geom, dem_data, dem_transform))
dec_trees["net_height"] = dec_trees["dsm_elevation"] - dec_trees["dem_elevation"]

for df in [con_trees, dec_trees]:
    df['net_height'] = df['net_height'].apply(
        lambda x: 10 if pd.isna(x) or x < 4 or x > 20 else x
    )

buildings['net_height'] = buildings['net_height'].apply(
    lambda x: 15 if pd.isna(x) or x < 4 or x > 28 else x
)

buildings['net_height'] = buildings.apply(
    lambda row: 0 if row['NAME'] is not None and isinstance(row['NAME'], str) and 'Quad' in row['NAME'] else row['net_height'],
    axis=1
)


# %% Ambient - Weather

def get_current_weather(lat, lon):
    # Define the API URL and parameters
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
        "forecast_days": 1
    }
    
    # Make the API request
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract hourly data
        hourly_data = data.get("hourly", {})
        times = hourly_data.get("time", [])
        temperatures = hourly_data.get("temperature_2m", [])
        humidities = hourly_data.get("relative_humidity_2m", [])
        
        # Get the current hour
        current_time = datetime.now().strftime("%Y-%m-%dT%H:00")  # Format: 2024-12-12T14:00
        
        # Find the index for the current hour
        if current_time in times:
            index = times.index(current_time)
            current_temp = temperatures[index]
            current_humidity = humidities[index]
            return {
                "time": current_time,
                "temperature": current_temp,
                "humidity": current_humidity
            }
        else:
            with placeholder:
                st.warning("Cannot fetch hour data.")
    else:
        with placeholder:
            st.warning("Cannot fetch weather data.")

latitude = 36.001465  
longitude = -78.939133 
weather_data = get_current_weather(latitude, longitude)

if weather_data:
    # Extract temperature and humidity into variables
    temperature = weather_data["temperature"]
    humidity = weather_data["humidity"]

    # Use the variables
    print(f"Temperature: {temperature}°F")
    print(f"Humidity: {humidity}%")
else:
    st.warning("Failed to retrieve weather data.")

x_test = np.array([[weather_data['temperature'], weather_data['humidity']]])
y_pred = lin_model_reg.predict(x_test)
y_pred = y_pred[0]

# %% Load Models & Keys
#Keys
load_dotenv()
api_key = os.getenv('API_KEY')
map_key = os.getenv('MAP_KEY')

if not api_key:
    raise KeyError("API_KEY is not set. Ensure your .env file is loaded and correctly configured.")

genai.configure(api_key=api_key)
gmaps = googlemaps.Client(key=map_key)

#Gemini
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# %% Gemini DEF

def refine_locations(user_input):
    refinement_prompt = f"""
Interpret this input: '{user_input}' as starting and ending locations on Duke University's campus.
Refine the locations to be specific and prepend "Duke University" to both locations.
Format your response exactly as:
Start: Duke University [start location], End: Duke University [end location].
Examples:
- Input: 'From Bryan Center to Wilson Gym' -> Start: Duke University Bryan Center, End: Duke University Wilson Gym
- Input: 'From Perkins Library to Cameron Indoor Stadium' -> Start: Duke University Perkins Library, End: Duke University Cameron Indoor Stadium
"""
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(refinement_prompt)
    
    if "Start:" in response.text and "End:" in response.text:
        start = response.text.split("Start:")[1].split("End:")[0].strip()
        end = response.text.split("End:")[1].strip()
        return start, end
    else:
        with placeholder:
            st.error("This isn't what I expected, Blue Devil, please specify where you are and where you'd like to go :)")
        return None, None

def get_coordinates(location_name, retries=3):
    """
    Fetch latitude and longitude for a location using Google Maps API.
    Retry using Gemini to refine the location if default coordinates are returned.
    """
    for attempt in range(retries):
        # Fetch geocode result from Google Maps
        geocode_result = gmaps.geocode(location_name)
        
        if geocode_result:
            lat = geocode_result[0]["geometry"]["location"]["lat"]
            lng = geocode_result[0]["geometry"]["location"]["lng"]

            # Check for default coordinates
            if lat == 36.0014258 and lng == -78.9382286:
                with placeholder:
                    st.info(f"Default coordinates detected for '{location_name}'. Retrying with refinement ({attempt + 1}/{retries})...")
                
                # Refine the location using Gemini
                refinement_prompt = f"""
                The query '{location_name}' returned default coordinates for Duke University. 
                Refine the query to be more specific, ensuring it includes an unambiguous landmark on Duke's campus.
                """
                refined_location = genai.GenerativeModel("gemini-1.5-flash").generate_content(refinement_prompt).text
                
                # Update the location name and retry
                location_name = refined_location
                continue
            
            return lat, lng  # Valid coordinates found

        with placeholder:
            st.info(f"Failed to get valid coordinates for '{location_name}'. Retrying ({attempt + 1}/{retries})...")

    # If all retries fail
    with placeholder:
        st.error(f"Unable to retrieve valid coordinates for '{location_name}' after {retries} retries.")
    return None, None

# %% Streamlit Visual

#Title, Subhead
st.title("Duke STRUT")
st.subheader("Shade Tracking and Route Utility Tool")

#Color text
st.write("Duke STRUT finds you the shadiest path to where you need to go.")

placeholder = st.empty()
placeholder_map = st.empty()

colors = {
    "paths": "#4A90E2",         # Light Blue
    "buildings": "#F5A623",     # Amber
    "trees": "#7ED321",     # Lime Green
}

fig, ax = plt.subplots(figsize=(40, 20))  # Larger figure for better readability

paths.plot(
    ax=ax, 
    color=colors["paths"], 
    linewidth=1, 
    alpha=0.8, 
    label="Paths"
)

# Plot buildings
buildings.plot(
    ax=ax, 
    color=colors["buildings"], 
    alpha=0.5, 
    label="Buildings"
)

# Merge the two tree datasets
all_trees = pd.concat([con_trees, dec_trees], ignore_index=True)  # Combine coniferous and deciduous trees
all_trees.plot(
    ax=ax, 
    color=colors["trees"], 
    alpha=0.6, 
    label="Trees"
)

# Add title and labels
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)

plt.legend(title="Map Layers", loc="upper right", fontsize=10, title_fontsize=12, frameon=True, edgecolor="gray")

# Remove axes for a cleaner look
ax.axis('off')

# Show the plot
with placeholder_map:
    st.pyplot(fig)

st.sidebar.header("Get Started")
user_input = st.sidebar.text_input("Tell us where you are and where you want to go, Blue Devil!")

st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.markdown("### Current Weather Conditions:")
st.sidebar.write(f"🌤️ **Temperature:** {temperature}°F")
st.sidebar.write(f"💧 **Humidity:** {humidity}%")

st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Custom slider labels
st.sidebar.markdown("### Shade Preference:")

st.sidebar.markdown(
    """
    <div style="text-align: center; font-weight: bold; font-size: 16px;">
        Adjust your shade preference
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span>🏎️</span>
        <span>🕶️</span>
    </div>
    """,
    unsafe_allow_html=True
)

# The slider itself
shade_preference = st.sidebar.slider(
    "",
    min_value=0.0,
    max_value=1.0,
    value=y_pred,  # Default value
    step=0.01,
)

submit_button = st.sidebar.button("Let's STRUT!")

# %% Streamlit Run

if submit_button:
    if not user_input.strip():  # Check if the input field is blank or whitespace
        with placeholder:
            st.warning("Please enter a prompt before submitting.")
    else:
        try:
            # Step 1: Use Gemini to refine the input
            start_location, end_location = refine_locations(user_input)
            
            # Step 2: Use Google Maps API to fetch coordinates
            start_coords = get_coordinates(start_location)
            end_coords = get_coordinates(end_location)
            gemini_success = True
            
            # Display the results
            with placeholder:
                st.success(f"Start Coordinates: {start_coords}, End Coordinates: {end_coords}")
        except ValueError as e:
            with placeholder:
                st.warning(f"Error refining locations: {e}")
        except Exception as e:
            with placeholder:
                st.warning(f"An error occurred: {e}")
    
    if gemini_success:
        with placeholder:
            st.info("Calculating sun angles...")
        latitude = 36.0  # Duke University's latitude
        longitude = -78.9  # Duke University's longitude
        campus_altitude = 123  # Campus altitude in meters
        timezone = pytz.timezone("America/New_York")

        # Current date and time
        current_time = datetime.now(timezone)

        # Calculate solar position with altitude adjustment
        solar_position = pvlib.solarposition.get_solarposition(
            time=current_time,
            latitude=latitude,
            longitude=longitude,
            altitude=campus_altitude  # Observer's elevation above sea level
        )

        # Extract sun angles
        sun_altitude = solar_position["apparent_elevation"].iloc[0]  # Angle above horizon in degrees
        sun_azimuth = solar_position["azimuth"].iloc[0]  # Compass direction in degrees
        sun_success = True
    
    if sun_success:
        with placeholder:
            st.info("Building shadows...")
        common_crs = "EPSG:32617"
        def reproject_to_crs(layer, crs):
            if layer.crs != crs:
                return layer.to_crs(crs)
            return layer

        con_trees = reproject_to_crs(con_trees, common_crs)
        dec_trees = reproject_to_crs(dec_trees, common_crs)
        buildings = reproject_to_crs(buildings, common_crs)

        def project_shadow_tree(geometry, height, sun_altitude, sun_azimuth):

            if geometry is None or not geometry.is_valid or geometry.is_empty or sun_altitude <= 0:
                return None
            
            shadow_length = height / math.tan(math.radians(sun_altitude))
            
            dx = shadow_length * math.sin(math.radians(sun_azimuth))
            dy = shadow_length * math.cos(math.radians(sun_azimuth))
            
            try:
                shadow = translate(geometry, xoff=dx, yoff=dy)
                return shadow
            except Exception as e:
                with placeholder:
                    st.warning(f"Error projecting tree shadow: {e}")
                return None
            
        def project_shadow_building(geometry, height, sun_altitude, sun_azimuth):

            if geometry is None or not geometry.is_valid or geometry.is_empty or sun_altitude <= 0:
                return None
            
            shadow_length = height / math.tan(math.radians(sun_altitude))
            
            dx = shadow_length * math.sin(math.radians(sun_azimuth))
            dy = shadow_length * math.cos(math.radians(sun_azimuth))
            
            try:
                shadow = translate(geometry, xoff=dx, yoff=dy)
                return shadow
            except Exception as e:
                with placeholder:
                    st.warning(f"Error projecting building shadow: {e}")
                return None

        con_trees["shadow"] = con_trees.apply(
            lambda row: project_shadow_tree(
                row.geometry, 
                row.net_height,  # Height
                sun_altitude, 
                sun_azimuth
            ),
            axis=1
        )

        dec_trees["shadow"] = dec_trees.apply(
            lambda row: project_shadow_tree(
                row.geometry, 
                row.net_height,  # Height
                sun_altitude, 
                sun_azimuth
            ),
            axis=1
        )

        buildings["shadow"] = buildings.apply(
            lambda row: project_shadow_building(
                row.geometry, 
                row.net_height,  # Height
                sun_altitude, 
                sun_azimuth
            ),
            axis=1
        )

        con_trees_shadow = con_trees[con_trees["shadow"].notnull()]
        dec_trees_shadow = dec_trees[dec_trees["shadow"].notnull()]
        buildings_shadow = buildings[buildings["shadow"].notnull()]

        shadows = gpd.GeoDataFrame(
            pd.concat([
                con_trees_shadow[["shadow"]],  # Select shadow column
                dec_trees_shadow[["shadow"]],
                buildings_shadow[["shadow"]]
            ], ignore_index=True),
            geometry="shadow",  # Specify the shadow column as the geometry
            crs=common_crs  # Use the common CRS
        )
        with placeholder:
            st.info("Merging shadows...")
        all_shadows = unary_union(shadows.geometry)
        all_shadows = all_shadows.simplify(tolerance=0.1, preserve_topology=True)
        all_edges = MultiLineString([LineString([pos[u], pos[v]]) for u, v in graph.edges])
        intersections = all_shadows.intersection(all_edges)
        intersections_list = list(intersections.geoms)  # Extract individual geometries from MultiLineString
        for (u, v), intersect_geom in zip(graph.edges, intersections_list):
            # Get the original edge geometry
            edge_geom = LineString([pos[u], pos[v]])
            # Calculate intersected length
            intersected_length = intersect_geom.length if intersect_geom.is_valid else 0
            # Calculate shadow score
            shadow_score = intersected_length / edge_geom.length if edge_geom.length > 0 else 0
            # Store the shadow score in the graph
            graph[u][v]['shadow_score'] = shadow_score
        with placeholder:
            st.info("Finalizing map")
        shadow_success = True

    if shadow_success: 
        with placeholder:
            st.info("Building paths...")
        start_lat, start_lon = start_coords
        end_lat, end_lon = end_coords

        shade_preference = shade_preference

        paths_crs = paths.crs
        transformer = Transformer.from_crs("EPSG:4326", paths_crs, always_xy=True)

        start_x, start_y = transformer.transform(start_lon, start_lat)
        end_x, end_y = transformer.transform(end_lon, end_lat)

        with placeholder:
            st.info("Finding your nearest starting and end-point")
        def find_nearest_node(point, graph):
            shapely_point = Point(point)
            return min(graph.nodes, key=lambda n: shapely_point.distance(Point(n)))

        # Find the nearest nodes for the start and end points
        start_node = find_nearest_node((start_x, start_y), graph)
        end_node = find_nearest_node((end_x, end_y), graph)
        clip_success = True

    if clip_success:
        with placeholder:
            st.info("Building optimal path...")
        for u, v, data in graph.edges(data=True):
            if 'shadow_score' not in data:
                data['shadow_score'] = 0  # Default value

            if 'distance' not in data:
                data['distance'] = ((pos[u][0] - pos[v][0])**2 + (pos[u][1] - pos[v][1])**2)**0.5

            # Ensure shadow_score is between 0 and 1
            data['shadow_score'] = max(0, min(data.get('shadow_score', 0), 1))

            # Calculate weight (always non-negative)
            data['weight'] = (
                (1 - shade_preference) * data['distance'] +
                shade_preference * max(0, 1 - data['shadow_score'])
            )

        optimal_path = nx.shortest_path(graph, source=start_node, target=end_node, weight='weight')
        optimal_path_coords = [pos[node] for node in optimal_path]
        optimal_path_geom = LineString(optimal_path_coords)

        optimal_path_gdf = gpd.GeoDataFrame({'geometry': [optimal_path_geom], 'is_optimal': [True]})
        paths_copy = paths.copy()
        paths_copy = gpd.GeoDataFrame(pd.concat([paths_copy, optimal_path_gdf], ignore_index=True))
        path_success = True

    if path_success:
        with placeholder:
            st.info("Graphing your route...")

        colors = {
            "paths": "#4A90E2",         # Light Blue
            "optimal_path": "#FF5A5F",  # Coral Red
            "shadows": "#4A4A4A",       # Charcoal Gray
            "buildings": "#F5A623",     # Amber
            "trees": "#7ED321",     # Lime Green
        }


        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(40, 20))  # Larger figure for better readability

        # Plot paths first
        paths_copy.plot(
            ax=ax, 
            color=colors["paths"], 
            linewidth=1, 
            alpha=0.8, 
            label="Paths"
        )


        # Plot shadows
        shadows.plot(
            ax=ax, 
            color=colors["shadows"], 
            alpha=0.4, 
            label="Shadows"
        )

        # Plot buildings
        buildings.plot(
            ax=ax, 
            color=colors["buildings"], 
            alpha=0.5, 
            label="Buildings"
        )

        # Merge the two tree datasets
        all_trees = pd.concat([con_trees, dec_trees], ignore_index=True)  # Combine coniferous and deciduous trees
        all_trees.plot(
            ax=ax, 
            color=colors["trees"], 
            alpha=0.6, 
            label="Trees"
        )

        # Highlight the optimal path
        paths_copy[paths_copy['is_optimal'] == True].plot(
            ax=ax, 
            color=colors["optimal_path"], 
            linewidth=3, 
            alpha=0.9, 
            label="Optimal Path"
        )

        # Add title and labels
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("Latitude", fontsize=12)

        # Add a legend with a modern style
        plt.legend(title="Map Layers", loc="upper right", fontsize=10, title_fontsize=12, frameon=True, edgecolor="gray")

        # Remove axes for a cleaner look
        ax.axis('off')

        # Show the plot
        with placeholder_map:
            st.pyplot(fig)
        
        paths_with_optimal = paths 
        with placeholder:
            st.success("Ta-da!")
