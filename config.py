import json
import numpy as np
import os
import sys

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

def load_and_calculate_config(config_filepath="config.json"):
    
    try:
        abs_path = get_resource_path(config_filepath)
        with open(abs_path, 'r') as f:
            data = json.load(f)
            
    except FileNotFoundError:
        # Log error and return None if the essential config file is missing.
        print(f"Error: Configuration file not found at {abs_path}")
        return None 
    
    screenDimensions = data['screenDimensions']
    detail = data['detail']
    data['camRes'] = [
        int(screenDimensions[0] * detail), 
        int(screenDimensions[1] * detail)
    ]

    G = data['G']
    MASS = data['MASS']
    c = data['c']
    scale = data['scale']
    
    G_norm = G / G
    MASS_norm = MASS / (MASS * scale)
    c_norm = c / c

    # Overwrite the raw values with normalized constants.
    data['G'] = G_norm
    data['MASS'] = MASS_norm
    data['c'] = c_norm
    
    # Calculate the Schwarzschild radius (r_s = 2GM / c^2) using normalized constants.
    data['r_s'] = (2 * G_norm * MASS_norm) / (c_norm ** 2)
    
    # Define SZR (Schwarzschild/2), a value frequently used in the metric equations.
    data['SZR'] = data['r_s'] / 2
    
    return data

# Dictionary holding all final, calculated configuration values used globally.
CONFIG = load_and_calculate_config()

# Expose calculated variables as global module constants for easy import (e.g., from config import c).
if CONFIG:
    screenDimensions = CONFIG['screenDimensions']
    camRes = CONFIG['camRes']
    screen_size = CONFIG['screen_size']
    G = CONFIG['G']
    MASS = CONFIG['MASS']
    c = CONFIG['c']
    r_s = CONFIG['r_s']
    SZR = CONFIG['SZR']