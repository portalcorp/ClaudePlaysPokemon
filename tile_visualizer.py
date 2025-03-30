from PIL import Image, ImageDraw
import numpy as np

def create_tile_overlay(collision_map_str, alpha=128):
    """
    Create a transparent overlay showing walkable/unwalkable tiles from the collision map string.
    
    Args:
        collision_map_str (str): ASCII collision map from emulator
        alpha (int): Transparency value (0-255)
        
    Returns:
        PIL.Image: RGBA image overlay
    """
    if not collision_map_str:
        return None
        
    # Parse the collision map string
    lines = collision_map_str.split('\n')
    # Remove the border lines and legend
    map_lines = [line[1:-1] for line in lines[1:-1] if line.startswith('|')]
    
    # Create a transparent image (160x144 is the Game Boy resolution)
    overlay = Image.new('RGBA', (160, 144), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Calculate tile size
    tile_width = 160 // 10  # 10 columns
    tile_height = 144 // 9  # 9 rows
    
    # Draw tiles
    for row, line in enumerate(map_lines):
        for col, char in enumerate(line):
            x1 = col * tile_width
            y1 = row * tile_height
            x2 = x1 + tile_width
            y2 = y1 + tile_height
            
            if char == '█':  # Wall/obstacle
                draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, alpha))  # Red for walls
            elif char == '·':  # Walkable path
                draw.rectangle([x1, y1, x2, y2], fill=(0, 255, 0, alpha))  # Green for paths
            elif char == 'S':  # Sprite/NPC
                draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 255, alpha))  # Blue for sprites
            elif char in '↑↓←→':  # Player
                draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 0, alpha))  # Yellow for player
                
    return overlay

def overlay_on_screenshot(screenshot, collision_map_str, alpha=128):
    """
    Create a new image with the tile overlay blended onto the screenshot.
    
    Args:
        screenshot (PIL.Image): Original screenshot
        collision_map_str (str): ASCII collision map from emulator
        alpha (int): Transparency value (0-255)
        
    Returns:
        PIL.Image: Screenshot with overlay
    """
    overlay = create_tile_overlay(collision_map_str, alpha)
    if overlay is None:
        return screenshot
        
    # Ensure screenshot is in RGBA mode for alpha blending
    if screenshot.mode != 'RGBA':
        screenshot = screenshot.convert('RGBA')
    
    # Composite the images
    return Image.alpha_composite(screenshot, overlay) 