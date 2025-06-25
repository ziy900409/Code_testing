import pptx
from pptx.util import Pt, Cm
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import datetime

import io
import cairosvg

folder_path = "/shox/S2"

# --- 0. 自動建立佔位符圖片 (已修正) ---
def create_placeholder_images():
    """自動生成所有需要的示意圖，方便直接執行看到結果。"""
    print("Creating placeholder images...")
    if not os.path.exists('placeholders'):
        os.makedirs('placeholders')

    placeholder_specs = {
        "logo.png": (2, 0.5),
        "hand_path.png": (2, 2),
        "speed_profile.png": (2, 2),
        "performance.png": (2, 2),
        "forearm_muscles.png": (2, 3),
        "muscle_charts_combined.png": (4, 3), # 修正：產生合併後的大圖
        "fatigue_index.png": (4, 2),
        "muscle_activation.png": (4, 2)
    }

    for name, size in placeholder_specs.items():
        path = os.path.join('placeholders', name)
        if not os.path.exists(path): # 僅在檔案不存在時建立
            fig, ax = plt.subplots(figsize=size)
            ax.text(0.5, 0.5, name.replace('.png', ''), ha='center', va='center', fontsize=10, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
    print("Placeholder images are ready in 'placeholders' folder.")

# --- 1. 參數設定 (Generated from your A4 image) ---

# -- Overall Layout --
SLIDE_WIDTH = Cm(21.0)
SLIDE_HEIGHT = Cm(29.7)
MARGIN_LEFT = Cm(0.6354)
CONTENT_WIDTH = Cm(SLIDE_WIDTH.cm - MARGIN_LEFT.cm * 2)
# 1 px ≈ 0.0353 cm
# -- Section Y-Coordinates (for easy vertical adjustment) --
HEADER_Y = Cm(0.6707)
PLAYER_BG_Y = Cm(2.8)
FLICK_SHOT_Y = Cm(7.0)
FOREARM_MUSCLES_Y = Cm(13.2)
FATIGUE_TEST_Y = Cm(22.2)
FOOTER_Y = Cm(28.5)

# -- Header --
HEADER_SPECS = {
    'logo': {
        'label': 'Company Logo',
        'path': 'placeholders/page2/logo.svg', # Change to your actual logo path
        'left': MARGIN_LEFT,
        'top': HEADER_Y,
        'width': Cm(4.5184),
        'height': Cm(0.5295)
    },
    'date': {
        'text': f'Measured on {datetime.date.today().strftime("%Y-%m-%d")}',
        'left': Cm(17.0911),
        'top': Cm(0.7766),
        # 'height': Cm(0.3177),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(8),
            'bold': False,        # From font-weight: 400 means not bold
            'italic': False,      # From font-style: normal
            'color_hex': '#757575' # From color
        },
        'alignment': PP_ALIGN.LEFT # And alignment
    },
    # 'line_top': Cm(5.295)
}

# -- Body: Player Background --
PLAYER_BG_HEIGHT = Cm(3.8)
PLAYER_SPECS = {
    'name': {
        'text': 'shox',
        'left': Cm(0.93545),
        'top': Cm(2.2945),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(20),
            'bold': True,
            'italic': False,      # From font-style: normal
            'color_hex': '#000000' # From color
        },
        'alignment': PP_ALIGN.LEFT # And alignment
        },
    'details': {
        'text': 'Male • From France',
        'left': Cm(0.93545),
        'top': Cm(3.4594),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(10),
            'bold': False,
            'italic': False,      # From font-style: normal
            'color_hex': '#000000' # From color
        },
        'alignment': PP_ALIGN.LEFT # And alignment
        },
    'mouse_pref_title': {
        'text': 'Mouse 2 Preferences',
        'left': Cm(15.16135),
        'top': Cm(2.3298),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(10),
            'bold': True,
            'italic': False,      # From font-style: normal
            'color_hex': '#000000' # From color
        },
        'alignment': PP_ALIGN.LEFT # And alignment
        },
    # 'mouse_pref_line_top': Cm(PLAYER_BG_Y.cm + 1.2),
    'Mouse_brand': {
        'text': 'Mouse brand\n\nMouse model',
        'left': Cm(15.16135),
        'top': Cm(3.177),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(8),
            'bold': False,
            'italic': False,      # From font-style: normal
            'color_hex': '#757575' # From color
        },
        'alignment': PP_ALIGN.LEFT # And alignment
    },
    'Mouse_info': {
        'text': 'BenQ ZOWIE\n\nS2 (M)',
        'left': Cm(19.07965),
        'top': Cm(3.177),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(8),
            'bold': False,
            'italic': False,      # From font-style: normal
            'color_hex': '#000000' # From color
        },
        'alignment': PP_ALIGN.RIGHT # And alignment
    },
    'vertical_separator': {
        'left': Cm(14.826),
        'top': Cm(2.2592),
        'height': Cm(2.1886),
        'style': {
            'width_pt': Pt(1),          # Translates from '1px'
            'color_hex': '#E5E5E5'      # Translates from your primary color
        }
    },
    'horizontal_separator': {
        'left': Cm(15.4261),
        'top': Cm(2.9652),
        'width': Cm(4.236),  # <-- 修改處：使用 'width' 來定義水平線的長度
        # 'height': Cm(PLAYER_BG_HEIGHT.cm - 0.6), # <-- 移除 'height'
        'style': {
            'width_pt': Pt(1),
            'color_hex': '#000000' # 1 CC0040 # 2 000000
            }
        },
    'background_box': {
        'shape_type': MSO_SHAPE.ROUNDED_RECTANGLE,
        'left': Cm(0.6354),
        'top': Cm(1.9415),
        'width': Cm(19.8033),
        'height': Cm(2.8593),
        'style': {
            'fill_color_hex': '#F2F2F2',
            'border_color_hex': '#E5E5E5',
            'border_width_pt': Pt(1),
            'corner_radius': 0.025
            }
        },
}

# -- Body: Flick Shot --
FLICK_SHOT_SPECS = {
    'horizontal_separator': {
        'left': Cm(0.6354),
        'top': Cm(5.295),
        'width': Cm(19.8033),  # <-- 修改處：使用 'width' 來定義水平線的長度
        # 'height': Cm(PLAYER_BG_HEIGHT.cm - 0.6), # <-- 移除 'height'
        'style': {
            'width_pt': Pt(1),
            'color_hex': '#CC0040'
            }
        },
    'flick_title': {
        'text': 'Flick Shot (Pre & Post)',
        'left': Cm(0.6354),
        'top': Cm(5.7892),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(12),
            'bold': True,
            'italic': False,      # From font-style: normal
            'color_hex': '#CC0040' # From color
        },
        'alignment': PP_ALIGN.LEFT # And alignment
        },
    # Row 1 of charts
    'hand_path_title': {
        'text': 'Hand Path',
        'left': Cm(0.6354),
        'top': Cm(6.9541),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(12),
            'bold': True,
            'italic': False,      # From font-style: normal
            'color_hex': '#000000' # From color
        },
        'alignment': PP_ALIGN.LEFT # And alignment
        },
    'hand_path_image': {
        'label': 'Hand Path Image',
        'path': "placeholders/page2/Hand Path W175_H137_0623.png",
        'left': Cm(0.6354),
        'top': Cm(7.7307),
        'width': Cm(6.0716),
        'height': Cm(4.8361)
        },
    
    'speed_profile_title': {
        'text': 'Speed Profile',
        'left': Cm(7.2365),
        'top': Cm(6.9894),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(12),
            'bold': True,
            'italic': False,      # From font-style: normal
            'color_hex': '#000000' # From color
        },
        'alignment': PP_ALIGN.LEFT # And alignment
        },
    'speed_profile_image': {
        'label': 'Speed Profile Image',
        'path': 'placeholders/page2' + folder_path + '/speed_profile (2).png',
        'left': Cm(7.2365),
        'top': Cm(7.8366),
        'width': Cm(6.3893),
        'height': Cm(4.5537)
        },
    
    'performance_title': {
        'text': 'Performance',
        'left': Cm(14.12),
        'top': Cm(6.9894),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(12),
            'bold': True,
            'italic': False,      # From font-style: normal
            'color_hex': '#000000' # From color
        },
        'alignment': PP_ALIGN.LEFT # And alignment
        },
    
    'performance_legend_image': {
        'label': 'Performance Image',
        'path': 'placeholders/page2/description (1).png',
        'left': Cm(18.3207),
        'top': Cm(7.2012),
        'width': Cm(2.2239),
        'height': Cm(0.3177)
        },
    
    'performance_image': {
        'label': 'Performance Image',
        'path': 'placeholders/page2' + folder_path + '/performance (2).png',
        'left': Cm(14.12),
        'top': Cm(7.8366),
        'width': Cm(6.2128),
        'height': Cm(4.5537)
        },

    # Row 2 of charts (Forearm Muscles)
    'forearm_muscles_title': {
        'text': 'Forearm Muscles',
        'left': Cm(3.4947),
        'top': Cm(13.1316),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(10),
            'bold': False,
            'italic': False,      # From font-style: normal
            'color_hex': '#000000' # From color
        },
        'alignment': PP_ALIGN.LEFT # And alignment
        },
    'type_title': {
        'text': 'Type',
        'left': Cm(1.2002),
        'top': Cm(13.1316),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(10),
            'bold': False,
            'italic': False,      # From font-style: normal
            'color_hex': '#757575' # From color
        },
        'alignment': PP_ALIGN.LEFT # And alignment
        },
    'section_header': {
        # The correct shape name from your library's list
        'shape_type': MSO_SHAPE.ROUND_2_SAME_RECTANGLE,
        'left': Cm(0.6354),
        'top': Cm(12.8845),
        'width': Cm(6.1775),
        'height': Cm(0.9178),
        'hide_bottom_border': True, # This flag is used by the add_custom_box, we can ignore it for add_styled_shape
        'style': {
            'fill_color_hex': '#FFFFFF',
            'border_color_hex': '#E5E5E5',
            'border_width_pt': Pt(1),
            'corner_radius': 0.2
            }
        },
    'section_block': {
        # The correct shape name from your library's list
        'shape_type': MSO_SHAPE.RECTANGLE,
        'left': Cm(0.6354),
        'top': Cm(13.8023),
        'width': Cm(6.1775),
        'height': Cm(8.0484),
        'hide_bottom_border': True, # This flag is used by the add_custom_box, we can ignore it for add_styled_shape
        'style': {
            'fill_color_hex': '#FFFFFF',
            'border_color_hex': '#E5E5E5',
            'border_width_pt': Pt(1),
            'corner_radius': 0
            }
        },
    'forearm_muscles_image': {
        'label': 'Forearm Muscles Image',
        'path': 'placeholders/page2/Forearm Muscles W175_H228_NoEdge.svg',
        'left': Cm(0.6354),
        'top': Cm(13.8023),
        'width': Cm(6.1775),
        'height': Cm(8.0484)
        },
    'muscle_charts_image': {
        'label': 'Muscle Charts (Combined)',
        'path': 'placeholders/page2' + folder_path + '/muscle_charts_combined (2).png',
        'left': Cm(7.2365),
        'top': Cm(12.7645),
        'width': Cm(13.2022),
        'height': Cm(9.18)
        }
    }
    

# -- Body: Fatigue Test --
FATIGUE_SPECS = {
    'horizontal_separator': {
        'left': Cm(0.6354),
        'top': Cm(22.3449),
        'width': Cm(19.8033),  # <-- 修改處：使用 'width' 來定義水平線的長度
        # 'height': Cm(PLAYER_BG_HEIGHT.cm - 0.6), # <-- 移除 'height'
        'style': {
            'width_pt': Pt(1),
            'color_hex': '#CC0040'
            }
        },
    'fatigue_title': {
        'text': 'Fatigue Test',
        'left': Cm(0.6354),
        'top': Cm(22.8391),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(12),
            'bold': True,
            'italic': False,      # From font-style: normal
            'color_hex': '#CC0040' # From color
        },
        'alignment': PP_ALIGN.LEFT # And alignment
        },
    'fatigue_legend_image': {
        'label': 'Performance Image',
        'path': 'placeholders/page2/description (1).png',
        'left': Cm(18.1795),
        'top': Cm(23.0156),
        'width': Cm(2.2239),
        'height': Cm(0.3177)
        },
    
    'fatigue_index_title': {
        'text': 'Fatigue Index',
        'left': Cm(4.0948),
        'top': Cm(23.31565),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(10),
            'bold': True,
            'italic': False,      # From font-style: normal
            'color_hex': '#000000' # From color
        },
        'alignment': PP_ALIGN.LEFT # And alignment
        },
    'fatigue_index_image': {
        'label': 'Fatigue Index Image',
        'path': 'placeholders/page2' + folder_path + '/fatigue_index (2).png',
        'left': Cm(0.6354),
        'top': Cm(24.00985),
        'width': Cm(9.54865),
        'height': Cm(3.883)
        },
    
    'muscle_activation_title': {
        'text': 'Muscle Activation Level',
        'left': Cm(13.43165),
        'top': Cm(23.31565),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(10),
            'bold': True,
            'italic': False,      # From font-style: normal
            'color_hex': '#000000' # From color
        },
        'alignment': PP_ALIGN.LEFT # And alignment
        },
    'muscle_activation_image': {
        'label': 'Muscle Activation Image',
        'path': 'placeholders/page2' + folder_path + '/muscle_activation (2).png',
        'left': Cm(10.78415),
        'top': Cm(24.00985),
        'width': Cm(9.54865),
        'height': Cm(3.883)
        },
    
    'bottom_text': {
        'text': '(x / x)',
        'left': MARGIN_LEFT,
        'top': Cm(FATIGUE_TEST_Y.cm + 5.8),
        'width': CONTENT_WIDTH,
        'height': Cm(0.7)
        }
}

# -- Footer --
FOOTER_SPECS = {
    'page_num': {
        'text': '3/3',
        'left': Cm(10.1664),
        'top': Cm(28.5224),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(8),
            'bold': False,
            'italic': False,      # From font-style: normal
            'color_hex': '#757575' # From color
        },
        'alignment': PP_ALIGN.LEFT # And alignment
        }
    }

def add_line(slide, spec):
    """
    Adds a line to the slide based on a detailed specification.
    Determines if the line is horizontal or vertical and applies styling.
    """
    style = spec.get('style', {})
    thickness = style.get('width_pt', Pt(1))

    # This logic determines if the line is horizontal or vertical
    # and sets its thickness correctly.
    if 'height' in spec and 'width' not in spec: # It's a vertical line
        spec['width'] = thickness
    elif 'width' in spec and 'height' not in spec: # It's a horizontal line
        spec['height'] = thickness
    
    geom = calculate_geometry(spec, SLIDE_WIDTH.cm, SLIDE_HEIGHT.cm)
    
    final_geom_for_shape = {
        'left': geom['left'],
        'top': geom['top'],
        'width': geom['width'],
        'height': geom['height']
    }
    
    # Add the line shape
    shape = slide.shapes.add_shape(MSO_SHAPE.LINE_INVERSE, **final_geom_for_shape)
    
    # --- THE FIX ---
    # Add this line to disable the default shadow effect
    shape.shadow.inherit = False
    # ---------------
    
    # Apply color and other line styles
    line = shape.line
    line.fill.solid()
    if 'color_hex' in style:
        line.fill.fore_color.rgb = RGBColor(*hex_to_rgb(style['color_hex']))
    else:
        line.fill.fore_color.rgb = RGBColor(0, 0, 0)
        
    return shape
    
def hex_to_rgb(hex_color):
    """Converts a hex color string (e.g., '#RRGGBB') to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def calculate_geometry(spec, slide_width_cm, slide_height_cm):
    """
    Calculates the final left, top, width, and height for an element.
    It can derive missing values, e.g., calculate 'left' if 'right' and 'width' are given.
    """
    final_spec = spec.copy()

    # --- Horizontal Calculation ---
    if 'width' not in final_spec:
        # If no width, calculate it from left and right
        if 'left' in final_spec and 'right' in final_spec:
            final_spec['width'] = Cm(slide_width_cm - final_spec['left'].cm - final_spec['right'].cm)
    elif 'left' not in final_spec:
        # If no left, calculate it from right and width
        if 'right' in final_spec and 'width' in final_spec:
            final_spec['left'] = Cm(slide_width_cm - final_spec['right'].cm - final_spec['width'].cm)

    # --- Vertical Calculation ---
    if 'height' not in final_spec:
        # If no height, calculate it from top and bottom
        if 'top' in final_spec and 'bottom' in final_spec:
            final_spec['height'] = Cm(slide_height_cm - final_spec['top'].cm - final_spec['bottom'].cm)
    elif 'top' not in final_spec:
        # If no top, calculate it from bottom and height
        if 'bottom' in final_spec and 'height' in final_spec:
            final_spec['top'] = Cm(slide_height_cm - final_spec['bottom'].cm - final_spec['height'].cm)
            
    return final_spec
    
def add_formatted_text(slide, spec):
    """
    A powerful function to add a textbox based on a detailed specification.
    Supports auto-sizing and rich font formatting from hex or RGB.
    """
    # Use placeholder dimensions if auto-sizing, otherwise use specified dimensions
    if spec.get('auto_size', False):
        width = height = Cm(1) 
    else:
        final_geom = calculate_geometry(spec, SLIDE_WIDTH.cm, SLIDE_HEIGHT.cm)
        width = final_geom.get('width', Cm(5))
        height = final_geom.get('height', Cm(1))

    tb = slide.shapes.add_textbox(spec['left'], spec['top'], width, height)
    p = tb.text_frame.paragraphs[0]
    p.text = spec.get('text', '') # Use .get for safety

    # Apply font settings from the spec
    if 'font' in spec and isinstance(spec['font'], dict):
        font_spec = spec['font']
        p.font.name = font_spec.get('name', None)
        p.font.size = font_spec.get('size', Pt(12))
        p.font.bold = font_spec.get('bold', False)
        p.font.italic = font_spec.get('italic', False)
        
        # ================== NEW LOGIC FOR COLOR ==================
        # Prioritize hex color, fall back to RGB
        if 'color_hex' in font_spec:
            rgb_tuple = hex_to_rgb(font_spec['color_hex'])
            p.font.color.rgb = RGBColor(*rgb_tuple)
        elif 'color_rgb' in font_spec:
            p.font.color.rgb = RGBColor(*font_spec['color_rgb'])
        # =======================================================
            
    # Apply alignment
    p.alignment = spec.get('alignment', PP_ALIGN.LEFT)
    
    # Apply auto-sizing at the end
    if spec.get('auto_size', False):
        tb.text_frame.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT
        
    return tb

def add_styled_shape(slide, spec):
    """
    Adds a shape with advanced styling (fill, border, corner radius)
    based on a detailed specification.
    """
    geom = calculate_geometry(spec, SLIDE_WIDTH.cm, SLIDE_HEIGHT.cm)
    style = spec.get('style', {})
    shape_type = spec.get('shape_type', MSO_SHAPE.RECTANGLE)

    # Create the shape
    shape = slide.shapes.add_shape(shape_type, geom['left'], geom['top'], geom['width'], geom['height'])

    # --- THE FIX ---
    # Add this line to disable the default shadow effect
    shape.shadow.inherit = False
    # ---------------

    # Apply fill styling
    if 'fill_color_hex' in style:
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(*hex_to_rgb(style['fill_color_hex']))
    else:
        shape.fill.background()

    # Apply border (line) styling
    if 'border_color_hex' in style or 'border_width_pt' in style:
        line = shape.line
        line.fill.solid()
        if 'border_color_hex' in style:
            line.fill.fore_color.rgb = RGBColor(*hex_to_rgb(style['border_color_hex']))
        line.width = style.get('border_width_pt', Pt(1))

    # Apply corner radius if the shape is a rounded rectangle
    if shape_type == MSO_SHAPE.ROUNDED_RECTANGLE and 'corner_radius' in style:
        shape.adjustments[0] = style['corner_radius']
        
    return shape

def add_smart_picture(slide, spec):
    """
    Adds a picture to the slide, automatically handling SVG or PNG/JPG.
    If 'height' is provided in the spec, it sets a fixed size.
    Otherwise, it scales the image based on 'width' while maintaining aspect ratio.
    """
    image_path = spec['path']
    print(f"Adding image: {os.path.basename(image_path)}")

    image_source = None
    
    # Check if the file path ends with .svg (case-insensitive)
    if image_path.lower().endswith('.svg'):
        # It's an SVG: convert to PNG in memory
        png_output = io.BytesIO()
        cairosvg.svg2png(url=image_path, write_to=png_output, output_width=2048)
        png_output.seek(0)
        image_source = png_output
    else:
        # It's a PNG, JPG, etc.: use the file path directly
        image_source = image_path

    # --- NEW LOGIC TO HANDLE FIXED HEIGHT ---
    # Check if a specific height is provided in the spec
    if 'height' in spec:
        # If yes, use both width and height (this may distort the image)
        print(f"  ... constraining to fixed size: {spec['width'].cm:.2f}cm x {spec['height'].cm:.2f}cm")
        slide.shapes.add_picture(
            image_source, 
            spec['left'], 
            spec['top'],
            width=spec['width'], 
            height=spec['height']
        )
    else:
        # If no height is given, only use width and maintain aspect ratio
        print(f"  ... scaling to width: {spec['width'].cm:.2f}cm (auto height)")
        slide.shapes.add_picture(
            image_source, 
            spec['left'], 
            spec['top'], 
            width=spec['width']
        )
        
def add_custom_box(slide, spec):
    """
    Adds a complex styled box, such as one with a 3-sided border and rounded corners.
    """
    # Calculate the final geometry for the shape
    geom = calculate_geometry(spec, SLIDE_WIDTH.cm, SLIDE_HEIGHT.cm)
    style = spec.get('style', {})
    shape_type = spec.get('shape_type', MSO_SHAPE.RECTANGLE)

    # --- Step A: Create the main shape ---
    main_shape = slide.shapes.add_shape(
        shape_type, 
        geom['left'], geom['top'], geom['width'], geom['height']
    )
    main_shape.shadow.inherit = False

    # Apply fill and border from the spec
    fill_color_hex = style.get('fill_color_hex', '#FFFFFF') # Default to white
    border_color_hex = style.get('border_color_hex', '#000000') # Default to black
    border_width_pt = style.get('border_width_pt', Pt(1))

    main_shape.fill.solid()
    main_shape.fill.fore_color.rgb = RGBColor(*hex_to_rgb(fill_color_hex))
    
    main_shape.line.fill.solid()
    main_shape.line.fill.fore_color.rgb = RGBColor(*hex_to_rgb(border_color_hex))
    main_shape.line.width = border_width_pt

    # Apply corner radius if specified
    if shape_type == MSO_SHAPE.ROUND_TOP_CORNERS_RECTANGLE and 'corner_radius' in style:
        main_shape.adjustments[0] = style.get('corner_radius', 0.1)

    # --- Step B: The trick to hide the bottom border ---
    if spec.get('hide_bottom_border', False):
        cover_left = geom['left']
        cover_top = geom['top'] + geom['height'] - border_width_pt
        cover_width = geom['width']
        cover_height = border_width_pt

        cover_shape = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, cover_left, cover_top, cover_width, cover_height
        )
        
        # Style the cover-up shape to be invisible
        cover_shape.fill.solid()
        cover_shape.fill.fore_color.rgb = RGBColor(*hex_to_rgb(fill_color_hex))
        cover_shape.line.fill.background()

    return main_shape

# --- 3. PPT 生成主函式 (已全面重構和清理) ---
def create_report(output_filename="science_report.pptx"):
    """根據以上所有設定，生成最終的 PowerPoint 報告。"""
    prs = pptx.Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # --- 1. Header ---
    print("Adding Header...")
    add_smart_picture(slide, HEADER_SPECS['logo'])
    add_formatted_text(slide, HEADER_SPECS['date'])
    # add_line(slide, {'left': MARGIN_LEFT, 'top': HEADER_SPECS['line_top'], 'width': CONTENT_WIDTH, 'style': {'width_pt': Pt(1), 'color_hex': '#E5E5E5'}})

    # --- 2. Player Background and Info ---
    print("Adding Player Info...")
    # Use the new helper function to add the styled background box
    add_styled_shape(slide, PLAYER_SPECS['background_box'])

    # Add all player text info
    for key in ['name', 'details', 'mouse_pref_title', 'Mouse_info', 'Mouse_brand']:
        add_formatted_text(slide, PLAYER_SPECS[key])
        
    # Add the vertical and horizontal separator lines
    for key in ['vertical_separator', 'horizontal_separator']:
        if key in PLAYER_SPECS: # Check if the line spec exists before adding
            add_line(slide, PLAYER_SPECS[key])

    # Add and format the mouse preferences table
    # if 'mouse_pref_table' in PLAYER_SPECS:
    #     table_spec = PLAYER_SPECS['mouse_pref_table']
    #     table_geom = calculate_geometry(table_spec, SLIDE_WIDTH.cm, SLIDE_HEIGHT.cm)
    #     shape = slide.shapes.add_table(len(table_spec['data']), len(table_spec['data'][0]), table_geom['left'], table_geom['top'], table_geom['width'], Cm(1.5))
    #     table = shape.table
    #     for i, width in enumerate(table_spec['col_widths']):
    #         table.columns[i].width = width
    #     for r, row_data in enumerate(table_spec['data']):
    #         for c, cell_text in enumerate(row_data):
    #             cell = table.cell(r, c)
    #             cell.fill.background()
    #             p = cell.text_frame.paragraphs[0]
    #             p.text = cell_text
    #             p.font.size = Pt(11)
    #             cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    #             if c == 0:
    #                 p.font.color.rgb = RGBColor(*hex_to_rgb("#757575"))
    #                 p.alignment = PP_ALIGN.LEFT
    #             else:
    #                 p.font.color.rgb = RGBColor(0,0,0)
    #                 p.alignment = PP_ALIGN.RIGHT
    
    # --- 3. Main Content Sections ---
    
    for key in ['section_block', 'section_header']:    
        add_styled_shape(slide, FLICK_SHOT_SPECS[key])
    all_sections = {'Flick Shot': FLICK_SHOT_SPECS, 'Fatigue Test': FATIGUE_SPECS}
    for section_name, section_specs in all_sections.items():
        print(f"Adding {section_name} Section...")
        print(section_name, section_specs)
        # for spec in section_specs.values():
        #     print(spec)
        #     if isinstance(spec, dict): # Process only dictionary specs
        #         if 'text' in spec:
        #             add_formatted_text(slide, spec)
        #         elif 'path' in spec:
        #             add_smart_picture(slide, spec)
        for key in section_specs.keys():
            print(key)
            if '_title' in key:
                add_formatted_text(slide, section_specs[key])
                print(key)
            elif '_image' in key:
                
                add_smart_picture(slide, section_specs[key])
            elif '_separator' in key:
                add_line(slide, section_specs[key])
    # Use the new function to create the styled header box
    # Use the correct function that reads the 'shape_type' from your config
    

    # --- 4. Footer ---
    print("Adding Footer...")
    add_formatted_text(slide, FOOTER_SPECS['page_num'])

    # --- 5. Save Presentation ---
    prs.save(output_filename)
    print(f"Report successfully saved as '{output_filename}'")

# --- Main Execution ---
if __name__ == '__main__':
    create_placeholder_images()
    # preview_layout()
    create_report()