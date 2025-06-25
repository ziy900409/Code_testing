# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 22:28:15 2025

@author: Hsin.YH.Yang
"""

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
        'type': 'figure',
        'label': 'Company Logo',
        'path': 'placeholders/summary_page/logo.svg', # Change to your actual logo path
        'left': MARGIN_LEFT,
        'top': HEADER_Y,
        'width': Cm(4.5184),
        'height': Cm(0.5295)
    },
    'date': {
        'type': 'text',
        'text': f'Measured on {datetime.date.today().strftime("%Y-%m-%d")}',
        'left': Cm(16.96165),
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
    'background_box': { # 外框底
        'type': 'block',
        'shape_type': MSO_SHAPE.ROUNDED_RECTANGLE,
        'left': Cm(0.6354),
        'top': Cm(1.9062),
        'width': Cm(19.7327),
        'height': Cm(5.0126),
        'style': {
            'fill_color_hex': '#F2F2F2',
            'border_color_hex': '#E5E5E5',
            'border_width_pt': Pt(1),
            'corner_radius': 0.025
            }
        },
    'name': { # 選手名字
        'type': 'text',
        'text': 'michu',
        'left': Cm(0.97075),
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
    'details': { # 性別、隊伍
        'type': 'text',
        'text': 'Male • From Poland',
        'left': Cm(0.97075),
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
    'vertical_separator': { #分割名字與 Game perferences
        'type': 'line',
        'left': Cm(8.4367),
        'top': Cm(2.2592),
        'height': Cm(1.765),
        'style': {
            'width_pt': Pt(1),          # Translates from '1px'
            'color_hex': '#E5E5E5'      # Translates from your primary color
        }
    },
    'horizontal_separator': {
        'type': 'line',
        'left': Cm(1.2002),
        'top': Cm(4.3066),
        'width': Cm(18.6031),  # <-- 修改處：使用 'width' 來定義水平線的長度
        # 'height': Cm(PLAYER_BG_HEIGHT.cm - 0.6), # <-- 移除 'height'
        'style': {
            'width_pt': Pt(1),
            'color_hex': '#E5E5E5'
            }
        },
    'game_perference_title': {
        'type': 'text',
        'text': 'Game Perferences',
        'left': Cm(8.6838),
        'top': Cm(2.2592),
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
    'game_brand': {
        'type': 'text',
        'text': 'Most plated game:\n\nIn-game sensitivity:\nOther games you play:',
        'left': Cm(12.09025),
        'top': Cm(2.3651),
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
    'game_info': {
        'type': 'text',
        'text': 'CS2\n\n1\nNAN',
        'left': Cm(19.07965),
        'top': Cm(2.3651),
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
    'mouse_1_pref_title': {
        'type': 'text',
        'text': 'Mouse 1 Preferences',
        'left': Cm(0.93545),
        'top': Cm(4.6243),
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
    'horizontal_1_separator': {
        'type': 'line',
        'left': Cm(1.2002),
        'top': Cm(5.2597),
        'width': Cm(9.01915),  # <-- 修改處：使用 'width' 來定義水平線的長度
        # 'height': Cm(PLAYER_BG_HEIGHT.cm - 0.6), # <-- 移除 'height'
        'style': {
            'width_pt': Pt(1),
            'color_hex': '#CC0040'
            }
        },
    # 'mouse_pref_line_top': Cm(PLAYER_BG_Y.cm + 1.2),
    'Mouse_1_brand': {
        'type': 'text',
        'text': 'Mouse brand\n\nMouse model',
        'left': Cm(0.93545),
        'top': Cm(5.4715),
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
    'Mouse_1_info': {
        'type': 'text',
        'text': 'BenQ ZOWIE\n\nMousa A (M)',
        'left': Cm(9.4604),
        'top': Cm(5.4715),
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
    # mouse 2
    'mouse_2_pref_title': {
        'type': 'text',
        'text': 'Mouse 2 Preferences',
        'left': Cm(10.493278),
        'top': Cm(4.6243),
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
    'horizontal_2_separator': {
        'type': 'line',
        'left': Cm(10.758028),
        'top': Cm(5.2597),
        'width': Cm(9.01915),  # <-- 修改處：使用 'width' 來定義水平線的長度
        # 'height': Cm(PLAYER_BG_HEIGHT.cm - 0.6), # <-- 移除 'height'
        'style': {
            'width_pt': Pt(1),
            'color_hex': '#212121'
            }
        },
    # 'mouse_pref_line_top': Cm(PLAYER_BG_Y.cm + 1.2),
    'Mouse_2_brand': {
        'type': 'text',
        'text': 'Mouse brand\n\nMouse model',
        'left': Cm(10.493278),
        'top': Cm(5.4715),
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
    'Mouse_2_info': {
        'type': 'text',
        'text': 'BenQ ZOWIE\n\nEC2 (M)',
        'left': Cm(19.04435),
        'top': Cm(5.4715),
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
    
}

# -- Body: SUMMARY --
FLICK_SHOT_SPECS = {
    'horizontal_separator': {
        'type': 'line',
        'left': Cm(0.6354),
        'top': Cm(7.413),
        'width': Cm(19.7327),  # <-- 修改處：使用 'width' 來定義水平線的長度
        # 'height': Cm(PLAYER_BG_HEIGHT.cm - 0.6), # <-- 移除 'height'
        'style': {
            'width_pt': Pt(1),
            'color_hex': '#CC0040'
            }
        },
    # Col 1 of charts
    'summary_title': {
        'type': 'text',
        'text': 'Summary',
        'left': Cm(0.6354),
        'top': Cm(7.9072),
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
    
    'radar_text_1': {
        'type': 'text',
        'text': 'Accuracy',
        'left': Cm(5.1185),
        'top': Cm(9.394036),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(10),
            'bold': False,
            'italic': False,      # From font-style: normal
            'color_hex': '#000000' # From color
            },
        'alignment': PP_ALIGN.CENTER # And alignment
        },
    'radar_text_2': {
        'type': 'text',
        'text': 'Speed',
        'left': Cm(0.6354),
        'top': Cm(12.350764),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(10),
            'bold': False,
            'italic': False,      # From font-style: normal
            'color_hex': '#000000' # From color
            },
        'alignment': PP_ALIGN.CENTER # And alignment
        },
    'radar_text_3': {
        'type': 'text',
        'text': 'Fatigue',
        'left': Cm(2.374278),
        'top': Cm(16.665836),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(10),
            'bold': False,
            'italic': False,      # From font-style: normal
            'color_hex': '#000000' # From color
            },
        'alignment': PP_ALIGN.CENTER # And alignment
        },
    'radar_text_4': {
        'type': 'text',
        'text': 'Micro adjustment',
        'left': Cm(7.739878),
        'top': Cm(16.665836),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(10),
            'bold': False,
            'italic': False,      # From font-style: normal
            'color_hex': '#000000' # From color
            },
        'alignment': PP_ALIGN.CENTER # And alignment
        },
    'radar_text_5': {
        'type': 'text',
        'text': 'Summary',
        'left': Cm(9.337203),
        'top': Cm(12.350764),
        'auto_size': True,  # <-- Tell the script to auto-fit the text
        'font': {
            'name': 'Roboto', # You can even specify the font name
            'size': Pt(10),
            'bold': False,
            'italic': False,      # From font-style: normal
            'color_hex': '#000000' # From color
            },
        'alignment': PP_ALIGN.CENTER # And alignment
        },
    'radar_image': {
        'type': 'figure',
        'label': 'Radar Image',
        'path': "placeholders/summary_page/radar_fig.png",
        'left': Cm(1.6354),
        'top': Cm(9.6956),
        'width': Cm(8),
        'height': Cm(8)
        },
    # Col 2 of charts
    'background_box': { # 外框底
        'type': 'block',
        'shape_type': MSO_SHAPE.ROUNDED_RECTANGLE,
        'left': Cm(11.3666),
        'top': Cm(8.8956),
        'width': Cm(9.0015),
        'height': Cm(8.6838),
        'style': {
            'fill_color_hex': '#F2F2F2',
            'border_color_hex': '#E5E5E5',
            'border_width_pt': Pt(1),
            'corner_radius': 0.025
            }
        },
    'summary_legend': {
        'type': 'figure',
        'label': 'Hand Path Image',
        'path': "placeholders/summary_page/summary_legend.svg",
        'left': Cm(17.7206),
        'top': Cm(9.2839),
        'width': Cm(2.2592),
        'height': Cm(0.3177)
        },
    'chart_1_title': {
        'type': 'text',
        'text': 'TTK',
        'left': Cm(11.59605),
        'top': Cm(9.3192),
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
    'chart_1_unit': {
        'type': 'text',
        'text': '(ms)',
        'left': Cm(12.54915),
        'top': Cm(9.40745),
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
    'chart_1_image': {
        'type': 'figure',
        'label': 'Speed Profile Image',
        'path': 'placeholders/summary_page/TTK.png',
        'left': Cm(11.8608),
        'top': Cm(10),
        'width': Cm(8.0484),
        'height': Cm(1.9768)
        },
    'chart_2_title': {
        'type': 'text',
        'text': 'Shot Counts',
        'left': Cm(11.59605),
        'top': Cm(12.0726),
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
    'chart_2_unit': {
        'type': 'text',
        'text': '(quantity)',
        'left': Cm(14.06705),
        'top': Cm(12.16085),
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
    'chart_2_image': {
        'type': 'figure',
        'label': 'Speed Profile Image',
        'path': 'placeholders/summary_page/ShotCounts.png',
        'left': Cm(11.8608),
        'top': Cm(12.7374),
        'width': Cm(8.0484),
        'height': Cm(1.9768)
        },
    'chart_3_title': {
        'type': 'text',
        'text': 'Accuracy',
        'left': Cm(11.59605),
        'top': Cm(14.826),
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
    'chart_3_unit': {
        'type': 'text',
        'text': '(%)',
        'left': Cm(13.50225),
        'top': Cm(14.91425),
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
    'chart_3_image': {
        'type': 'figure',
        'label': 'Speed Profile Image',
        'path': 'placeholders/summary_page/Accuracy.png',
        'left': Cm(11.8608),
        'top': Cm(15.5108),
        'width': Cm(8.0484),
        'height': Cm(1.9768)
        },
    
    }
    

# -- Body: Table --
FATIGUE_SPECS = {
    'horizontal_separator': {
        'type': 'line',
        'left': Cm(0.6354),
        'top': Cm(18.0736),
        'width': Cm(19.7327),  # <-- 修改處：使用 'width' 來定義水平線的長度
        # 'height': Cm(PLAYER_BG_HEIGHT.cm - 0.6), # <-- 移除 'height'
        'style': {
            'width_pt': Pt(1),
            'color_hex': '#CC0040'
            }
        },
    'table_title': {
        'type': 'text',
        'text': 'Table',
        'left': Cm(0.6354),
        'top': Cm(18.5678),
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
    
    # 'fatigue_index_title': {
    #     'type': 'text',
    #     'text': 'Fatigue Index',
    #     'left': Cm(4.1301),
    #     'top': Cm(23.31565),
    #     'auto_size': True,  # <-- Tell the script to auto-fit the text
    #     'font': {
    #         'name': 'Roboto', # You can even specify the font name
    #         'size': Pt(10),
    #         'bold': True,
    #         'italic': False,      # From font-style: normal
    #         'color_hex': '#000000' # From color
    #     },
    #     'alignment': PP_ALIGN.LEFT # And alignment
    #     },
    # 'fatigue_index_image': {
    #     'type': 'figure',
    #     'label': 'Fatigue Index Image',
    #     'path': 'placeholders/fatigue_index (2).png',
    #     'left': Cm(0.6707),
    #     'top': Cm(24.00985),
    #     'width': Cm(9.54865),
    #     'height': Cm(3.883)
    #     },
    
    # 'muscle_activation_title': {
    #     'type': 'text',
    #     'text': 'Muscle Activation Level',
    #     'left': Cm(13.46695),
    #     'top': Cm(23.31565),
    #     'auto_size': True,  # <-- Tell the script to auto-fit the text
    #     'font': {
    #         'name': 'Roboto', # You can even specify the font name
    #         'size': Pt(10),
    #         'bold': True,
    #         'italic': False,      # From font-style: normal
    #         'color_hex': '#000000' # From color
    #     },
    #     'alignment': PP_ALIGN.LEFT # And alignment
    #     },
    # 'muscle_activation_image': {
    #     'type': 'figure',
    #     'label': 'Muscle Activation Image',
    #     'path': 'placeholders/muscle_activation (2).png',
    #     'left': Cm(10.81945),
    #     'top': Cm(24.00985),
    #     'width': Cm(9.54865),
    #     'height': Cm(3.883)
    #     },
    
    # 'bottom_text': {
    #     'type': 'text',
    #     'text': '(x / x)',
    #     'left': MARGIN_LEFT,
    #     'top': Cm(FATIGUE_TEST_Y.cm + 5.8),
    #     'width': CONTENT_WIDTH,
    #     'height': Cm(0.7)
    #     }
}

# -- Footer --
FOOTER_SPECS = {
    'page_num': {
        'text': '1/3',
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
def create_report(output_filename="summary_page.pptx"):
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
    
    for key in PLAYER_SPECS:
        print(PLAYER_SPECS[key]['type'])
        if 'block' in PLAYER_SPECS[key]['type']:
            print(PLAYER_SPECS[key])
            add_styled_shape(slide, PLAYER_SPECS['background_box'])
    for key in PLAYER_SPECS:
        if 'text' in PLAYER_SPECS[key]['type']:
            print(PLAYER_SPECS[key])
            add_formatted_text(slide, PLAYER_SPECS[key])
    for key in PLAYER_SPECS:
        if 'line' in PLAYER_SPECS[key]['type']:
            add_line(slide, PLAYER_SPECS[key])
            
    
    # --- 3. Main Content Sections ---
    # for key in ['section_block', 'section_header']:    
    #     add_styled_shape(slide, FLICK_SHOT_SPECS[key])
    
    # for key in FLICK_SHOT_SPECS:
    #     print(key)
        
    
    all_sections = {'Flick Shot': FLICK_SHOT_SPECS, 'Fatigue Test': FATIGUE_SPECS}
    for section_name, section_specs in all_sections.items():
        print(f"Adding {section_name} Section...")
        print(section_name, section_specs)
        for key in section_specs.keys():
            if 'block' in section_specs[key]['type']:
                # print(FLICK_SHOT_SPECS[key])
                add_styled_shape(slide, section_specs[key])
        for key in section_specs.keys():
            if 'figure' in section_specs[key]['type']:
                add_smart_picture(slide, section_specs[key])
        for key in section_specs.keys():
            if 'line' in section_specs[key]['type']:
                add_line(slide, section_specs[key])
        for key in section_specs.keys():
            print(key)
            if 'text' in section_specs[key]['type']:
                add_formatted_text(slide, section_specs[key])
                print(key)
        
        
    
    
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