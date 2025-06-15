import pptx
from pptx.util import Pt, Cm
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import datetime

import io
import cairosvg

def add_smart_picture(slide, spec):
    """
    Adds a picture to the slide, automatically handling SVG or PNG/JPG.
    If the path is SVG, it converts it to a high-res PNG in memory.
    Otherwise, it adds the image directly.
    """
    image_path = spec['path']
    print(f"Adding image: {os.path.basename(image_path)}")

    # Check if the file path ends with .svg (case-insensitive)
    if image_path.lower().endswith('.svg'):
        # It's an SVG: convert to PNG in memory
        png_output = io.BytesIO()
        
        # CORRECTED: Call the function from the cairosvg library
        cairosvg.svg2png(url=image_path, write_to=png_output, output_width=2048)
        
        png_output.seek(0) # Rewind the stream to the beginning
        
        # Add the picture from the in-memory PNG stream
        slide.shapes.add_picture(
            png_output,
            spec['left'],
            spec['top'],
            width=spec['width']
        )
    else:
        # It's a PNG, JPG, etc.: add it directly from the file path
        slide.shapes.add_picture(
            image_path,
            spec['left'],
            spec['top'],
            width=spec['width']
        )

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

# --- 1. 參數設定 (所有元件的尺寸、位置和內容) ---

# -- 整體版面 --
SLIDE_WIDTH = Cm(21.0)
SLIDE_HEIGHT = Cm(29.7)
MARGIN_LEFT = Cm(1.5)
# Corrected calculation pattern applied here
CONTENT_WIDTH = Cm(SLIDE_WIDTH.cm - MARGIN_LEFT.cm * 2)

# -- Header --
HEADER_Y = Cm(1.0)
HEADER_SPECS = {
    # MODIFIED: 'logo' is now an image with a 'path' instead of 'text'
    'logo': {
        'label': 'Company Logo',
        'path': 'placeholders/logo.svg',
        # 'path': r"D:\BenQ_Project\gitgit\Code_testing\LabProject\PerformanceAnalysis\placeholders\logo.svg",
        'left': MARGIN_LEFT,
        'top': HEADER_Y,
        'width': Cm(5),
        'height': Cm(1.5)
    },
    'date': {
        'text': f'Measured on {datetime.date.today().strftime("%Y-%m-%d")}', 
        'left': Cm(15), 'top': HEADER_Y, 'width': Cm(4.5), 'height': Cm(1)
    },
    'line_top': Cm(2.2)
}
# -- Body: Player Background --
PLAYER_BG_Y = Cm(2.8)
PLAYER_BG_HEIGHT = Cm(3.2)
PLAYER_SPECS = {
    'name': {'text': 'Robert Fox', 'left': Cm(MARGIN_LEFT.cm + 0.5), 'top': Cm(PLAYER_BG_Y.cm + 0.3), 'width': Cm(8), 'height': Cm(1)},
    'details': {'text': 'Male • From Fnatic', 'left': Cm(MARGIN_LEFT.cm + 0.5), 'top': Cm(PLAYER_BG_Y.cm + 1.3), 'width': Cm(8), 'height': Cm(1)},
    'mouse_title': {'text': 'Mouse Preferences', 'left': Cm(11), 'top': Cm(PLAYER_BG_Y.cm + 0.3), 'width': Cm(4), 'height': Cm(0.7)},
    'mouse_details': {'text': 'Mouse brand:\tBenQ ZOWIE\nMouse model:\tWWWW\n\t\tEC (S)', 'left': Cm(11), 'top': Cm(PLAYER_BG_Y.cm + 1.0), 'width': Cm(8), 'height': Cm(2)},
    'line_left': Cm(10.5), 
    'line_top': Cm(PLAYER_BG_Y.cm + 0.3), 
    'line_height': Cm(PLAYER_BG_HEIGHT.cm - 0.6)
}

# -- Body: Flick Shot --
FLICK_SHOT_Y = Cm(6.8)
FLICK_SHOT_SPECS = {
    'title': {'text': 'Flick Shot (Pre & Post)', 'left': MARGIN_LEFT, 'top': FLICK_SHOT_Y, 'width': Cm(6), 'height': Cm(1)},
    'hand_path': {'label': 'Hand Path', 'path': 'placeholders/hand_path.png', 'left': MARGIN_LEFT, 'top': Cm(FLICK_SHOT_Y.cm + 1.2), 'width': Cm(5.5)},
    'speed_profile': {'label': 'Speed Profile', 'path': 'placeholders/speed_profile.png', 'left': Cm(MARGIN_LEFT.cm + 6.0), 'top': Cm(FLICK_SHOT_Y.cm + 1.2), 'width': Cm(5.5)},
    'performance': {'label': 'Performance', 'path': 'placeholders/performance.png', 'left': Cm(MARGIN_LEFT.cm + 12.0), 'top': Cm(FLICK_SHOT_Y.cm + 1.2), 'width': Cm(6)},
    'forearm_muscles': {'label': 'Forearm Muscles', 'path': 'placeholders/forearm_muscles.png', 'left': Cm(MARGIN_LEFT.cm + 0.5), 'top': Cm(FLICK_SHOT_Y.cm + 6.5), 'width': Cm(4.5)},
    'muscle_charts_combined': {'label': 'Muscle Charts (Combined)', 'path': 'placeholders/muscle_charts_combined.png', 'left': Cm(MARGIN_LEFT.cm + 6.0), 'top': Cm(FLICK_SHOT_Y.cm + 6.0), 'width': Cm(11.5), 'height': Cm(6.0)}
}

# -- Body: Fatigue Test --
FATIGUE_Y = Cm(20.0)
FATIGUE_SPECS = {
    'title': {'text': 'Fatigue Test', 'left': MARGIN_LEFT, 'top': FATIGUE_Y, 'width': CONTENT_WIDTH, 'height': Cm(1)},
    'fatigue_index': {'label': 'Fatigue Index', 'path': 'placeholders/fatigue_index.png', 'left': MARGIN_LEFT, 'top': Cm(FATIGUE_Y.cm + 1.5), 'width': Cm(8.5)},
    'muscle_activation': {'label': 'Muscle Activation', 'path': 'placeholders/muscle_activation.png', 'left': Cm(MARGIN_LEFT.cm + 9.5), 'top': Cm(FATIGUE_Y.cm + 1.5), 'width': Cm(8.5)},
}

# -- Footer --
FOOTER_Y = Cm(28.5)
FOOTER_SPECS = {
    'page_num': {'text': '1', 'left': Cm(SLIDE_WIDTH.cm - MARGIN_LEFT.cm - 1), 'top': FOOTER_Y, 'width': Cm(1), 'height': Cm(1)}
}

# --- 2. 預覽函式 (已修正) ---
def preview_layout():
    """使用 Matplotlib 繪製排版示意圖。"""
    fig, ax = plt.subplots(figsize=(SLIDE_WIDTH.cm / 2.54, SLIDE_HEIGHT.cm / 2.54))
    ax.set_xlim(0, SLIDE_WIDTH.cm)
    ax.set_ylim(0, SLIDE_HEIGHT.cm)
    ax.invert_yaxis()
    ax.set_title('Layout Preview')
    ax.set_xlabel('Width (cm)')
    ax.set_ylabel('Top (cm)')

    # 建立一個包含所有要繪製元件的列表
    elements_to_draw = [
        {'label': 'Logo', **HEADER_SPECS['logo']},
        {'label': 'Date', **HEADER_SPECS['date']},
        {'label': 'Player BG Box', 'left': MARGIN_LEFT, 'top': PLAYER_BG_Y, 'width': CONTENT_WIDTH, 'height': PLAYER_BG_HEIGHT},
        {'label': 'Player Name', **PLAYER_SPECS['name']},
        {'label': 'Flick Shot Title', **FLICK_SHOT_SPECS['title']},
        {'label': 'Fatigue Title', **FATIGUE_SPECS['title']},
        {'label': 'Page Num', **FOOTER_SPECS['page_num']},
    ]
    all_image_specs = {**FLICK_SHOT_SPECS, **FATIGUE_SPECS}
    for spec in all_image_specs.values():
        if 'path' in spec:
            # 為圖片提供預設高度，以確保預覽圖能正確顯示
            spec_with_height = {'height': spec.get('height', Cm(4.5)), **spec}
            elements_to_draw.append(spec_with_height)

    # 遍歷所有元件並繪製矩形色塊
    for elem in elements_to_draw:
        try:
            left_cm = elem['left'].cm
            top_cm = elem['top'].cm
            width_cm = elem['width'].cm
            height_cm = elem['height'].cm

            rect = patches.Rectangle(
                (left_cm, top_cm), width_cm, height_cm,
                linewidth=1, edgecolor='r', facecolor='skyblue', alpha=0.6
            )
            ax.add_patch(rect)
            ax.text(
                left_cm + width_cm / 2, top_cm + height_cm / 2,
                elem.get('label', ''), ha='center', va='center', color='black', fontsize=6
            )
        except AttributeError:
            print(f"Error: Could not draw '{elem.get('label', 'Unnamed')}' due to an invalid dimension. Check its spec.")
            continue # 跳過有問題的元件，繼續繪製其他部分

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# --- 3. PPT 生成主函式 (已完整修正結構與邏輯) ---
def create_report(output_filename="science_report.pptx"):
    """根據以上所有設定，生成最終的 PowerPoint 報告。"""
    prs = pptx.Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT
    slide = prs.slides.add_slide(prs.slide_layouts[6]) # 空白版面

    # 1. --- Header ---
    print("Adding Header...")
    add_smart_picture(slide, HEADER_SPECS['logo'])
    
    date_spec = HEADER_SPECS['date']
    tb = slide.shapes.add_textbox(date_spec['left'], date_spec['top'], date_spec['width'], date_spec['height'])
    tb.text_frame.paragraphs[0].text = date_spec['text']
    tb.text_frame.paragraphs[0].font.size = Pt(11)
    tb.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
    
    line = slide.shapes.add_shape(MSO_SHAPE.LINE_INVERSE, MARGIN_LEFT, HEADER_SPECS['line_top'], CONTENT_WIDTH, Pt(1))
    line.line.fill.solid()
    line.line.fill.fore_color.rgb = RGBColor(220, 220, 220)

    # 2. --- Player Background ---
    print("Adding Player Info...")
    bg_box_spec = {'left': MARGIN_LEFT, 'top': PLAYER_BG_Y, 'width': CONTENT_WIDTH, 'height': PLAYER_BG_HEIGHT}
    bg_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, **bg_box_spec)
    bg_box.fill.background()
    bg_box.line.fill.solid()
    bg_box.line.fill.fore_color.rgb = RGBColor(220, 220, 220)
    bg_box.line.width = Pt(1.5)
    bg_box.adjustments[0] = 0.15

    for key in ['name', 'details', 'mouse_title', 'mouse_details']:
        spec = PLAYER_SPECS[key]
        tb = slide.shapes.add_textbox(spec['left'], spec['top'], spec['width'], spec['height'])
        tb.text_frame.text = spec['text']
        if key == 'name':
            tb.text_frame.paragraphs[0].font.size = Pt(24)
            tb.text_frame.paragraphs[0].font.bold = True
        elif key == 'details':
            tb.text_frame.paragraphs[0].font.size = Pt(11)
        elif key == 'mouse_title':
            tb.text_frame.paragraphs[0].font.bold = True
        elif key == 'mouse_details':
            tb.text_frame.paragraphs[0].font.size = Pt(10)
    
    line = slide.shapes.add_shape(MSO_SHAPE.LINE_INVERSE, PLAYER_SPECS['line_left'], PLAYER_SPECS['line_top'], Pt(1), PLAYER_SPECS['line_height'])
    line.line.fill.solid()
    line.line.fill.fore_color.rgb = RGBColor(220, 220, 220)

    # 3. --- Flick Shot Section ---
    print("Adding Flick Shot Section...")
    title_spec = FLICK_SHOT_SPECS['title']
    tb = slide.shapes.add_textbox(title_spec['left'], title_spec['top'], title_spec['width'], title_spec['height'])
    tb.text_frame.paragraphs[0].text = title_spec['text']
    tb.text_frame.paragraphs[0].font.size = Pt(16)

    for key in ['hand_path', 'speed_profile', 'performance', 'forearm_muscles', 'muscle_charts_combined']:
        add_smart_picture(slide, FLICK_SHOT_SPECS[key])

    # 4. --- Fatigue Test Section ---
    print("Adding Fatigue Test Section...")
    title_spec = FATIGUE_SPECS['title']
    geometry_spec = {k: v for k, v in title_spec.items() if k != 'text'}
    title_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, **geometry_spec)
    title_shape.fill.solid()
    title_shape.fill.fore_color.rgb = RGBColor(237, 28, 36)
    title_shape.text_frame.text = title_spec['text']
    p = title_shape.text_frame.paragraphs[0]
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.font.bold = True
    p.font.size = Pt(14)
    
    for key in ['fatigue_index', 'muscle_activation']:
        add_smart_picture(slide, FATIGUE_SPECS[key])

    # 5. --- Footer ---
    print("Adding Footer...")
    footer_spec = FOOTER_SPECS['page_num']
    tb = slide.shapes.add_textbox(footer_spec['left'], footer_spec['top'], footer_spec['width'], footer_spec['height'])
    p = tb.text_frame.paragraphs[0]
    p.text = footer_spec['text']
    p.font.size = Pt(10)
    p.alignment = PP_ALIGN.RIGHT

    # 6. --- Save Presentation ---
    prs.save(output_filename)
    print(f"Report successfully saved as '{output_filename}'")

# --- Main Execution ---
if __name__ == '__main__':
    create_placeholder_images()
    preview_layout()
    create_report()