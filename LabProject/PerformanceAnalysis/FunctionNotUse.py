# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 14:16:02 2025

@author: Hsin.YH.Yang
"""


# --- 2. 預覽函式 (已修正) ---
# def preview_layout():
#     """使用 Matplotlib 繪製排版示意圖。"""
#     fig, ax = plt.subplots(figsize=(SLIDE_WIDTH.cm / 2.54, SLIDE_HEIGHT.cm / 2.54))
#     ax.set_xlim(0, SLIDE_WIDTH.cm)
#     ax.set_ylim(0, SLIDE_HEIGHT.cm)
#     ax.invert_yaxis()
#     ax.set_title('Layout Preview')
#     ax.set_xlabel('Width (cm)')
#     ax.set_ylabel('Top (cm)')

#     # 建立一個包含所有要繪製元件的列表
#     elements_to_draw = [
#         {'label': 'Logo', **HEADER_SPECS['logo']},
#         {'label': 'Date', **HEADER_SPECS['date']},
#         {'label': 'Player BG Box', 'left': MARGIN_LEFT, 'top': PLAYER_BG_Y, 'width': CONTENT_WIDTH, 'height': PLAYER_BG_HEIGHT},
#         {'label': 'Player Name', **PLAYER_SPECS['name']},
#         {'label': 'Flick Shot Title', **FLICK_SHOT_SPECS['title']},
#         {'label': 'Fatigue Title', **FATIGUE_SPECS['title']},
#         {'label': 'Page Num', **FOOTER_SPECS['page_num']},
#     ]
#     all_image_specs = {**FLICK_SHOT_SPECS, **FATIGUE_SPECS}
#     for spec in all_image_specs.values():
#         if 'path' in spec:
#             # 為圖片提供預設高度，以確保預覽圖能正確顯示
#             spec_with_height = {'height': spec.get('height', Cm(4.5)), **spec}
#             elements_to_draw.append(spec_with_height)

#     # 遍歷所有元件並繪製矩形色塊
#     for elem in elements_to_draw:
#         try:
#             left_cm = elem['left'].cm
#             top_cm = elem['top'].cm
#             width_cm = elem['width'].cm
#             height_cm = elem['height'].cm

#             rect = patches.Rectangle(
#                 (left_cm, top_cm), width_cm, height_cm,
#                 linewidth=1, edgecolor='r', facecolor='skyblue', alpha=0.6
#             )
#             ax.add_patch(rect)
#             ax.text(
#                 left_cm + width_cm / 2, top_cm + height_cm / 2,
#                 elem.get('label', ''), ha='center', va='center', color='black', fontsize=6
#             )
#         except AttributeError:
#             print(f"Error: Could not draw '{elem.get('label', 'Unnamed')}' due to an invalid dimension. Check its spec.")
#             continue # 跳過有問題的元件，繼續繪製其他部分

#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.show()



# --- 3. PPT 生成主函式 (已完整修正結構與邏輯) ---
# def create_report(output_filename="science_report.pptx"):
#     """根據以上所有設定，生成最終的 PowerPoint 報告。"""
#     prs = pptx.Presentation()
#     prs.slide_width = SLIDE_WIDTH
#     prs.slide_height = SLIDE_HEIGHT
#     slide = prs.slides.add_slide(prs.slide_layouts[6]) # 空白版面

#     # 1. --- Header ---
#     print("Adding Header...")
#     add_smart_picture(slide, HEADER_SPECS['logo'])
    
#     date_spec = HEADER_SPECS['date']
#     tb = slide.shapes.add_textbox(date_spec['left'], date_spec['top'], date_spec['width'], date_spec['height'])
#     tb.text_frame.paragraphs[0].text = date_spec['text']
#     tb.text_frame.paragraphs[0].font.size = Pt(11)
#     tb.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
    
#     line = slide.shapes.add_shape(MSO_SHAPE.LINE_INVERSE, MARGIN_LEFT, HEADER_SPECS['line_top'], CONTENT_WIDTH, Pt(1))
#     line.line.fill.solid()
#     line.line.fill.fore_color.rgb = RGBColor(220, 220, 220)

#     # 2. --- Player Background ---
#     print("Adding Player Info...")
#     # Add the main rounded rectangle outline
#     bg_box_spec = {'left': MARGIN_LEFT, 'top': PLAYER_BG_Y, 'width': CONTENT_WIDTH, 'height': PLAYER_BG_HEIGHT}
#     bg_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, **bg_box_spec)
#     bg_box.fill.background()
#     bg_box.line.fill.solid()
#     bg_box.line.fill.fore_color.rgb = RGBColor(220, 220, 220)
#     bg_box.line.width = Pt(1.5)
#     bg_box.adjustments[0] = 0.15

#     # Add Player Name and Details (this part is the same)
#     for key in ['name', 'details']:
#         spec = PLAYER_SPECS[key]
#         tb = slide.shapes.add_textbox(spec['left'], spec['top'], spec['width'], spec['height'])
#         tb.text_frame.text = spec['text']
#         if key == 'name':
#             tb.text_frame.paragraphs[0].font.size = Pt(24)
#             tb.text_frame.paragraphs[0].font.bold = True
#         elif key == 'details':
#             tb.text_frame.paragraphs[0].font.size = Pt(11)

#     # Add the vertical separator line (this part is the same)
#     line = slide.shapes.add_shape(MSO_SHAPE.LINE_INVERSE, PLAYER_SPECS['line_left'], PLAYER_SPECS['line_top'], Pt(1), PLAYER_SPECS['line_height'])
#     line.line.fill.solid()
#     line.line.fill.fore_color.rgb = RGBColor(220, 220, 220)
    
#     # ================== NEW LOGIC FOR MOUSE PREFERENCES ==================
#     # Add the new title: "Mouse 1 Preferences"
#     title_spec = PLAYER_SPECS['mouse_pref_title']
#     tb = slide.shapes.add_textbox(title_spec['left'], title_spec['top'], title_spec['width'], title_spec['height'])
#     tb.text_frame.text = title_spec['text']
#     tb.text_frame.paragraphs[0].font.size = Pt(16)
#     # tb.text_frame.paragraphs[0].font.bold = True
    
#     # Add the red line separator
#     red_line_spec = PLAYER_SPECS['mouse_pref_title'] # Use title spec for positioning
#     line = slide.shapes.add_shape(
#         MSO_SHAPE.LINE_INVERSE, 
#         red_line_spec['left'], PLAYER_SPECS['mouse_pref_line_top'], 
#         red_line_spec['width'], Pt(1.5)
#     )
#     line.line.fill.solid()
#     line.line.fill.fore_color.rgb = RGBColor(237, 28, 36) # Red color

#     # Add and format the 2x2 table
#     table_spec = PLAYER_SPECS['mouse_pref_table']
#     table_data = table_spec['data']
#     rows, cols = len(table_data), len(table_data[0])
    
#     shape = slide.shapes.add_table(
#         rows, cols, 
#         table_spec['left'], table_spec['top'], 
#         table_spec['width'], Cm(1.5) # Height is less important here
#     )
#     table = shape.table

#     # Set column widths
#     for i, width in enumerate(table_spec['col_widths']):
#         table.columns[i].width = width

#     # Populate and format each cell
#     for r, row_data in enumerate(table_data):
#         for c, cell_text in enumerate(row_data):
#             cell = table.cell(r, c)
            
#             # Make cell background transparent
#             cell.fill.background()
            
#             p = cell.text_frame.paragraphs[0]
#             p.text = cell_text
#             p.font.size = Pt(11)
#             cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            
#             # Left column (c=0) formatting
#             if c == 0:
#                 p.font.color.rgb = RGBColor(166, 166, 166) # Grey text
#                 p.alignment = PP_ALIGN.LEFT
#             # Right column (c=1) formatting
#             else:
#                 p.font.color.rgb = RGBColor(0, 0, 0) # Black text
#                 p.alignment = PP_ALIGN.RIGHT

#     # 3. --- Flick Shot Section ---
#     print("Adding Flick Shot Section...")
#     # Add the main section title
#     title_spec = FLICK_SHOT_SPECS['title']
#     tb = slide.shapes.add_textbox(title_spec['left'], title_spec['top'], title_spec['width'], title_spec['height'])
#     tb.text_frame.paragraphs[0].text = title_spec['text']
#     tb.text_frame.paragraphs[0].font.size = Pt(16)

#     # --- Add all the small titles above the charts ---
#     for key in ['hand_path_title', 'speed_profile_title', 'performance_title', 'forearm_muscles_title']:
#         spec = FLICK_SHOT_SPECS[key]
#         tb = slide.shapes.add_textbox(spec['left'], spec['top'], spec['width'], spec['height'])
#         tb.text_frame.text = spec['text']
#         tb.text_frame.paragraphs[0].font.size = Pt(11)

#     # --- Add all the images using the corrected key names ---
#     image_keys = [
#         'hand_path_image', 'speed_profile_image', 'performance_image', 
#         'forearm_muscles_image', 'muscle_charts_combined'
#     ]
#     for key in image_keys:
#         add_smart_picture(slide, FLICK_SHOT_SPECS[key])

#     # 4. --- Fatigue Test Section ---
#     print("Adding Fatigue Test Section...")
#     # Add the main red title box
#     title_spec = FATIGUE_SPECS['title']
#     geometry_spec = {k: v for k, v in title_spec.items() if k != 'text'}
#     title_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, **geometry_spec)
#     title_shape.fill.solid()
#     title_shape.fill.fore_color.rgb = RGBColor(237, 28, 36)
#     title_shape.text_frame.text = title_spec['text']
#     p = title_shape.text_frame.paragraphs[0]
#     p.font.color.rgb = RGBColor(255, 255, 255)
#     p.font.bold = True
#     p.font.size = Pt(14)

#     # --- Add all the small titles above the charts ---
#     for key in ['fatigue_index_title', 'muscle_activation_title']:
#         spec = FATIGUE_SPECS[key]
#         tb = slide.shapes.add_textbox(spec['left'], spec['top'], spec['width'], spec['height'])
#         tb.text_frame.text = spec['text']
#         tb.text_frame.paragraphs[0].font.size = Pt(11)

#     # --- Add all the images using the corrected key names ---
#     image_keys = ['fatigue_index_image', 'muscle_activation_image']
#     for key in image_keys:
#         add_smart_picture(slide, FATIGUE_SPECS[key])

#     # --- Add the (x / x) text at the bottom ---
#     spec = FATIGUE_SPECS['bottom_text']
#     tb = slide.shapes.add_textbox(spec['left'], spec['top'], spec['width'], spec['height'])
#     tb.text_frame.text = spec['text']
#     p = tb.text_frame.paragraphs[0]
#     p.font.size = Pt(10)
#     p.alignment = PP_ALIGN.CENTER

#     # 5. --- Footer ---
#     print("Adding Footer...")
#     footer_spec = FOOTER_SPECS['page_num']
#     tb = slide.shapes.add_textbox(footer_spec['left'], footer_spec['top'], footer_spec['width'], footer_spec['height'])
#     p = tb.text_frame.paragraphs[0]
#     p.text = footer_spec['text']
#     p.font.size = Pt(10)
#     p.alignment = PP_ALIGN.RIGHT

#     # 6. --- Save Presentation ---
#     prs.save(output_filename)
#     print(f"Report successfully saved as '{output_filename}'")



# --- 2. 預覽函式 (已升級為可顯示圖片內容) ---
# def preview_layout():
#     """使用 Matplotlib 繪製排版示意圖，並顯示圖片內容。"""
#     fig, ax = plt.subplots(figsize=(SLIDE_WIDTH.cm / 2.54, SLIDE_HEIGHT.cm / 2.54))
#     ax.set_xlim(0, SLIDE_WIDTH.cm)
#     ax.set_ylim(0, SLIDE_HEIGHT.cm)
#     ax.invert_yaxis()
#     ax.set_title('Layout Preview (with Images)')
#     ax.set_xlabel('Width (cm)')
#     ax.set_ylabel('Top (cm)')

#     # 建立一個包含所有要繪製元件的列表
#     elements_to_draw = [
#         {'label': 'Date', **HEADER_SPECS['date']},
#         {'label': 'Player BG Box', 'left': MARGIN_LEFT, 'top': PLAYER_BG_Y, 'width': CONTENT_WIDTH, 'height': PLAYER_BG_HEIGHT},
#         {'label': 'Player Name', **PLAYER_SPECS['name']},
#         {'label': 'Flick Shot Title', **FLICK_SHOT_SPECS['title']},
#         {'label': 'Fatigue Title', **FATIGUE_SPECS['title']},
#         {'label': 'Page Num', **FOOTER_SPECS['page_num']},
#     ]
#     # 將所有圖片規格合併，以便處理
#     all_image_specs = {**HEADER_SPECS, **FLICK_SHOT_SPECS, **FATIGUE_SPECS}
#     for spec in all_image_specs.values():
#         if isinstance(spec, dict) and 'path' in spec:
#             spec_with_height = {'height': spec.get('height', Cm(4.5)), **spec}
#             elements_to_draw.append(spec_with_height)

#     # 遍歷所有元件並繪製
#     for elem in elements_to_draw:
#         try:
#             left_cm = elem['left'].cm
#             top_cm = elem['top'].cm
#             width_cm = elem['width'].cm
#             height_cm = elem['height'].cm

#             # ✨ 新增的邏輯：檢查元件是否為圖片
#             if 'path' in elem:
#                 try:
#                     image_data = None
#                     # 如果是 SVG，先轉換成 PNG
#                     if elem['path'].lower().endswith('.svg'):
#                         png_output = io.BytesIO()
#                         cairosvg.svg2png(url=elem['path'], write_to=png_output)
#                         png_output.seek(0)
#                         image_data = plt.imread(png_output)
#                     # 如果是 PNG 或其他格式，直接讀取
#                     else:
#                         image_data = plt.imread(elem['path'])
                    
#                     # 在指定位置顯示圖片
#                     ax.imshow(image_data, extent=[left_cm, left_cm + width_cm, top_cm, top_cm + height_cm])

#                 except FileNotFoundError:
#                     print(f"Preview warning: Image file not found at '{elem['path']}'. Drawing placeholder.")
#                     # 如果找不到圖片，則畫一個紅色錯誤框
#                     rect = patches.Rectangle((left_cm, top_cm), width_cm, height_cm, linewidth=2, edgecolor='red', facecolor='pink')
#                     ax.add_patch(rect)
#                     ax.text(left_cm + width_cm / 2, top_cm + height_cm / 2, f"File Not Found:\n{elem.get('label', '')}", ha='center', va='center', color='red', fontsize=6)
            
#             # ✨ 如果不是圖片，則畫藍色佔位符
#             else:
#                 rect = patches.Rectangle((left_cm, top_cm), width_cm, height_cm, linewidth=1, edgecolor='r', facecolor='skyblue', alpha=0.6)
#                 ax.add_patch(rect)
#                 ax.text(left_cm + width_cm / 2, top_cm + height_cm / 2, elem.get('label', ''), ha='center', va='center', color='black', fontsize=6)

#         except Exception as e:
#             print(f"Error drawing '{elem.get('label', 'Unnamed')}': {e}")
#             continue

#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.show()