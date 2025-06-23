from pptx import Presentation
from pptx.util import Cm
from pptx.enum.text import PP_ALIGN

# --- 1. A4 尺寸投影片 ---
prs = Presentation()
prs.slide_width = Cm(21.0)
prs.slide_height = Cm(29.7)

slide_layout = prs.slide_layouts[6] # '空白' 版面
slide = prs.slides.add_slide(slide_layout)

# --- 2. 精確的尺寸和位置 ---
rows, cols = 4, 3
# 使用您先前提供的尺寸
left = Cm(0.6354)
top = Cm(18.7443)
width = Cm(19.7327)
height = Cm(9.0721)

table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
table = table_shape.table

# --- 準備資料並填入表格 ---
data = [
    ['編號', '產品名稱', '價格'],
    ['001', '筆記型電腦', 'NT$ 45,000'],
    ['002', '智慧型手機', 'NT$ 28,000'],
    ['003', '無線耳機', 'NT$ 5,500']
]

for r in range(rows):
    for c in range(cols):
        cell = table.cell(r, c)
        cell.text = str(data[r][c])
        cell.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

# --- 3. 【全新策略】套用一個內建的表格樣式 ---
# 我們不再逐一設定邊框，而是套用一個乾淨、有格線的樣式
# 'Table Grid' 是一個通用樣式，其 GUID 如下：
# 您也可以在 PowerPoint 中自行設計樣式，然後找到其 GUID 來使用
table_style_guid = '{2D5ABB26-0587-4C30-8999-92F81FD0307C}' # 這是 "No Style, Table Grid" 樣式
table.apply_style(table_style_guid)


# --- 儲存檔案 ---
prs.save("Final_Styled_Table.pptx")
print("簡報 'Final_Styled_Table.pptx' 已成功生成！")