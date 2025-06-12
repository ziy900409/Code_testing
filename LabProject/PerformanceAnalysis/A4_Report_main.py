import pptx
import io
# 移除 cairosvg，因為我們在 Conda 環境中直接使用 cairocffi
# import cairosvg 
import cairocffi as cairo # Conda 安裝後，cairosvg 會使用這個

from pptx.util import Pt, Cm
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

# 引入 matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 參數設定 (所有元件的尺寸和位置) ---

# 0. 投影片尺寸 (A4 直向)
SLIDE_WIDTH_CM = 21.0
SLIDE_HEIGHT_CM = 29.7

# 1. 圖片設定
SVG_PATH = r"D:\BenQ_Project\01_UR_lab\2025_02 Lab 體驗量測\fig\Forearm Muscles.svg"
IMG_SPECS = {
    'label': 'Logo Image',
    'left': Cm(1.5), 'top': Cm(1.5),
    'width': Cm(10), 'height': Cm(6) # 預估一個高度以便繪圖
}

# 2. 表格一設定
TABLE1_DATA = [
    ['產品名稱', '季度一', '季度二', '季度三', '季度四'],
    ['產品A', 150, 200, 220, 280]
]
TABLE1_SPECS = {
    'label': 'Table 1: Sales',
    'left': Cm(1.5), 'top': Cm(12),
    'width': Cm(18), 'height': Cm(2)
}

# 3. 表格二設定
TABLE2_DATA = [
    ['部門', '預算 (萬)', '實際花費 (萬)'],
    ['行銷部', 50, 45]
]
TABLE2_SPECS = {
    'label': 'Table 2: Budget',
    'left': Cm(1.5), 'top': Cm(18),
    'width': Cm(18), 'height': Cm(2)
}


# --- 預覽函式 ---
def preview_layout(slide_width, slide_height, elements):
    """
    使用 Matplotlib 繪製排版示意圖。
    - slide_width, slide_height: 投影片的總寬高 (in Cm)
    - elements: 一個包含所有元件規格的列表
    """
    fig, ax = plt.subplots(figsize=(slide_width / 2.54, slide_height / 2.54)) # 轉換為英寸
    
    # 設定畫布尺寸，符合投影片
    ax.set_xlim(0, slide_width)
    ax.set_ylim(0, slide_height)
    
    # PowerPoint 的座標原點在左上角，Y軸向下。Matplotlib 的在左下角，Y軸向上。
    # 我們將 Matplotlib 的 Y 軸反轉以匹配 PowerPoint 的座標系，這樣更直觀。
    ax.invert_yaxis()
    
    ax.set_aspect('equal', adjustable='box')
    plt.title('Slide Layout Preview')
    plt.xlabel('Width (cm)')
    plt.ylabel('Top (cm)')
    
    # 遍歷所有元件並繪製矩形色塊
    for elem in elements:
        # 將 pptx 的 Cm 單位轉換為純數字
        left_cm = elem['left'].cm
        top_cm = elem['top'].cm
        width_cm = elem['width'].cm
        height_cm = elem['height'].cm
        
        # 建立矩形
        rect = patches.Rectangle(
            (left_cm, top_cm), width_cm, height_cm,
            linewidth=1, edgecolor='r', facecolor='skyblue', alpha=0.6
        )
        ax.add_patch(rect)
        
        # 在矩形中心加上標籤
        ax.text(
            left_cm + width_cm / 2, 
            top_cm + height_cm / 2, 
            elem['label'],
            ha='center', va='center', color='black', fontsize=8
        )
        
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# --- 主要程式碼 ---

def create_presentation_with_preview(output_filename):
    """
    先顯示預覽圖，再生成 PPT。
    """
    # 1. 顯示排版預覽圖
    print("Displaying layout preview in Spyder's 'Plots' pane...")
    all_elements = [IMG_SPECS, TABLE1_SPECS, TABLE2_SPECS]
    preview_layout(SLIDE_WIDTH_CM, SLIDE_HEIGHT_CM, all_elements)

    # 2. 生成 PPT (這部分程式碼和之前一樣)
    print("Generating actual PowerPoint file...")
    prs = pptx.Presentation()
    prs.slide_width = Cm(SLIDE_WIDTH_CM)
    prs.slide_height = Cm(SLIDE_HEIGHT_CM)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # ... (此處省略了 add_picture 和 add_table 的完整程式碼，它們會使用上面定義的 SPECS) ...
    # 例如: pic = slide.shapes.add_picture(png_output, IMG_SPECS['left'], IMG_SPECS['top'], width=IMG_SPECS['width'])
    
    # prs.save(output_filename)
    print(f"To generate the file, uncomment the 'prs.save()' line.")


if __name__ == '__main__':
    create_presentation_with_preview('final_presentation.pptx')