# %%
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

# 啟動瀏覽器
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 不開啟視窗
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 開啟頁面
url = "https://prosettings.net/lists/cs2/"
driver.get(url)
time.sleep(5)  # 等待 JavaScript 載入資料

# 擷取 HTML 並解析
soup = BeautifulSoup(driver.page_source, 'html.parser')
driver.quit()

# 擷取表格資料
table = soup.find('table')
headers = [th.text.strip() for th in table.find_all('th')]
rows = [
    [td.text.strip() for td in tr.find_all('td')]
    for tr in table.find_all('tr')[1:]
]

df = pd.DataFrame(rows, columns=headers)
df.to_csv("cs2_settings.csv", index=False)
print("✅ 資料已成功儲存！")


# %%
