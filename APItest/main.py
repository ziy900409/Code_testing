from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "資料分析後端已啟動成功！"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # 先計算平均值
    mean_values = df.mean(numeric_only=True).to_dict()

    # 繪圖
    plt.figure(figsize=(8, 5))
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.plot(df[col], label=col)
    plt.title("Uploaded Data Plot")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    
    # 把圖存進 memory 中
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="png")
    img_bytes.seek(0)
    plt.close()

    # 轉成 base64
    img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

    return JSONResponse(content={
        "mean_values": mean_values,
        "plot_base64": img_base64
    })
