"""
http://127.0.0.1:5000
"""


from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

# 确保事件日志文件夹存在
if not os.path.exists('logs'):
    os.makedirs('logs')

# 路由到主页
@app.route('/')
def index():
    return render_template('index.html')

# 处理鼠标事件的路由
@app.route('/log_mouse_event', methods=['POST'])
def log_mouse_event():
    data = request.json
    print(data)  # 在服务器端打印鼠标事件数据

    # 将事件记录到日志文件中
    with open('logs/mouse_events.log', 'a') as f:
        f.write(json.dumps(data) + '\n')

    return jsonify(status="success")

# 处理结束测试的路由
@app.route('/end_test', methods=['POST'])
def end_test():
    # 读取日志文件中的所有事件
    with open('logs/mouse_events.log', 'r') as f:
        events = f.readlines()
    
    # 输出所有事件到一个结果文件中
    with open('logs/mouse_events_result.json', 'w') as f:
        f.write(json.dumps(events, indent=4))
    
    return jsonify(status="success")

if __name__ == '__main__':
    app.run(debug=True)
