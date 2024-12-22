from flask import Flask, render_template
import requests

# Khởi tạo Flask app
app = Flask(__name__)

# Route chính cho Dashboard
@app.route('/')
def index():
    # Dữ liệu mặc định cho hai bot
    data_xau = {"current_price": "N/A", "trend": "N/A", "position_type": "N/A",
                "entry_price": "N/A", "balance": "N/A", "pnl": "N/A",
                "tpo_poc_price": "N/A"}
    data_btc = {"current_price": "N/A", "trend": "N/A", "position_type": "N/A",
                "entry_price": "N/A", "balance": "N/A", "pnl": "N/A",
                "tpo_poc_price": "N/A"}

    # Lấy dữ liệu từ bot XAUUSD (8080)
    try:
        print("Đang kết nối đến bot XAUUSD...")
        response_xau = requests.get("http://localhost:8080/status", timeout=5)
        if response_xau.status_code == 200:
            data_xau = response_xau.json()
    except Exception as e:
        print(f"Lỗi khi kết nối đến bot XAUUSD: {e}")

    # Lấy dữ liệu từ bot BTCUSD (5000)
    try:
        print("Đang kết nối đến bot BTCUSD...")
        response_btc = requests.get("http://localhost:5000/status", timeout=5)
        if response_btc.status_code == 200:
            data_btc = response_btc.json()
    except Exception as e:
        print(f"Lỗi khi kết nối đến bot BTCUSD: {e}")

    # Render giao diện với dữ liệu từ hai bot
    return render_template('dashboard.html', data_xau=data_xau, data_btc=data_btc)

# Route kiểm tra trạng thái Dashboard
@app.route('/ping')
def ping():
    return "Dashboard is running!"

# Chạy ứng dụng Flask
if __name__ == "__main__":
    app.run(port=6060, debug=True, use_reloader=False)
