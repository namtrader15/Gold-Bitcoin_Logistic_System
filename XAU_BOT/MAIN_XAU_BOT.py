from flask import Flask, render_template, jsonify, request
import time
import threading
import MetaTrader5 as mt5
from Entry_Super_XAU import get_final_trend_XAU  # Hàm lấy xu hướng
from TPO_POC import calculate_poc_value_XAU  # Hàm tính POC
from place_order import place_order_mt5  # Hàm thực hiện lệnh giao dịch

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Thông tin tài khoản MT5
MT5_ACCOUNT = 7510016
MT5_PASSWORD = "7lTa+zUw"
MT5_SERVER = "VantageInternational-Demo"
RISK_AMOUNT = 60  # USD

# Biến lưu trữ trạng thái giao dịch và điều khiển bot
trade_status = {
    "position_type": None,
    "profit": 0.0,
    "balance": 0.0,
    "status": "Chưa có vị thế",
    "trend": "N/A"
}
bot_running = False  # Biến để điều khiển trạng thái bot

# Kết nối MT5
def connect_mt5():
    if not mt5.initialize():
        print("Lỗi khi khởi động MT5:", mt5.last_error())
        return False
    authorized = mt5.login(MT5_ACCOUNT, password=MT5_PASSWORD, server=MT5_SERVER)
    if not authorized:
        print("Lỗi kết nối đến MT5")
        mt5.shutdown()
        return False
    print("Kết nối thành công đến MT5")
    return True

# Hàm lấy số dư tài khoản từ MT5
def get_account_balance():
    account_info = mt5.account_info()
    return account_info.balance if account_info else None

# Lấy thông tin vị thế hiện tại
def get_position_info():
    positions = mt5.positions_get(symbol="XAUUSD")  # Thay BTCUSD bằng XAUUSD
    if positions:
        position = positions[0]
        return {
            "type": "Buy" if position.type == mt5.ORDER_TYPE_BUY else "Sell",
            "profit": position.profit,
            "volume": position.volume,
            "ticket": position.ticket
        }
    return None
def get_positions_pnl_xau(symbol="XAUUSD"):
    """
    Hàm lấy tổng PNL của tất cả các lệnh mở trên XAUUSD.
    """
    positions = mt5.positions_get(symbol=symbol)  # Lấy danh sách vị thế cho XAUUSD
    if positions:
        total_pnl = sum(position.profit for position in positions)  # Tổng PNL của tất cả các lệnh mở
        return round(total_pnl, 2)  # Làm tròn đến 2 chữ số thập phân
    print(f"Không có vị thế nào cho {symbol}.")
    return 0.0

# Hàm cập nhật trạng thái giao dịch
def update_trade_status():
    balance = get_account_balance()  # Lấy số dư hiện tại từ MT5
    pnl_xau = get_positions_pnl_xau("XAUUSD")  # Lấy PNL của XAUUSD
    position_info = get_position_info()  # Lấy thông tin vị thế hiện tại

    trade_status["balance"] = balance if balance is not None else "N/A"
    trade_status["pnl"] = pnl_xau if pnl_xau is not None else "N/A"  # Gán PNL của XAUUSD
    trade_status["trend"] = get_trend() if bot_running else "Bot đang tạm dừng"
    
    if position_info:
        trade_status["position_type"] = position_info["type"]
        trade_status["profit"] = position_info["profit"]
        trade_status["status"] = "Đang có vị thế"
    else:
        trade_status["position_type"] = None
        trade_status["profit"] = 0.0
        trade_status["status"] = "Chưa có vị thế"

    #
def get_account_pnl():
    account_info = mt5.account_info()
    if account_info:
        return round(account_info.profit, 2)  # Trả về PNL làm tròn đến 2 chữ số thập phân
    print("Không thể lấy thông tin PNL từ MT5.")
    return None

# Hàm lấy xu hướng hiện tại từ hàm phân tích `get_final_trend_XAU`
def get_trend():
    trend = get_final_trend_XAU()  # Không cần client Binance nữa
    print(f"Kết quả xu hướng XAUUSD hiện tại: {trend}")
    return trend

# Kiểm tra giá POC và thực hiện lệnh
def check_poc_and_place_order(final_trend, symbol="XAUUSD"):  
    position = get_position_info()
    if position:
        print("Đã có một vị thế mở. Theo dõi vị thế hiện tại và không mở thêm lệnh.")
        close_position_if_needed(position)
        return

    mark_price = get_realtime_price_mt5(symbol)
    if mark_price is None:
        return

    poc_value = calculate_poc_value_XAU()  # Chắc chắn rằng `calculate_poc_value` có thể làm việc với XAUUSD
    price_difference_percent = abs((poc_value - mark_price) / mark_price) * 100
    print(f"Chênh lệch giữa POC và mark price: {price_difference_percent:.2f}%")

    if price_difference_percent <= 0.25:
        if final_trend == "Xu hướng tăng":
            print("Xu hướng tăng. POC value gần mark price. Thực hiện lệnh mua.")
            place_order_mt5("buy", symbol, risk_amount=RISK_AMOUNT)
        elif final_trend == "Xu hướng giảm":
            print("Xu hướng giảm. POC value gần mark price. Thực hiện lệnh bán.")
            place_order_mt5("sell", symbol, risk_amount=RISK_AMOUNT)
    else:
        print("Không thực hiện lệnh vì chênh lệch vượt quá 0.25%.")

# Hàm đóng lệnh
def close_position(position):
    order_type = mt5.ORDER_TYPE_SELL if position["type"] == "Buy" else mt5.ORDER_TYPE_BUY
    
    # Tạo yêu cầu đóng lệnh với chế độ IOC
    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": "XAUUSD",  # Thay BTCUSD bằng XAUUSD
        "volume": position["volume"],
        "type": order_type,
        "position": position["ticket"],
        "deviation": 20,
        "magic": 234000,
        "type_filling": mt5.ORDER_FILLING_IOC,  # Áp dụng chế độ khớp lệnh IOC
    }
    
    # Log thông tin yêu cầu
    print("Yêu cầu đóng lệnh (IOC):", close_request)
    
    # Gửi yêu cầu
    result = mt5.order_send(close_request)
    
    # Kiểm tra kết quả
    if result is None:
        print("Gửi lệnh thất bại. Lỗi:", mt5.last_error())
        return False
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print("Đóng lệnh thành công:", result)
        return True
    else:
        print(f"Đóng lệnh thất bại. Mã lỗi: {result.retcode}, Thông tin chi tiết: {result}")
        return False

# Lấy giá real-time từ MT5
def get_realtime_price_mt5(symbol):
    tick = mt5.symbol_info_tick(symbol)
    return tick.ask if tick else None

# Vòng lặp kiểm tra xu hướng và vị thế
def trading_loop():
    while bot_running:
        final_trend = get_trend()
        
        # Nếu không rõ xu hướng, nghỉ 600 giây
        if final_trend == "Xu hướng không rõ ràng":
            print("Xu hướng không rõ ràng. Nghỉ 600 giây trước khi kiểm tra lại.")
            time.sleep(600)  # Nghỉ 600 giây
            continue

        # Kiểm tra vị thế hiện tại
        position = get_position_info()
        if position:
            current_position_type = position["type"]  # "Buy" hoặc "Sell"
            current_profit = position["profit"]
            
            # Kiểm tra mức lỗ
            if current_profit < -RISK_AMOUNT:
                print(f"Lỗ vượt ngưỡng {RISK_AMOUNT} USD. Đóng lệnh.")
                close_position(position)
                continue  # Tiếp tục vòng lặp sau khi đóng lệnh
            
            # Kiểm tra mức lãi
            if current_profit > 1.7 * RISK_AMOUNT:
                print(f"Lãi vượt ngưỡng {1.7 * RISK_AMOUNT} USD. Đóng lệnh.")
                close_position(position)
                continue  # Tiếp tục vòng lặp sau khi đóng lệnh
            
            # Nếu xu hướng đối nghịch với vị thế, đóng lệnh
            if (current_position_type == "Buy" and final_trend == "Xu hướng giảm") or \
               (current_position_type == "Sell" and final_trend == "Xu hướng tăng"):
                print(f"Xu hướng đối nghịch với vị thế {current_position_type}. Đóng lệnh.")
                close_position(position)
                continue  # Tiếp tục vòng lặp sau khi đóng lệnh
            
            # Nếu không đối nghịch, bot tiếp tục theo dõi vị thế
            print("Xu hướng phù hợp với vị thế hiện tại. Tiếp tục theo dõi.")
        
        else:
            # Nếu không có vị thế, kiểm tra điều kiện mở lệnh
            print("Không có vị thế. Kiểm tra cơ hội giao dịch mới.")
            check_poc_and_place_order(final_trend)

        # Nghỉ 60 giây trước lần kiểm tra tiếp theo
        print("Nghỉ 60 giây trước khi kiểm tra lại.")
        time.sleep(60)

# Hàm để bắt đầu bot
def start_bot():
    global bot_running
    if not bot_running:
        bot_running = True
        threading.Thread(target=trading_loop).start()

# Hàm để tạm dừng bot
def pause_bot():
    global bot_running
    bot_running = False

# Route chính hiển thị giao diện
@app.route('/')
def index():
    update_trade_status()
    return render_template('index.html', trade_status=trade_status)

#Button buy,sell,close
@app.route('/buy_market', methods=['POST'])
def buy_market():
    client = mt5  # Tham số `client`, giả định nó là một phần của MetaTrader5
    result = place_order_mt5(client, order_type="buy", symbol="XAUUSD", risk_amount=RISK_AMOUNT)
    return jsonify({"message": "Lệnh Buy Market đã được thực hiện." if result else "Thất bại khi thực hiện lệnh Buy Market."})
@app.route('/sell_market', methods=['POST'])
def sell_market():
    client = mt5  # Tham số `client`, giả định nó là một phần của MetaTrader5
    result = place_order_mt5(client, order_type="sell", symbol="XAUUSD", risk_amount=RISK_AMOUNT)
    return jsonify({"message": "Lệnh Sell Market đã được thực hiện." if result else "Thất bại khi thực hiện lệnh Sell Market."})

@app.route('/close_market', methods=['POST'])
def close_market():
    # Gọi trực tiếp hàm close_position nếu có vị thế
    position = get_position_info()
    if position:
        result = close_position(position)
        return jsonify({"message": "Đóng lệnh thành công" if result else "Thất bại khi đóng lệnh"})
    return jsonify({"message": "Không có vị thế nào để đóng."})

# Route để lấy trạng thái dưới dạng JSON
@app.route('/status')
def status():
    update_trade_status()
    return jsonify({
        "current_price": get_realtime_price_mt5("XAUUSD"),
        "trend": trade_status["trend"],
        "position_type": trade_status["position_type"],
        "entry_price": trade_status.get("entry_price"),  # Giá mở lệnh
        "balance": trade_status["balance"],  # Số dư tài khoản
        "pnl": trade_status["pnl"],  # PNL chỉ của XAUUSD
        "tpo_poc_price": calculate_poc_value_XAU()  # Giá TPO/POC
    })

# Route để bắt đầu bot
@app.route('/start_bot', methods=['POST'])
def start_bot_route():
    start_bot()
    return jsonify({"message": "Bot đã bắt đầu chạy"})

# Route để tạm dừng bot
@app.route('/pause_bot', methods=['POST'])
def pause_bot_route():
    pause_bot()
    return jsonify({"message": "Bot đã dừng"})

# Chạy Flask server và vòng lặp kiểm tra giao dịch song song
if __name__ == '__main__':
    if connect_mt5():
        print("Khởi động bot giao dịch VÀNG trên server Flask.")
        # Tự động bắt đầu bot
        start_bot()  
        app.run(debug=True, use_reloader=False, port=8080)
    else:
        print("Không thể kết nối đến MT5.")
        mt5.shutdown()
