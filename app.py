from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import os
import io
import base64
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import math
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

# Đường dẫn để lưu tệp tải lên
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Upload file
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            return redirect(url_for('options', file_path=file_path))
    return render_template('upload.html')

# Hiển thị các lựa chọn phân tích
@app.route('/options')
def options():
    file_path = request.args.get('file_path')  # Không cần \uploads\baskets.csv
    if not file_path:
        return "File path is missing", 400  # Kiểm tra nếu file_path không có giá trị
    return render_template('options.html', file_path=file_path)

# Phân tích sản phẩm
@app.route('/products_analysis', methods=['GET'])
def products_analysis():
    file_path = request.args.get('file_path')
    if not file_path:
        return "File path is missing", 400
    # Đảm bảo đường dẫn là tuyệt đối và kiểm tra xem tệp có tồn tại không
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        return f"File at {file_path} does not exist.", 400  # Thông báo nếu tệp không tồn tại
    df = pd.read_csv(file_path)
    # Kiểm tra nếu cột cần thiết có tồn tại
    if 'Member_number' not in df or 'Date' not in df or 'itemDescription' not in df:
        return "Dữ liệu không hợp lệ. Vui lòng kiểm tra tệp tải lên."
    
    transactions = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list)
    
    all_items = set(item for transaction in transactions for item in transaction)
    item_df = pd.DataFrame([{item: item in transaction for item in all_items} for transaction in transactions])
    
    frequent_itemsets = apriori(item_df, min_support=0.001, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    rules_sorted = rules.sort_values(by="lift", ascending=False).head(20)
    
    pairs = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20)
    pairs['antecedents'] = pairs['antecedents'].apply(lambda x: ', '.join(list(x)))
    pairs['consequents'] = pairs['consequents'].apply(lambda x: ', '.join(list(x)))

    
    
    frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)
    rules.to_csv('association_rules.csv', index=False)
    
    return render_template('products_analysis.html', rules=rules_sorted, products_analysis=pairs.to_dict(orient='records'))


# Phân tích theo thời gian
@app.route('/time_analysis')
def time_analysis():
    file_path = request.args.get('file_path')
    df = pd.read_csv(file_path)
    # Thêm logic phân tích thời gian từ đoạn mã của bạn
    # Trả về dữ liệu và vẽ biểu đồ
    return render_template('time_analysis.html')

# Dự báo
@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        file_path = request.form['file_path']
        product_name = request.form['product_name']
        df = pd.read_csv(file_path)
        # logic dự báo
        # Chuyển đổi dữ liệu thành định dạng ngày tháng
        df['Date'] = pd.to_datetime(df['Date'])
        df_grouped = df.groupby(['Date', 'itemDescription']).size().reset_index(name='count')

        # Kiểm tra xem sản phẩm có tồn tại trong dữ liệu hay không
        if product_name in df_grouped['itemDescription'].unique():
            product_data = df_grouped[df_grouped['itemDescription'] == product_name]
        else:
            raise ValueError("Sản phẩm này không được tìm thấy trong dữ liệu.")

        # Đặt cột 'Date' là chỉ số
        product_data.set_index('Date', inplace=True)

        # Kiểm tra kích thước dữ liệu, điều chỉnh order nếu cần và huấn luyện mô hình
        if len(product_data) < 10: # ví dụ: nếu ít hơn 10 điểm dữ liệu
            model = ARIMA(product_data['count'], order=(1, 1, 0))  # Giảm order
        else:
            model = ARIMA(product_data['count'], order=(100, 1, 0))  # Chọn các tham số (p,d,q) phù hợp

        model_fit = model.fit()

        # Tạo biểu đồ chuỗi thời gian
        plot_path = os.path.join('static', 'forecast_plot.png')
        plt.figure(figsize=(12, 6))
        plt.plot(product_data['count'], label='Dữ liệu gốc')
        plt.title('Dữ liệu gốc')
        plt.legend()
        plt.savefig(plot_path)
        plt.close()

        # Dự báo trong 30 ngày
        forecast_values = model_fit.forecast(steps=30)
        forecast_dates = pd.date_range(product_data.index[-1] + pd.DateOffset(days=1), periods=30, freq='D')

        # Tạo biểu đồ dự báo
        plt.figure(figsize=(12, 6))
        plt.plot(product_data.index, product_data['count'], label='Dữ liệu thực tế')
        plt.plot(forecast_dates, forecast_values, label='Dự báo', color='red')
        plt.title('Dự Báo Nhu Cầu Sản Phẩm')
        plt.xlabel('Ngày')
        plt.ylabel('Số lượng mua')
        plt.legend()
        plt.savefig(plot_path)
        plt.close()

        # Trả về kết quả
        return render_template('forecast.html', forecast_results=forecast_values, plot_path=plot_path)

    return render_template('forecast.html')

# EOQ
@app.route('/eoq', methods=['GET', 'POST'])
def eoq():
    if request.method == 'POST':
        demand_rate = float(request.form['demand_rate'])
        ordering_cost = float(request.form['ordering_cost'])
        holding_cost = float(request.form['holding_cost'])
        EOQ = math.sqrt((2 * demand_rate * ordering_cost) / holding_cost)
        return render_template('eoq.html', eoq=EOQ)
    return render_template('eoq.html')

# Quyết định đặt hàng
@app.route('/reorder', methods=['GET', 'POST'])
def reorder():
    if request.method == 'POST':
        forecasted_demand = [float(x) for x in request.form['forecasted_demand'].split(',')]
        current_stock = float(request.form['current_stock'])
        safety_stock = float(request.form['safety_stock'])
        
        reorder_needed = current_stock + safety_stock < sum(forecasted_demand)
        return render_template('reorder.html', reorder_needed=reorder_needed)
    return render_template('reorder.html')

# Vẽ biểu đồ trực quan
@app.route('/visualize')
def visualize():
    # Thêm logic từ đoạn mã biểu đồ
    return render_template('visualize.html')

if __name__ == '__main__':
    app.run(debug=True)
