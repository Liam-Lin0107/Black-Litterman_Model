# 資料來源
## 股票價格
price_filename = "./Data/Price_Data.xlsx"
pritce_sheet_name = "weekly"

## 股票市值
mv_filename = "./Data/Market_Value.xlsx"
mv_sheet_name = "weekly"

# 模型參數
tau = 0.3 # 事後收益率共變異數矩陣的縮放尺度，在0~1之間

#  模型回測
## 回測參數
back_test_T = 200 # 回測期間：200期
start_index = 273 # 開始時間：2015/1/2
end_index = 324 # 結束時間：2015/12/25
index_number = 0 # 股票指數索引：0. S&P500 1.Dow jones 2. 那斯達克

## 繪圖參數
back_test_x_label = 'week'
back_test_y_label = 'Accumulated Return(log)'
back_test_period_name = '2015'

## 觀點參數
view_type = 2 # 對下面觀點list進行索引
view_type_name = ['Market value as view', 'Arbitrary views', 'Reasonable views', 'Near period return as view'] # 觀點list
view_T = 10 # 若為Near period return as view 的話需要定義近期時間參數，取view_T期的歷史平均作為預期收益率
