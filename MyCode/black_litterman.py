import numpy as np
import pandas as pd
import scipy.optimize as optim
import imageio
import matplotlib
import matplotlib.pyplot as plt
import time
from structure import *

"""
Function 包含以下功能
1. print_data()：打印重要參數
2. read_data(filename, sheetname)：讀取數據
3. get_return()：處理數據，得到股票列表名稱與報酬率
4. get_market_value_weight()：計算市值權重矩陣
5. get_implied_excess_equilibrium_retun()：計算風險厭惡係數lambda、事前預期報酬率(mu_0)
6. get_views_P_Q_matrix()：設定觀點矩陣(P)、先對報酬率強度矩陣(Q) (共三種views可以選)
7. get_views_omega()：計算Omega矩陣
8. get_posteriod_combine_return()：計算事後報酬率mu_p
9. get_weight_bl()：定義ＢＬ模型的計算公式
10.get_psot_weight_weight()：計算不同時間下、不同views下得到的ＢＬ模型權重、真實報酬率
11.calculate_comparative_return()：計算equal-weighted累積報酬率
"""

class BlackLitterman:
    # 初始化參數
    def __init__(self):
        # 數據參數
        self.price_filename = price_filename
        self.pritce_sheet_name = pritce_sheet_name
        self.mv_filename = mv_filename
        self.mv_sheet_name = mv_sheet_name

        # 模型參數
        self.tau = tau

        # 回測參數
        self.back_test_T = back_test_T

        # 觀點參數
        self.view_type = view_type
        self.view_T = view_T

        # 投資品種
        ## 股票參數
        self.stock_return = 0
        self.stock_names = 0
        self.stock_number = 0
        self.market_value_weight = 0
        ## 股指參數
        self.index_num = index_number
        self.index_name = 0
        self.index_return = 0

    # 打印重要參數
    def print_data(self):
        print(slef.index_return)
        print(self.stock_return)
        print(self.index_name)
        print(self.stock_names)

    # 讀取數據
    def read_data(self, filename, sheet_name):
        df = pd.read_excel(filename, sheet_name)
        df.set_index("Date", inpalce = True)
        df.index = range(len(df))
        df = df.astype("float64")
        return df

    #  數據處理：股票名稱、股票報酬率
    def get_return(self):
        filename = self.price_filename
        sheet_name = self.sheet_name

        index_num = self.index_num
        df = self.read_data(filename, sheet_name)
        # 計算報酬
        log_return = np.log(df/df.shift(1))
        log_return = log_return.drop(index = [0])

        # 取得股票名稱or指數名稱
        names = log_return.columns.tolist()

        index_name = names[index_num]

        stock_names = names[3:]
        # 指數報酬
        index_reuturn = log_return[index_name]
        # 股票報酬
        stock_reutrn = log_return[stock_names]

        # 更新數據
        self.index_return = index_reuturn
        self.stock_return = stock_reutrn
        self.index_name = index_name
        self.stock_names = stock_names
        self.stock_number - len(stock_names)

    # 市值權重矩陣
    def get_market_value_weight(self):
        filename = self.mv_filename
        sheet_name = self.mv_sheet_name
        mv = self.read_data(filename, sheet_name)

        # 刪去最後一列的Total
        stock_names = mv.columns.tolist()[0:-1]

        # 計算 market value weighted
        for n in stock_names:
            mv[n] = mv[n] / mv["Total"]

        # 去掉第一列
        mv = mv.drop(index = [0])

        mv = mv[stock_names]

        self.market_value_weight = np.array(mv)


    # 計算風險厭惡係數lambda、先驗預期報酬率
    def get_implied_excess_equilibrium_retun(self, stock_return, w_mkt):
        """
        param stock_return : 使定股票報酬數據
        param w_mkt : 當前市場權重
        return : 風險厭惡係數、先驗報酬率
        """
        # weelly risk-free return
        rf = 0.0006132

        # 根據股票報酬率得到市場共變異數矩陣
        mkt_cov = np.array(stock_return.cov())

        # lambd
        lambd = ((np.dot(w_mkt, stock_return().mean())) - rf) / np.dot(np.dot(w_mkt, mkt_cov), w_mkt.T)

        # 計算先驗報酬率
        implied_return = lambd * np.dot(mkt_cov, w_mkt)
        return implied_return, lambd

    # views matrix P, 報酬率向量 Q (有三種views可以選)
    def get_views_P_Q_matrix(self, view_type, stock_return):
        N = self.stock_number
        if (view_type == 0 or view_type == 1):
            # view_type = 0 : 投資者無觀點，依市值權重
            # view_type = 1 : 投資者有觀點 （三種）
            """
            觀點1. (BRK_B)比(XOM)的期望報酬高0.01%
            觀點2. (MSFT)比(JPM)的期望報酬高0.025%
            觀點3. 10%(JPM)+90%(V)的投资组合比10%(WMT)+90%(BAC)的投资组合期望報酬高0.01%
            """
            P = np.zeros([3, N])
            P[0, 8] = 1
            P[0, 9] = -1
            P[1, 1] = 1
            P[1, 3] = -1
            P[2, 3] = 0.1
            P[2, 4] = 0.9
            P[2, 6] = -0.1
            P[2, 7] = -0.9
            
            Q = np.array([0.0001, 0.00025, 0.0001])
        elif(view_type == 2):
            # view_type = 2: Reasonable views
            P = np.zeros([1, N])
            P[0, 2] = 1
            P[0, 3] = -1
            Q = [0.017]
        elif(view_type == 3):
            # view_type = 3: 最近VIEW_T期的历史平均收益率作为预期收益率
            # T_near: 使用近期T_near期数据的历史平均收益率作为预期收益率
            T_near = self.view_T
            P = np.identity(N)
            stock_cc_ret_near = stock_cc_ret.iloc[-T_near:]
            Q = np.array(stock_cc_ret_near.mean())
        else:
            print("There is no such kind of view type!")
        return P, Q

    # Step8. 计算Omega矩阵
    def get_views_omega(self, mkt_cov, P):
        tau = self.tau
        # K: 投资者观点的数量
        K = len(P)
        # 生成K维度的对角矩阵（对角线上全为1）
        omega = np.identity(K)
        for i in range(K):
            # 逐行选取P（Views矩阵，维度：K*N，此处N=10）
            P_i = P[i]
            omg_i = np.dot(np.dot(P_i, mkt_cov), P_i.T) * tau
            # 将得到的结果赋值到矩阵对角线元素
            omega[i][i] = omg_i
        return omega

    # Step9. 计算后验期望收益率mu_p
    def get_posterior_combined_return(self, implied_ret, mkt_cov, P, Q, omega):
        # tau为缩放尺度
        tau = self.tau
        # 后验期望收益率mu_p的计算公式
        k = np.linalg.inv(np.linalg.inv(tau * mkt_cov) + np.dot(np.dot(P.T, np.linalg.inv(omega)), P))
        posterior_ret = np.dot(k, np.dot(np.linalg.inv(tau * mkt_cov), implied_ret) +
                            np.dot(np.dot(P.T, np.linalg.inv(omega)), Q))
        return posterior_ret

    # Step10. 计算由BL模型得到的新权重weight_bl
    def get_weight_bl(self, posterior_ret, mkt_cov, lambd):
        weight_bl = np.dot(np.linalg.inv(lambd * mkt_cov), posterior_ret)
        return weight_bl

    # Step11. 计算指定时间窗口T和不同Views下得到的BL模型新权重、真实收益率
    def get_post_weight(self, start_idx):
        T = self.back_test_T                                                # T区间：200（个数据）
        view_type = self.view_type                                          # 三种观点类型：例如“0”，意味着['Market value as view']
        index_cc_ret, stock_cc_ret = self.index_cc_ret, self.stock_cc_ret   # 传入原始标普500和10只股票收益率
        real_ret = np.array(stock_cc_ret.iloc[start_idx])                   # 真实收益率：按行索引提取数据
        stock_cc_ret = stock_cc_ret.iloc[start_idx - T: start_idx]          # 提取指定T回测区间10只股票收益率数据(2014年底以前的200天）
        index_cc_ret = index_cc_ret.iloc[start_idx - T: start_idx]          # 提取指定T回测区间标普500收益率数据(2014年底以前的200天）
        mkt_cov = np.array(stock_cc_ret.cov())                              # 将T区间部分的股票收益率计算成协方差矩阵

        # Get market value weight of these stock at current time（取得这些股票当前的市场权重mv_i，日期：2014年的最后一行）
        mv_i = self.market_value_weight[start_idx - 1]

        # 得到T区间内的风险厌恶系数lambda、先验预期收益率implied_ret（即mu_0)
        implied_ret, lambd = self.get_implied_excess_equilibrium_return(stock_cc_ret, mv_i)
        P, Q = self.get_views_P_Q_matrix(view_type, stock_cc_ret)           # 根据选定的View类型，设置P和Q矩阵
        omega = self.get_views_omega(mkt_cov, P)                            # 根据选定的View类型，计算Omega矩阵

        posterior_ret = self.get_posterior_combined_return(implied_ret, mkt_cov, P, Q, omega)
        if (view_type == 0):
            # weight_type == 0: 无观点，使用当前市值权重作为BL模型的权重（即无需代入BL公式计算）
            weight_bl = np.array(mv_i)
        elif (view_type == 1 or view_type == 2 or view_type == 3):
            # weight_type == 1: 根据Views的类型，计算BL模型得到的新权重weight_bl
            weight_bl = self.get_weight_bl(posterior_ret, mkt_cov, lambd)

        return weight_bl, real_ret

    # Step12. 计算等权重累计收益率：eq_acc（作为对照组）
    def calculate_comparative_return(self, start_idx, end_index):
        stock_names = self.stock_names                                          # 传入10只股票名称列表
        stock_cc_ret = self.stock_cc_ret                                        # 传入10只股票收益率
        stock_cc_ret = stock_cc_ret.iloc[start_idx: end_index + 1]              # 选定2015年10只股票收益率数据
        stock_cc_ret["mean"] = stock_cc_ret.loc[:, stock_names].mean(axis=1)    # 新增一列mean：2015年10只股票每日平均收益率
        eq_acc = [0]
        eq_ret = np.array(stock_cc_ret["mean"])
        for r in eq_ret:
            eq_acc.append(eq_acc[-1] + r)                                       # 累加每日收益率，形成列表
        return eq_acc

