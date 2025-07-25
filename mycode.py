import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, recall_score
from sklearn.preprocessing import StandardScaler

def main():
    options_data = pd.read_pickle("spx_eom_expiry_options_2010_2022.bz2")
    options_data.index.name = 'index'
    underlying_data = pd.read_csv("sp500_index_2010_2022.csv", index_col='Date', parse_dates=['Date'])[['Open', 'High', 'Low', 'Close']]
    underlying_data = underlying_data.loc[underlying_data.index.isin(options_data.index)].dropna()
    underlying_data['atm_strike_price'] = np.nan

    for date in underlying_data.index:
        td = options_data.loc[date]
        idx = td[' [STRIKE_DISTANCE_PCT]'].idxmin()
        underlying_data.loc[date, 'atm_strike_price'] = td.loc[idx, ' [STRIKE]']

    options_data[' [QUOTE_DATE]'] = pd.to_datetime(options_data.index)
    df = pd.merge(
        underlying_data,
        options_data,
        left_on=['Date', 'atm_strike_price'],
        right_on=[' [QUOTE_DATE]', ' [STRIKE]']
    )
    underlying_data = df.copy()
    underlying_data.columns = (
        underlying_data.columns
        .str.replace(r'[\[\]]', '', regex=True)
        .str.strip()
        .str.lower()
        .str.replace('c_', 'call_')
        .str.replace('p_', 'put_')
    )
    underlying_data = underlying_data[(underlying_data.call_last != 0) & (underlying_data.put_last != 0)]

    intervals = [1, 5, 10, 22, 44, 88]
    for t in intervals:
        underlying_data[f'f_ret_{t}'] = underlying_data.close.pct_change(t)

    underlying_data['f_rsi']  = talib.RSI(underlying_data.close)
    underlying_data['f_natr'] = talib.NATR(underlying_data.high, underlying_data.low, underlying_data.close)

    upper, middle, lower = talib.BBANDS(underlying_data.close)
    underlying_data['f_norm_upper']  = upper   / underlying_data.close
    underlying_data['f_norm_middle'] = middle  / underlying_data.close
    underlying_data['f_norm_lower']  = lower   / underlying_data.close

    underlying_data['cost_straddle']     = underlying_data.call_last + underlying_data.put_last
    underlying_data['straddle_returns'] = np.where(
        underlying_data.cost_straddle > underlying_data.cost_straddle.shift(-3),
        1, 0
    )

    model_cols = [
        'call_last', 'dte',
        'call_delta','call_gamma','call_vega','call_theta','call_rho','call_iv',
        'put_delta','put_gamma','put_vega','put_theta','put_rho','put_iv',
    ] + [f'f_ret_{t}' for t in intervals] + [
        'f_natr','f_rsi','f_norm_upper','f_norm_middle','f_norm_lower',
        'straddle_returns'
    ]

    md = underlying_data[model_cols].copy()
    md['call_iv'] = pd.to_numeric(md.call_iv, errors='coerce')
    md['put_iv']  = pd.to_numeric(md.put_iv, errors='coerce')
    md.dropna(inplace=True)

    X = md.drop('straddle_returns', axis=1)
    y = md['straddle_returns'].shift(-1)
    X, y = X.iloc[:-1], y.iloc[:-1].astype(int)

    test_size = int(0.2 * len(X))
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),  index=X_test.index,  columns=X_test.columns)

    w0 = int(100 * (len(y_train) - y_train.sum()) / y_train.sum())
    w1 = 100 - w0
    model = LogisticRegression(class_weight={0: w0, 1: w1})
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"AUC:      {roc_auc_score(y_test, y_pred):.2f}")
    print(f"Recall:   {recall_score(y_test, y_pred):.2f}")

    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_s)[:,1])
    plt.figure(figsize=(12,6))
    plt.plot(fpr, tpr, label=f"LR (AUC={roc_auc_score(y_test, y_pred):.2f})")
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
