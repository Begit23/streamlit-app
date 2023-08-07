import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import joblib
import os
import streamlit as st
from datetime import date,timedelta,datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from scipy.special import inv_boxcox1p
from plotly import graph_objs as go
from requests_html import HTMLSession
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

#the evaluations of traind model
csv_file_path = os.path.join('Data', 'reg_evaluation.csv')
reg_evaluations=pd.read_csv(csv_file_path)

def scrape_and_predict(ticker,scaler,reg_model):
    end_date=datetime.strftime(date.today()+timedelta(days=1), '%Y-%m-%d')
    data =  yf.Ticker(ticker).history(start='2016-01-01', end=end_date)

    def Feature_Generation(df):
        # adding the SMA features
        df['SMA_3']=df['Close'].rolling(window=3,min_periods=1).mean()
        df['SMA_7']=df['Close'].rolling(window=7,min_periods=1).mean()
        df['SMA_14']=df['Close'].rolling(window=14,min_periods=1).mean()
        df['SMA_50']=df['Close'].rolling(window=50,min_periods=1).mean()
        df['SMA_150']=df['Close'].rolling(window=150,min_periods=1).mean()
        # adding the EMA features
        df['EMA_3']=df['Close'].ewm(span=3, adjust=False,min_periods=1).mean()
        df['EMA_7']=df['Close'].ewm(span=7, adjust=False,min_periods=1).mean()
        df['EMA_14']=df['Close'].ewm(span=14, adjust=False,min_periods=1).mean()
        df['EMA_50']=df['Close'].ewm(span=50, adjust=False,min_periods=1).mean()
        df['EMA_150']=df['Close'].ewm(span=150, adjust=False,min_periods=1).mean()
        #Bollinger Bands
        TP=(df['High']+df['Low'])/2
        df['BOLM']=TP.rolling(window=20).mean()
        std_tp=TP.rolling(window=20).std()
        df['BOLU']=df['BOLM']+2*std_tp
        df['BOLD']=df['BOLM']-2*std_tp
    
        #CCI
        ma =TP.rolling(window=20).mean()
        df['CCI'] = (TP - ma) / (0.015 * std_tp) 
    
        #William’s %R
        high_h= df['High'].rolling(window=14).max() 
        low_l= df['Low'].rolling(window=14).min()
        df['wr'] = -100 * (( high_h - df['Low']) / (high_h - low_l))
    
        #Stochastic %K
    
        # Uses the min/max values to calculate the %k (as a percentage)
        df['%K'] = (df['Low'] - low_l) * 100 / (high_h -low_l)
        # Uses the %k to calculates a SMA over the past 3 values of %k
        df['%D'] = df['%K'].rolling(window=14).mean()
    
        #MFI
        money_flow = TP * df['Volume']
        mf_sign = np.where(TP>TP.shift(1), 1, -1)
        signed_mf =money_flow * mf_sign
        mf_avg_pos = signed_mf.rolling(window=14).apply(lambda x: ((x>0)*x).sum(), raw=True)
        mf_avg_neg= signed_mf.rolling(window=14).apply(lambda x: ((x<0)*x).sum(), raw=True)
        df['MFI']=100 - (100 / (1 + (mf_avg_pos / abs(mf_avg_neg))))
    
        #ATR
        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Close'].shift())
        low_cp = np.abs(df['Low'] - df['Close'].shift())
    
        true_range = np.max(pd.concat([high_low, high_cp, low_cp], axis=1), axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        df=df.fillna(0, inplace=True)
    
        return df
    Feature_Generation(data)

    #drop null columns
    data.drop(['Stock Splits','Dividends'], axis=1, inplace=True)
    #insert date column
    data.insert(loc=0, column='Date', value=data.index.get_level_values(0))
    #sort the rows by date
    data.sort_index(ascending=True, axis=0, inplace=True)
    data.reset_index(inplace=True, drop=True)


    negative_features = []
    for column in data.columns.drop('Date'):
        # Check if any value in the column is negative and skewed
        if (data[column] < 0).any() and skew(data[column]>60):
           negative_features.append(column)
    for feature in negative_features:
        feature_values = data[feature]
        # Add a constant to make all values positive
        feature_positive = feature_values - np.min(feature_values) +1 
        # Apply boxcox transformation
        transformed_feature = boxcox1p(feature_positive, 0.15) 
        # Save the transformed feature back into the original DataFrame
        data[feature] = transformed_feature
    #positive features:
    negative_features.append('Date')
    numeric_feats =data.columns.drop(negative_features)
    count=0
    for feat in numeric_feats:
        if skew(data[feat]>60):
           data[feat] = boxcox1p(data[feat], 0.15) 
           count=count+1

    #predictions        
    df_pred=data.tail(1)
    new_observation_scaled = scaler.transform(df_pred.drop(columns=['Date','Close'], axis=1))
    new_observation_scaled_df = pd.DataFrame(new_observation_scaled, columns=df_pred.drop(columns=['Date','Close'], axis=1).columns)
    prediction= reg_model.predict(new_observation_scaled_df)
    pred = inv_boxcox1p(prediction[0][0], 0.15)
    
    return(pred)


def CandlePlot(start_date,select,tickerName):
    end_date=datetime.strftime(date.today()+timedelta(days=1), '%Y-%m-%d')
    df1 =  yf.Ticker(tickerName).history(start='2016-01-01', end=end_date)
    if start_date!='All':
       df1 =  yf.Ticker(tickerName).history(start=start_date, end=end_date)
       if df1.empty ==True:
           df1=yf.Ticker(tickerName).history(period='1d')
    if select!='select':
       df1['SMA-7 days']=df1['Close'].rolling(window=7,min_periods=1).mean()
       df1['SMA-14 days']=df1['Close'].rolling(window=14,min_periods=1).mean()
       df1['SMA-50 days']=df1['Close'].rolling(window=50,min_periods=1).mean()
       df1['EMA-7 days']=df1['Close'].ewm(span=7, adjust=False,min_periods=1).mean()
       df1['EMA-14 days']=df1['Close'].ewm(span=14, adjust=False,min_periods=1).mean()
       df1['EMA-50 days']=df1['Close'].ewm(span=50, adjust=False,min_periods=1).mean()
       # Uses the min/max values to calculate the %k (as a percentage)
       high_h= df1['High'].shift().rolling(window=14).max() 
       low_l= df1['Low'].shift().rolling(window=14).min()
       df1['%K'] = (df1['Low'] - low_l) * 100 / (high_h -low_l)
       # Uses the %k to calculates a SMA over the past 3 values of %k
       df1['Stochastic %K'] = df1['%K'].rolling(window=14).mean()
       
    df1.insert(loc=0, column='Date', value=df1.index.get_level_values(0))
    fig = go.Figure(data=go.Ohlc(x=df1['Date'],
                    open=df1['Open'],
                    high=df1['High'],
                    low=df1['Low'],
                    close=df1['Close']))
    if select!='select':
        fig.add_trace(go.Scatter(x=df1['Date'], y=df1[select], mode='lines', name=select,line=dict(color='Blue')))
    st.plotly_chart(fig) 


#streamlit
st.set_page_config(layout='wide')

def main():
  col1,col2=st.columns((2,1))
  #sidebar
  st.sidebar.header("graph setting")
    
  ticker_list={'S&P500 (^GSPC)':'^GSPC','Nike (NKE)':'NKE','Tesla (TSLA)':'TSLA','Microsoft (MSFT)':'MSFT',
                 'Amazon (AMZN)':'AMZN','Teva (TEVA)':'TEVA','Apple (AAPL)':'AAPL'}
  stock_selectBox= st.sidebar.selectbox(
    "Select stock:",
    ('S&P500 (^GSPC)','Nike (NKE)','Tesla (TSLA)','Microsoft (MSFT)',
                 'Amazon (AMZN)','Teva (TEVA)','Apple (AAPL)'),
    index=0)
  featuer_selectBox= st.sidebar.selectbox(
    "Add featuer:",
    ("select","SMA-7 days", "SMA-14 days", "SMA-50 days","EMA-7 days","EMA-14 days","EMA-50 days","Stochastic %K"),
    index=0)  
    
  if "visibility" not in st.session_state:
        st.session_state.visibility = "All"
        st.session_state.disabled = False
  st.sidebar.write("Show close price for:")
  date_range = st.sidebar.radio(" ",
                 key="visibility",
                 options=["All", "1D", "5D","1M","3M","6M","1Y","5Y"],
                 )
  st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
  def graph_setting(date_range,featuer_selectBox,stock_selectBox):
        if date_range =='All':
           CandlePlot('All',featuer_selectBox,ticker_list[stock_selectBox])
        
        if date_range =="1D":
           CandlePlot(datetime.strftime(date.today()-timedelta(1), '%Y-%m-%d'),featuer_selectBox,ticker_list[stock_selectBox])
        if date_range =="5D":
           CandlePlot(datetime.strftime(date.today()-timedelta(5), '%Y-%m-%d'),featuer_selectBox,ticker_list[stock_selectBox])
        if date_range =="1M":
           CandlePlot(datetime.strftime(date.today()-relativedelta(months=1), '%Y-%m-%d'),featuer_selectBox,ticker_list[stock_selectBox])
        if date_range =="3M":
           CandlePlot(datetime.strftime(date.today()-relativedelta(months=3), '%Y-%m-%d'),featuer_selectBox,ticker_list[stock_selectBox])
        if date_range =="6M":
           CandlePlot(datetime.strftime(date.today()-relativedelta(months=6), '%Y-%m-%d'),featuer_selectBox,ticker_list[stock_selectBox])
        if date_range =="1Y":
           CandlePlot(datetime.strftime(date.today()-relativedelta(years=1), '%Y-%m-%d'),featuer_selectBox,ticker_list[stock_selectBox])
        if date_range =="5Y":
           CandlePlot(datetime.strftime(date.today()-relativedelta(years=5), '%Y-%m-%d'),featuer_selectBox,ticker_list[stock_selectBox])
   ##end of sidebar setting
   
  with col1:
        #st.title(f'{stock_selectBox} data and prediction')
        st.header(f':withe[{stock_selectBox} data and prediction]')
        #scraping the last change data from cnbc website
        st.markdown(
                """
                <style>
                [data-testid="stMetricValue"] {
                font-size: 50px;
                }
                </style>
                """,
                unsafe_allow_html=True,
                )
        s=HTMLSession()
        cnbc_list={'S&P500 (^GSPC)':'.SPX','Nike (NKE)':'NKE','Tesla (TSLA)':'TSLA','Microsoft (MSFT)':'MSFT',
                 'Amazon (AMZN)':'AMZN','Teva (TEVA)':'TEVA','Apple (AAPL)':'AAPL'}
        r=s.get(f'https://www.cnbc.com/quotes/{cnbc_list[stock_selectBox]}') 
        item=r.html.find('div.QuoteStrip-dataContainer',first=True)
        #scrap the last change:
        try:
          Delta = item.find('span.QuoteStrip-changeUp', first=True).text
        except AttributeError:
        # If 'span.QuoteStrip-changeUp' is not found, try to find 'span.QuoteStrip-changeDown'
           try:
              Delta = item.find('span.QuoteStrip-changeDown', first=True).text
           except AttributeError:
        # If neither 'span.QuoteStrip-changeUp' nor 'span.QuoteStrip-changeDown' is found
               Delta = None  # or you can handle the default value you want
        
        st.metric(label=item.find('div.QuoteStrip-dataContainer',first=True).text.split('\n')[0], value=item.find('span.QuoteStrip-lastPrice',first=True).text, delta=Delta)
        graph_setting(date_range,featuer_selectBox,stock_selectBox)
        
        #The prediction button
        if st.button("Predict the next stock price"):
           #read the traind model and the scaler from the 'Data' folder
           scaler_file_path = os.path.join('Data', f'scaler_{ticker_list[stock_selectBox]}.pkl')
           model_file_path = os.path.join('Data', f'reg_{ticker_list[stock_selectBox]}.pkl')
           scaler = joblib.load(scaler_file_path)
           reg_model = joblib.load(model_file_path)
           #predict
           result = scrape_and_predict(ticker_list[stock_selectBox],scaler,reg_model)
           st.write('todays prediction',result)
           st.write('todays prediction',reg_evaluations[reg_evaluations['Ticker']==ticker_list[stock_selectBox]])
  with col2:
        #add space
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        #summary section-scrap the summary container
        items=r.html.find('section,Summary-container',first=True).find('li,Summary-stat')
        dict={}
        for item in (items):
            dict[item.find('span,Summary-label')[0].text]=item.find('span,Summary-label')[1].text
        df = pd.DataFrame({'Keys':dict.keys(),'Values':dict.values()})
        st.markdown("<h4 style='text-align: center; color: 'green';'>Summaray Data </h4>", unsafe_allow_html=True)
        #st.write("Summaray Data",)
        st.dataframe(df,width=700, height=250)
        
if __name__ == '__main__':
    main()


