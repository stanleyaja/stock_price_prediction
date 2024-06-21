import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import tensorflow as tf
import datetime
import time
import plotly.graph_objects as go
from keras.models import load_model

def view_predict():
    # Set the start and end dates for the data testing
    start_train = "2019-01-01"
    end_train = start_date = "2024-01-01"
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Set the window size and horizon
    window_size = 5
    horizon = 1

    # Create a function to get the current data
    def currentdata(ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        data['Date'] = data['Date'].dt.strftime('%d-%m-%Y')
        
        def str_to_datetime(s):
            split = s.split('-')
            day, month, year = int(split[0]), int(split[1]), int(split[2])
            return datetime.datetime(day=day, month=month, year=year)
        data['Date'] = data['Date'].apply(str_to_datetime)
        
        first_row = data.iloc[:1]
        rest_of_data = data.iloc[1:]
        rest_of_data = rest_of_data[rest_of_data['Volume'] != 0]

        data = pd.concat([first_row, rest_of_data])
        data = data[['Date', 'Close']]
        data.index = data.pop('Date')
        return data

    # Create a function to create windows and horizons
    def create_windows_horizons(series, window_size, horizon_size):
        X, y = [], []
        for i in range(len(series) - window_size - horizon_size + 1):
            X.append(series[i:(i + window_size)])
            y.append(series[i + window_size:i + window_size + horizon_size])
        return np.array(X), np.array(y).reshape(-1, horizon_size)

    # Create a function to fetch the stock data
    def get_stock_data(ticker):
        data = yf.download(ticker, start=start_date, end=end_date)
        return data

    st.title("Make Predictions")

    ticker_options = {
        'BBCA': 'BBCA.JK',
        'BBRI': 'BBRI.JK',
        'BMRI': 'BMRI.JK',
    }
    # Get the user input for the stock ticker
    ticker = ticker_options[st.selectbox('Choose stock ticker', list(ticker_options.keys()), index=0, help="Select the stock ticker to predict the stock price.")]

    test_df = currentdata(ticker, start_date, end_date)
    train_df = currentdata(ticker, start_train, end_train)

    close_test = test_df['Close'].values
    X_test, y_test = create_windows_horizons(close_test, window_size, horizon)

    if ticker:
        # Fetch the stock data
        data = get_stock_data(ticker)

        # st.subheader("Stock Data")
        st.caption("Click to show/hide history data visualization.")

        # Toggle the show_data state
        def toggle_data_visibility():
            st.session_state.show_data = not st.session_state.show_data

        if 'show_data' not in st.session_state:
            st.session_state.show_data = False

        button_label = "Hide Graph" if st.session_state.show_data else "Show Graph"
        if st.button(button_label, on_click=toggle_data_visibility):
            pass
        
        if ticker == "BBCA.JK":
            stock = "PT Bank Central Asia Tbk (BBCA)"
        elif ticker == "BBRI.JK":
            stock = "PT Bank Rakyat Indonesia Tbk (BBRI)"
        elif ticker == "BMRI.JK":
            stock = "PT Bank Mandiri Tbk (BMRI)"

        if st.session_state.show_data:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_df.index, y=train_df['Close'], mode='lines', name='Actual Price', line=dict(color='blue', width=1.5, dash='solid')))
            fig.update_layout(title='Price Visualization (January 2019 - December 2023) for {stock}'.format(stock=stock),
                        xaxis_title='Date',
                        yaxis_title='Price')
            st.plotly_chart(fig)
            

        # total_days = (end - start).days
        total_days = len(data)-5
        selected_periods = st.slider("Periods (days)", 2, total_days, 20, help="Select the number of days to predict the stock price.")
        
        st.caption(f"Stock: :blue[{stock}]")
        st.caption(f"Prediction date: :blue[{(test_df.index[5]).strftime('%d-%m-%Y')}] until :blue[{(test_df.index[4 + selected_periods]).strftime('%d-%m-%Y')}]")
        
        if st.button("Predict"):
            # Display the prediction results
            # Load the model based on the ticker
            if ticker == "BBCA.JK":
                lstm_model = load_model("saved_models/BBCA_LSTM_window_5.h5")
                gru_model = load_model("saved_models/BBCA_GRU_window_5.h5")
            elif ticker == "BBRI.JK":
                lstm_model = load_model("saved_models/BBRI_LSTM_window_5.h5")
                gru_model = load_model("saved_models/BBRI_GRU_window_5.h5")
            elif ticker == "BMRI.JK":
                lstm_model = load_model("saved_models/BMRI_LSTM_window_5.h5")
                gru_model = load_model("saved_models/BMRI_GRU_window_5.h5")

            # Make predictions using the loaded model
            lstm_pred = lstm_model.predict(X_test[:selected_periods])
            gru_pred = gru_model.predict(X_test[:selected_periods])
            
            test_df = test_df.iloc[5:selected_periods+5]

            lstm_pred_df = pd.DataFrame(lstm_pred, columns=['lstm_pred'])
            gru_pred_df = pd.DataFrame(gru_pred, columns=['gru_pred'])

            test_df['lstm_pred'] = lstm_pred_df.values
            test_df['gru_pred'] = gru_pred_df.values
            test_df = test_df.dropna(subset=['lstm_pred'])

            with st.spinner("Predicting..."):
                time.sleep(3)
            alert = st.success("Prediction completed!")

            # Create a Plotly figure
            fig = go.Figure()
            if selected_periods > 60:
                width = 1
            elif selected_periods > 30:
                width = 1.5
            else:
                width = 2
            
            fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Close'], mode='lines', name='Actual Price', line=dict(color='blue', width=width, dash='solid')))
            fig.add_trace(go.Scatter(x=test_df.index, y=test_df['lstm_pred'], mode='lines', name='LSTM Prediction', line=dict(color='green', width=width, dash='dashdot')))
            fig.add_trace(go.Scatter(x=test_df.index, y=test_df['gru_pred'], mode='lines', name='GRU Prediction', line=dict(color='red', width=width, dash='dot')))
            
            # Customize the layout
            fig.update_layout(title='Price Prediction for {stock}'.format(stock=stock),
                            xaxis_title='Date',
                            yaxis_title='Price',
                            legend_title='Legend')

            # Add a hover mode to the figure
            fig.update_layout(hovermode='x')

            # Add a hover label to display the values of Close, lstm_pred, and gru_pred
            fig.update_traces(hovertemplate='%{y}',
                            customdata=np.column_stack((test_df['lstm_pred'], test_df['gru_pred'])))

            # Display the figure in Streamlit
            st.plotly_chart(fig)

            # Calculate the MAPE for LSTM and GRU models
            lstm_mape = np.mean(np.abs((test_df['Close'] - test_df['lstm_pred']) / test_df['Close'])) * 100
            gru_mape = np.mean(np.abs((test_df['Close'] - test_df['gru_pred']) / test_df['Close'])) * 100

            # st.write("LSTM model error percentage:", round(lstm_mape, 4), "%")
            # st.write("GRU model error percentage:", round(gru_mape, 4), "%")
            st.markdown(f"LSTM model error percentage: <span style='color: green; background-color: none;'>{round(lstm_mape, 4)}%</span>", unsafe_allow_html=True)
            st.markdown(f"GRU model error percentage: <span style='color: red; background-color: none;'>{round(gru_mape, 4)}%</span>", unsafe_allow_html=True)
            
            if lstm_mape < gru_mape:
                st.write("**LSTM** model makes more accurate prediction compared to **GRU** model based on the model's error percentage.")
            else:
                st.write("**GRU** model makes more accurate prediction compared to **LSTM** model based on the model's error percentage.")
            st.caption("Note: These error percentages are the deviation of the predicted price from the actual price.")

            time.sleep(3)
            alert.empty()