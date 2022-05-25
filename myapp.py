#import required libraries
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import time
import tensorflow as tf
import keras
from keras.models import Sequential,model_from_json
from keras.layers import LSTM
from keras.layers import Dense,Dropout,Activation
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# choose your df from dropdown
flag=False
st.markdown('##### A Study on the Future Prediction of Cryptocurrency prices using Deep Learning')
st.image("Images//title_card.JPG")

st.markdown('###### Crypto Currencies Choosen')
st.image("Images//crypto_figure.JPG")
df_name = st.sidebar.selectbox(
    'Select df',
    ('Bitcoin', 'Ethereum', 'Dogecoin'))
data_dict = {'Bitcoin':'BTC-INR','Ethereum':'ETH-INR','Dogecoin':'DOGE-INR'}
st.write(f"## {df_name} dataset")

@st.cache

def fetch_dataset(f,name,data_dict):
    # select dataframe
    if name in data_dict.keys():
        f=True
        df = yf.download(data_dict[name],period='6mo',interval='1h')
        df.drop(columns=['Adj Close','Volume'],inplace=True)
        df['Avg_price'] = (df['Open']+df['Close'])/2
        df = df.iloc[:-1,:]
        return df,f
    else:
        f=False
        df = 0
        return df,f
df,flag = fetch_dataset(flag,df_name,data_dict)

global ind 
ind = df.index

if flag==True:
    st.success('Data downloaded successfully')
    st.dataframe(df,width=900,height=300)
    description = df.describe()
    st.markdown('##### Descriptive Statistics ')
    st.table(description)
else:
    st.error('Data not Loaded')
    

# select target column

if flag==True:
    target_column = st.sidebar.selectbox(
        'Select Target column',
        ('Close', 'Avg_price'))
    st.write(f"###### Selected target column : {target_column}")


    # plot close price
    if target_column=='Close':
        plt.rcParams['figure.figsize']=[18,8]
        fig = plt.figure()
        plt.plot(df['Close'],color='black')
        plt.grid()
        plt.title(data_dict[df_name],fontsize=15,color='b')
        plt.xlabel('Last 6 Months data',fontsize=15,color='b')
        plt.ylabel('Closing price of '+data_dict[df_name]+' on hourly  basis',fontsize=15,color='r')
        plt.show()
        st.pyplot(fig)
        #dynamic = st.button("Click Here For Dynamic Chart")
        #if dynamic:
           # st.line_chart(df['Avg_price'])
    #st.balloons()
    else:
    #plot for close price
        fig = plt.figure()
        plt.plot(df['Avg_price'],color='black')
        plt.grid()
        plt.title(data_dict[df_name],fontsize=15,color='b')
        plt.xlabel('Last 6 Months data',fontsize=15,color='b')
        plt.ylabel('Average price of '+data_dict[df_name]+' on hourly basis',fontsize=15,color='r')
        plt.show()
        st.pyplot(fig)
        #dynamic = st.button("Click Here For Dynamic Chart")
       
        #if dynamic:
           # st.line_chart(df['Avg_price'])
else:
    st.error('Data is not laoded')

# create button return True on Clicking the button

f = st.sidebar.slider('forecast_Records on Hourly basis', 100, 168)

create_train_test = st.button("Load your model")

@st.cache

def create_dataset(dataset,step):
        data_X,data_y =[],[]
        for i in range(len(dataset)-step-1):
                get_val = dataset[i:(i+step),0]
                data_X.append(get_val)
                data_y.append(dataset[i+step,0])
        return np.array(data_X),np.array(data_y)
    

def performance_evaluation(train_actual,train_predicted,test_actual,test_predicted): 
    
    st.write('RMSE for test data =',round(np.sqrt(mean_squared_error(test_actual,test_predicted)),4))
    st.write('RMSE for training data = ',round(np.sqrt(mean_squared_error(train_actual,train_predicted)),4))
    st.write('coefficient of determinant testing data= ',round(r2_score(test_actual,test_predicted),4))
    st.write('coefficient of determinant for training data= ',round(r2_score(train_actual,train_predicted),4))
    
def mape(actual,predicted,name):
    st.write('MAPE for '+name+' =',round(np.mean( np.abs( (actual - predicted)*100/actual )),4))
    
    

def pred_vs_actual(actual,predicted,tit):
    fig = plt.figure()
    plt.plot(actual,color='black',label='Actual')
    plt.plot(predicted,color='red',label='Predicted')
    s = 'Actual  vs predicted for' +tit
    plt.title(s)
    plt.legend()
    plt.show()
    st.pyplot(fig)

def forecast_future(inputs,no_hours,ts,LSTM):  
    forecast = []
    i = 0
    while i<no_hours:    
        if len(inputs)>ts:
            X_inp = np.array(inputs[1:])
            #print('Input',i+1," : ",X_inp)
            X_inp = X_inp.reshape(-1,1)
            X_inp = X_inp.reshape(1,ts,1)
            y_hat = LSTM.predict(X_inp)
            #print('Forecasted value :',y_hat)
            inputs.extend(y_hat[0])
            inputs = inputs[1:]
            forecast.extend(y_hat.tolist())
        else:
            X_inp = X_input.reshape((1,X_input.shape[0],X_input.shape[1]))
            y_hat = LSTM.predict(X_inp,verbose=1)
            #print('Input',i+1," : ",X_inp.reshape(1,30))
            #print('Forecasted value :',y_hat)
            inputs.extend(y_hat[0])
            forecast.extend(y_hat.tolist())
        i= i+1 
    return forecast    

def plot_last_130(target_column,length,forecasted,future):
    day_past = np.arange(1,101)
    day_pred = np.arange(101,101+future)
    past_data = np.array(target_column)[length-100:,]
    fig = plt.figure()
    plt.plot(day_past,past_data,color='black',label='Past')
    plt.plot(day_pred,forecasted,color='red',label='Future')
    s = 'past 100 vs future' + str(future)
    plt.title(s)
    plt.legend()
    st.pyplot(fig)
    collection = list(past_data)
    collection.extend(forecasted)
    fig1 = plt.figure()
    plt.plot(collection,color='green')
    st.pyplot(fig1)

def prediction(target_column,forecasted,records):
    final_data = list(np.array(target_column))
    final_data.extend(forecasted) # add future f days
    future_days = pd.date_range(ind[-1],periods=records+1,freq='1H')[1:]
    #print(len(future_days))
    dates = list(ind)
    #print("last record",future_days[-1])
    dates.extend(list(future_days))
    d = pd.DataFrame(data={'prediction':np.array(final_data)})
    d.index = dates
    d['prediction'] = d['prediction'].astype('float64')
    #fig = plt.figure()
    #plt.plot(dates,final_data)
    #s = 'Final plot'
    #plt.title(s)
    #plt.grid()
    #st.pyplot(fig)
    st.line_chart(d)

def predict(train,test,loaded_model):
    y_train_pred = loaded_model.predict(X_tr)
    y_test_pred = loaded_model.predict(X_te)
    return y_train_pred,y_test_pred

def visualize_training(target,train,ytrain,ytest):
    time_step=100
    trainPredictPlot = np.empty_like(target)
    trainPredictPlot[:,:] = np.nan
    trainPredictPlot[time_step:len(train)+time_step, :] = ytrain
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(target)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train)+(time_step*2)+1:len(target)-1, :] = ytest
    # plot baseline and predictions
    fig = plt.figure()
    plt.plot(target,label='actual',color='blue')
    plt.plot(trainPredictPlot,label='train predicted',color='green')
    plt.plot(testPredictPlot,label='test predicted',color='red')
    plt.legend()
    plt.grid()
    plt.show()
    st.pyplot(fig)

if flag==True and create_train_test:
    # Scalling
    target = np.array(df[target_column]).reshape(-1,1)
    
    Std_scaler = MinMaxScaler()
    price_list = pd.DataFrame(data=Std_scaler.fit_transform(target))
    price_list.columns = [target_column]
    price_list.index = df.index
    price_list = np.array(price_list)    
    train_size = round(len(price_list)*0.70)
    test_size = round(len(price_list) - train_size)
    train_data = price_list[:train_size,:]
    test_data = price_list[train_size:,:]
    
    # visualize data in dataframe
    time_step = 100
    
    col=[]
    for i in range(time_step):
        s = 'T'+str(i+1)
        col.append(s)
    col.append('Target')
    
    # create dataset
    
    
    X_train,y_train = create_dataset(train_data,time_step)
    X_test,y_test = create_dataset(test_data,time_step)
    
    X_tr= X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    X_te = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
    
    #model_name = st.sidebar.selectbox('Select model',('Single LSTM', 'Stacked LSTM', 'Bidirectional LSTM'))
    
    prog = st.progress(0)
    for i in range(100):
        time.sleep(0.0001)
        prog.progress(i+1)
        if i+1==100: 
            st.success('Completed :)')
    if data_dict[df_name] == 'BTC-INR':
            if target_column=='Close':
                    
                    file = open('Stacked_LSTM_BTC_Close.json','r')
                    load_model = file.read()
                    model = model_from_json(load_model)
                    file.close()
                    model.load_weights('Stacked_LSTM_BTC_Close.h5')
             
                    y_tr,y_te = predict(X_tr,X_te,model)
             
                    pred_vs_actual(y_train,y_tr,"Training")
                    pred_vs_actual(y_test,y_te,"Testing")
             
                    visualize_training(price_list,X_tr,y_tr,y_te)
             
                    performance_evaluation(y_train,y_tr,y_test,y_te)
                
                    tr_a = Std_scaler.inverse_transform(y_train.reshape(-1,1))
                    tr_p = Std_scaler.inverse_transform(y_tr.reshape(-1,1))
                    
                    mape(tr_a,tr_p,'training')
                    
                    te_a = Std_scaler.inverse_transform(y_test.reshape(-1,1))
                    te_p = Std_scaler.inverse_transform(y_te.reshape(-1,1))
                    
                    mape(te_a,te_p,'testing')
             
                    X_input = test_data[len(test_data)-time_step:]
             
                    temp_input = list(X_input.reshape(time_step))
             
                    forecasted_values = forecast_future(temp_input,f,time_step,model)
                    Actual_forecast = Std_scaler.inverse_transform(forecasted_values)
             
                    prediction(df[target_column],Actual_forecast,f)
                     
                    st.balloons()
             
            else:
             
                    file = open('Stacked_BTC-INR_Avg price.json','r')
                    load_model = file.read()
                    file.close()
                    model = model_from_json(load_model)
                    model.load_weights('Stacked_BTC-INR_Avg price.h5')
             
                    y_tr,y_te = predict(X_tr,X_te,model)
             
                    pred_vs_actual(y_train,y_tr,"Training")
                    pred_vs_actual(y_test,y_te,"Testing")
             
                    visualize_training(price_list,X_tr,y_tr,y_te)
             
                    performance_evaluation(y_train,y_tr,y_test,y_te) 
                    
                    tr_a = Std_scaler.inverse_transform(y_train.reshape(-1,1))
                    tr_p = Std_scaler.inverse_transform(y_tr.reshape(-1,1))
                    
                    mape(tr_a,tr_p,'training')
                    
                    te_a = Std_scaler.inverse_transform(y_test.reshape(-1,1))
                    te_p = Std_scaler.inverse_transform(y_te.reshape(-1,1))
                    
                    mape(te_a,te_p,'testing')

             
                    X_input = test_data[len(test_data)-time_step:]            
                    temp_input = list(X_input.reshape(time_step))
             
                    forecasted_values = forecast_future(temp_input,f,time_step,model)
                    Actual_forecast = Std_scaler.inverse_transform(forecasted_values)
             
                    prediction(df[target_column],Actual_forecast,f)
                    st.balloons()
             
    elif data_dict[df_name] == 'ETH-INR':
            if target_column=='Close':
             
                    file = open('Stacked_ETH-INR_Close.json','r')
                    load_model = file.read()
                    model = model_from_json(load_model)
                    file.close()
                    model.load_weights('Stacked_ETH-INR_Close.h5')
             
                    y_tr,y_te = predict(X_tr,X_te,model)
             
                    pred_vs_actual(y_train,y_tr,"Training")
                    pred_vs_actual(y_test,y_te,"Testing")
             
                    visualize_training(price_list,X_tr,y_tr,y_te)
             
                    performance_evaluation(y_train,y_tr,y_test,y_te) 
                    
                    tr_a = Std_scaler.inverse_transform(y_train.reshape(-1,1))
                    tr_p = Std_scaler.inverse_transform(y_tr.reshape(-1,1))
                    
                    mape(tr_a,tr_p,'training')
                    
                    te_a = Std_scaler.inverse_transform(y_test.reshape(-1,1))
                    te_p = Std_scaler.inverse_transform(y_te.reshape(-1,1))
                    
                    mape(te_a,te_p,'testing')

                    X_input = test_data[len(test_data)-time_step:]            
                    temp_input = list(X_input.reshape(time_step))
             
                    forecasted_values = forecast_future(temp_input,f,time_step,model)
                    Actual_forecast = Std_scaler.inverse_transform(forecasted_values)
             
                    prediction(df[target_column],Actual_forecast,f)
                    st.balloons()
            else:
             
                    file = open('Stacked_ETH-INR_Avg price.json','r')
                    load_model = file.read()
                    file.close()
                    model = model_from_json(load_model)
                    model.load_weights('Stacked_ETH-INR_Avg price.h5')
             
                    y_tr,y_te = predict(X_tr,X_te,model)
             
                    pred_vs_actual(y_train,y_tr,"Training")
                    pred_vs_actual(y_test,y_te,"Testing")
             
                    visualize_training(price_list,X_tr,y_tr,y_te)
             
                    performance_evaluation(y_train,y_tr,y_test,y_te)
                    
                    tr_a = Std_scaler.inverse_transform(y_train.reshape(-1,1))
                    tr_p = Std_scaler.inverse_transform(y_tr.reshape(-1,1))
                    
                    mape(tr_a,tr_p,'training')
                    
                    te_a = Std_scaler.inverse_transform(y_test.reshape(-1,1))
                    te_p = Std_scaler.inverse_transform(y_te.reshape(-1,1))
                    
                    mape(te_a,te_p,'testing')

                    X_input = test_data[len(test_data)-time_step:]            
                    temp_input = list(X_input.reshape(time_step))
             
                    forecasted_values = forecast_future(temp_input,f,time_step,model)
                    Actual_forecast = Std_scaler.inverse_transform(forecasted_values)
             
                    prediction(df[target_column],Actual_forecast,f)
                    st.balloons()
    else:
            if target_column=='Close':
                    file = open('bi_LSTM_DOGE-INR_Close.json','r')
                    load_model = file.read()
                    model = model_from_json(load_model)
                    file.close()
                    model.load_weights('bi_LSTM_DOGE-INR_Close.h5')
             
                    y_tr,y_te = predict(X_tr,X_te,model)
             
                    pred_vs_actual(y_train,y_tr,"Training")
                    pred_vs_actual(y_test,y_te,"Testing")
             
                    visualize_training(price_list,X_tr,y_tr,y_te)
             
                    performance_evaluation(y_train,y_tr,y_test,y_te)  
                    
                    tr_a = Std_scaler.inverse_transform(y_train.reshape(-1,1))
                    tr_p = Std_scaler.inverse_transform(y_tr.reshape(-1,1))
                    
                    mape(tr_a,tr_p,'training')
                    
                    te_a = Std_scaler.inverse_transform(y_test.reshape(-1,1))
                    te_p = Std_scaler.inverse_transform(y_te.reshape(-1,1))
                    
                    mape(te_a,te_p,'testing')

                    X_input = test_data[len(test_data)-time_step:]            
                    temp_input = list(X_input.reshape(time_step))
             
                    forecasted_values = forecast_future(temp_input,f,time_step,model)
                    Actual_forecast = Std_scaler.inverse_transform(forecasted_values)
             
                    prediction(df[target_column],Actual_forecast,f)
                    st.balloons()
            else:
                    file = open('bi_LSTM_DOGE-INR_Avg price.json','r')
                    load_model = file.read()
                    file.close()
                    model = model_from_json(load_model)
                    model.load_weights('bi_LSTM_DOGE-INR_Avg price.h5')
             
                    y_tr,y_te = predict(X_tr,X_te,model)
             
                    pred_vs_actual(y_train,y_tr,"Training")
                    pred_vs_actual(y_test,y_te,"Testing")
             
                    visualize_training(price_list,X_tr,y_tr,y_te)
             
                    performance_evaluation(y_train,y_tr,y_test,y_te)  
                    
                    tr_a = Std_scaler.inverse_transform(y_train.reshape(-1,1))
                    tr_p = Std_scaler.inverse_transform(y_tr.reshape(-1,1))
                    
                    mape(tr_a,tr_p,'training')
                    
                    te_a = Std_scaler.inverse_transform(y_test.reshape(-1,1))
                    te_p = Std_scaler.inverse_transform(y_te.reshape(-1,1))
                    
                    mape(te_a,te_p,'testing')

                    X_input = test_data[len(test_data)-time_step:]            
                    temp_input = list(X_input.reshape(time_step))
             
                    forecasted_values = forecast_future(temp_input,f,time_step,model)
                    Actual_forecast = Std_scaler.inverse_transform(forecasted_values)
             
                    prediction(df[target_column],Actual_forecast,f)
                    st.balloons()
        

            
        
    
    

    
    

    
    
    
    
    
    
    
    
    
        
    #prog = st.progress(0)
    #for i in range(100):
        #time.sleep(0.0001)
        #prog.progress(i+1)
        #if i+1==100: 
            #st.success('Completed :)')
            #st.write(f'###### No. of  training data  :{train_size}')
            #st.write(f'###### No. of test data  : {test_size}')
            # display training data
            #Train_X = pd.DataFrame(data=X_train)
            #Train_Y = pd.Series(y_train)
            #Train_dataset = pd.concat([Train_X,Train_Y],axis=1)
            #Train_dataset = pd.DataFrame(data=Train_dataset.values,columns=col)
            #st.header('Training Data')
            #st.dataframe(Train_dataset.iloc[:100,:],width=900,height=300)           
            # display test data
            #Test_X = pd.DataFrame(data=X_test)
            #Test_Y = pd.Series(y_test)
            #Test_dataset = pd.concat([Test_X,Test_Y],axis=1)
            #Test_dataset = pd.DataFrame(data=Test_dataset.values,columns=col)
            #st.header('Testing Data')
            #st.dataframe(Test_dataset.iloc[:100,:],width=900,height=300)

        

    





#'''fname = Single_LSTM_model.to_json()
#    with open(filename,"w") as file:
#        file.write(fname)
#    Single_LSTM_model.save_weights('Single_LSTM_'+df_name+'_'+target_column+.h5')'''


#'''
#model_name = st.sidebar.selectbox(
#    'Select model',
#    ('Single LSTM', 'Stacked LSTM', 'Bidirectional LSTM')
#)'''
#)'''