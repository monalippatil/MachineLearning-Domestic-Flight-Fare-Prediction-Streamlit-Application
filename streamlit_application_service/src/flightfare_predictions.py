import numpy as np
import pandas as pd
import streamlit as st
import joblib
import tensorflow as tf


def predict_adaboost_tyler_model():
    """
    --------------------
    Description
    --------------------
    Function to predict flight fare using adaboost model 1 and store prediction results into session state variables

    --------------------
    Parameters
    --------------------
    None

    --------------------
    Pseudo-Code
    --------------------
    Extract input values from session state variables
    Combine input values into dataframes 
    Load adaboost model 1
    Predict flight fare for direct and connecting flight
    Format predicted results 
    Store predicted results into session state variables

    --------------------
    Returns
    --------------------
    """        

    # Extract user input from session state variables
    flightDepartureMonth = (st.session_state['flightDepartureMonth'])
    flightDepartureDay = (st.session_state['flightDepartureDay'])
    flightDepartureDOW = (st.session_state['flightDepartureDOW'])
    flightDepartureHour = (st.session_state['flightDepartureHour'])
    flightDepartureMinute = (st.session_state['flightDepartureMinute'])
    startingAirport = st.session_state['startingAirport']
    destinationAirport = st.session_state['destinationAirport']
    isBasicEconomy = (st.session_state['isBasicEconomy'])
    isRefundable = (st.session_state['isRefundable'])
    cabinCode = st.session_state['cabinCode']
    
    # Combine input values for direct flight into a dataframe
    isNonStop = False
    features_direct = pd.DataFrame({'flightDate_month': flightDepartureMonth, 
                                'flightDate_day': flightDepartureDay, 
                                'flightDate_dow': flightDepartureDOW, 
                                'flightDepartureHour': flightDepartureHour, 
                                'flightDepartureMinute': flightDepartureMinute,
                                'isBasicEconomy': isBasicEconomy,
                                'isRefundable': isRefundable,
                                'isNonStop': isNonStop, 
                                'startingAirport': startingAirport, 
                                'destinationAirport': destinationAirport, 
                                'cabinCode': cabinCode.lower()}, index=[0])

    # Combine input values for flight with transit into a dataframe
    isNonStop = True
    features_transit = pd.DataFrame({'flightDate_month': flightDepartureMonth, 
                                'flightDate_day': flightDepartureDay, 
                                'flightDate_dow': flightDepartureDOW, 
                                'flightDepartureHour': flightDepartureHour, 
                                'flightDepartureMinute': flightDepartureMinute,
                                'isBasicEconomy': isBasicEconomy,
                                'isRefundable': isRefundable,
                                'isNonStop': isNonStop, 
                                'startingAirport': startingAirport, 
                                'destinationAirport': destinationAirport, 
                                'cabinCode': cabinCode.lower()}, index=[0])

    # Load adaboost model 1
    model = joblib.load('models/adaboost.joblib')

    # Predict flight fares using the adaboost model 1 for direct and connecting flight
    predict_fare_direct = model.predict(features_direct)
    predict_fare_transit = model.predict(features_transit)

    # Format and store predicted results into session state variables
    st.session_state['m1_result_direct'] = f'${round(predict_fare_direct.tolist()[0], 2)}'
    st.session_state['m1_result_transit'] = f'${round(predict_fare_transit.tolist()[0], 2)}'


def predict_tf_kesar_monali_model():
    """
    --------------------
    Description
    --------------------
    Function to predict flight fare using tensorflow keras model 2 and store prediction results into session state variables

    --------------------
    Parameters
    --------------------
    None

    --------------------
    Pseudo-Code
    --------------------
    Extract and transform input values from session state variables 
    Combine input values into dataframes
    Load tensorflow keras model 2
    Predict flight fare for direct and connecting flight
    Format predicted results 
    Store predicted results into session state variables

    --------------------
    Returns
    --------------------
    """        

    # Extract and transform input values from session state variables 
    flightDepartureMonth = int(st.session_state['flightDepartureMonth'])
    flightDepartureDay = int(st.session_state['flightDepartureDay'])
    flightDepartureDOW = int(st.session_state['flightDepartureDOW'])
    flightDepartureHour = int(st.session_state['flightDepartureHour'])
    flightDepartureMinute = int(st.session_state['flightDepartureMinute'])
    startingAirport = st.session_state['startingAirport']
    destinationAirport = st.session_state['destinationAirport']
    isBasicEconomy = str(st.session_state['isBasicEconomy'])
    isRefundable = str(st.session_state['isRefundable'])
    cabinCode = st.session_state['cabinCode']

    # Combine input values for direct flight into a dataframe
    isNonStop = True
    isNonStop = str(isNonStop)
    features_direct = pd.DataFrame({'flightDepartureMonth': flightDepartureMonth, 
                                    'flightDepartureDay': flightDepartureDay, 
                                    'flightDepartureDOW': flightDepartureDOW, 
                                    'flightDepartureHour': flightDepartureHour, 
                                    'flightDepartureMinute': flightDepartureMinute,
                                    'isBasicEconomy': isBasicEconomy,
                                    'isRefundable': isRefundable,
                                    'isNonStop': isNonStop, 
                                    'startingAirport': startingAirport, 
                                    'destinationAirport': destinationAirport, 
                                    'cabinCode': cabinCode.lower()}, index=[0])

    # Combine input values for flight with transit into a dataframe
    isNonStop = False
    isNonStop = str(isNonStop)
    features_transit = pd.DataFrame({'flightDepartureMonth': flightDepartureMonth, 
                                    'flightDepartureDay': flightDepartureDay, 
                                    'flightDepartureDOW': flightDepartureDOW, 
                                    'flightDepartureHour': flightDepartureHour, 
                                    'flightDepartureMinute': flightDepartureMinute,
                                    'isBasicEconomy': isBasicEconomy,
                                    'isRefundable': isRefundable,
                                    'isNonStop': isNonStop, 
                                    'startingAirport': startingAirport, 
                                    'destinationAirport': destinationAirport, 
                                    'cabinCode': cabinCode.lower()}, index=[0])
    
    # Load tensorflow keras model 2
    model = tf.keras.models.load_model('models/tf_keras_model')

    # Predict flight fares using the tensorflow keras model 2 for direct and connecting flight
    predict_fare_direct = model.predict(features_direct.to_dict('series'))
    predict_fare_transit = model.predict(features_transit.to_dict('series'))
            
    # Format and store predicted results into session state variables
    flight_direct = predict_fare_direct[0][0]
    flight_transit = predict_fare_transit[0][0]

    st.session_state['m2_result_direct'] = '${:.2f}'.format(flight_direct)
    st.session_state['m2_result_transit'] = '${:.2f}'.format(flight_transit)


def predict_tf_kesar_michael_model():
    """
    --------------------
    Description
    --------------------
    Function to predict flight fare using tensorflow keras model 3 and store prediction results into session state variables

    --------------------
    Parameters
    --------------------
    None

    --------------------
    Pseudo-Code
    --------------------
    Extract and transform input values from session state variables 
    Combine input values into dataframes
    Load tensorflow keras model 3
    Predict flight fare for direct and connecting flight
    Format predicted results 
    Store predicted results into session state variables

    --------------------
    Returns
    --------------------
    """        

    # Extract and transform input values from session state variables 
    flightDepartureMonth = int(st.session_state['flightDepartureMonth'])
    flightDepartureDay = int(st.session_state['flightDepartureDay'])
    flightDepartureDOW = int(st.session_state['flightDepartureDOW'])
    flightDepartureHour = int(st.session_state['flightDepartureHour'])
    flightDepartureMinute = int(st.session_state['flightDepartureMinute'])
    startingAirport = st.session_state['startingAirport']
    destinationAirport = st.session_state['destinationAirport']
    isBasicEconomy = (st.session_state['isBasicEconomy'])
    isRefundable = (st.session_state['isRefundable'])
    cabinCode = st.session_state['cabinCode']
    
    # Combine input values for direct flight into a dataframe
    isNonStop = False
    features_direct = pd.DataFrame({'startingAirport': startingAirport,
                                        'destinationAirport': destinationAirport,
                                        'isBasicEconomy': isBasicEconomy,
                                        'isRefundable': isRefundable,
                                        'isNonStop': isNonStop,
                                        'cabinCode': cabinCode.lower(),
                                        'flightDepartureHour': flightDepartureHour,
                                        'flightDepartureMinute': flightDepartureMinute,
                                        'flightDepartureMonth': flightDepartureMonth,
                                        'flightDepartureDay': flightDepartureDay,
                                        'flightDepartureDayofweek': flightDepartureDOW
                                        }, index=[0])

  
    # Combine input values for flight with transit into a dataframe
    isNonStop = True
    features_transit = pd.DataFrame({'startingAirport': startingAirport,
                                        'destinationAirport': destinationAirport,
                                        'isBasicEconomy': isBasicEconomy,
                                        'isRefundable': isRefundable,
                                        'isNonStop': isNonStop,
                                        'cabinCode': cabinCode.lower(),
                                        'flightDepartureHour': flightDepartureHour,
                                        'flightDepartureMinute': flightDepartureMinute,
                                        'flightDepartureMonth': flightDepartureMonth,
                                        'flightDepartureDay': flightDepartureDay,
                                        'flightDepartureDayofweek': flightDepartureDOW
                                        }, index=[0])
    
    # Load tensorflow keras model 3
    model = tf.keras.models.load_model('models/tf_model_12')

    # Predict flight fares using the tensorflow keras model 2 for direct and connecting flight
    predict_fare_direct = model.predict(dict(features_direct))
    predict_fare_transit = model.predict(dict(features_transit))
 
    # Format and store predicted results into session state variables
    rounded_direct_fare = [ round(elem, 2) for elem in predict_fare_direct.tolist()[0] ]
    direct_flight_result = f'${rounded_direct_fare[0]}'
 
    rounded_transit_fare = [ round(elem, 2) for elem in predict_fare_transit.tolist()[0] ]
    transit_flight_result = f'${rounded_transit_fare[0]}'

    st.session_state['m3_result_direct'] = direct_flight_result
    st.session_state['m3_result_transit'] = transit_flight_result


def predict_tf_kesar_charles_model():
    """
    --------------------
    Description
    --------------------
    Function to predict flight fare using tensorflow keras model 4 and store prediction results into session state variables

    --------------------
    Parameters
    --------------------
    None

    --------------------
    Pseudo-Code
    --------------------
    Extract and transform input values from session state variables 
    Combine input values into dataframes
    Load tensorflow keras model 4
    Predict flight fare for direct and connecting flight
    Format predicted results 
    Store predicted results into session state variables

    --------------------
    Returns
    --------------------
    """        

    # Extract and transform input values from session state variables 
    flightDepartureMonth = int(st.session_state['flightDepartureMonth'])
    flightDepartureDay = int(st.session_state['flightDepartureDay'])
    flightDepartureDOW = int(st.session_state['flightDepartureDOW'])
    flightDepartureHour = int(st.session_state['flightDepartureHour'])
    flightDepartureMinute = int(st.session_state['flightDepartureMinute'])
    startingAirport = st.session_state['startingAirport']
    destinationAirport = st.session_state['destinationAirport']
    isBasicEconomy = (st.session_state['isBasicEconomy'])
    isRefundable = (st.session_state['isRefundable'])
    cabinCode = st.session_state['cabinCode']

    # Combine input values for direct flight into a dataframe
    isNonStop = False
    features_direct_charles = pd.DataFrame({'startingAirport': startingAirport,
                                        'destinationAirport': destinationAirport,
                                        'isBasicEconomy': isBasicEconomy,
                                        'isRefundable': isRefundable,
                                        'isNonStop': isNonStop,
                                        'cabinCode': cabinCode.lower(),
                                        'flightDepartureHour': flightDepartureHour,
                                        'flightDepartureMinute': flightDepartureMinute,
                                        'flightDate_month': flightDepartureMonth,
                                        'flightDate_Day': flightDepartureDay,
                                        'flightDate_DayofWeek': flightDepartureDOW
                                        }, index=[0])


    # Combine input values for flight with transit into a dataframe
    isNonStop = True
    features_transit_charles = pd.DataFrame({'startingAirport': startingAirport,
                                        'destinationAirport': destinationAirport,
                                        'isBasicEconomy': isBasicEconomy,
                                        'isRefundable': isRefundable,
                                        'isNonStop': isNonStop,
                                        'cabinCode': cabinCode.lower(),
                                        'flightDepartureHour': flightDepartureHour,
                                        'flightDepartureMinute': flightDepartureMinute,
                                        'flightDate_month': flightDepartureMonth,
                                        'flightDate_Day': flightDepartureDay,
                                        'flightDate_DayofWeek': flightDepartureDOW
                                        }, index=[0])

    def df_to_dataset(features, target, shuffle=True, batch_size=32):
        ds = tf.data.Dataset.from_tensor_slices((features.to_dict(orient='list'), target))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(features))
        ds = ds.batch(batch_size)
        return ds

    ### for direct dataframe
    features_direct_processed = features_direct_charles.copy()  # Make a copy to avoid modifying the original data

    # Step 1: Apply Preprocessing
    features_direct_processed['isBasicEconomy'] = features_direct_processed['isBasicEconomy'].astype(bool)
    features_direct_processed['isRefundable'] = features_direct_processed['isRefundable'].astype(bool)
    features_direct_processed['isNonStop'] = features_direct_processed['isNonStop'].astype(bool)

    # Step 2: Create a Dataset
    direct_to_predict = df_to_dataset(features_direct_processed, None, shuffle=False, batch_size=1)

    #### For transit dataframe
    features_transit_processed = features_transit_charles.copy()  # Make a copy to avoid modifying the original data

    # Step 1: Apply Preprocessing
    features_transit_processed['isBasicEconomy'] = features_transit_processed['isBasicEconomy'].astype(bool)
    features_transit_processed['isRefundable'] = features_transit_processed['isRefundable'].astype(bool)
    features_transit_processed['isNonStop'] = features_transit_processed['isNonStop'].astype(bool)

    # Step 2: Create a Dataset
    transit_to_predict = df_to_dataset(features_transit_processed, None, shuffle=False, batch_size=1)

    # Load tensorflow keras model 4
    model_charles = tf.keras.models.load_model('models/model_drop.tf')

    # Predict flight fares using the tensorflow keras model 2 for direct and connecting flight
    predict_fare_direct_charles = model_charles.predict(direct_to_predict)
    predict_fare_transit_charles = model_charles.predict(transit_to_predict)

    # Format and store predicted results into session state variables
    rounded_direct_fare = [ round(elem, 2) for elem in predict_fare_direct_charles.tolist()[0] ]
    direct_flight_result = f'${rounded_direct_fare[0]}'

    rounded_transit_fare = [ round(elem, 2) for elem in predict_fare_transit_charles.tolist()[0] ]
    transit_flight_result = f'${rounded_transit_fare[0]}'

    st.session_state['m4_result_direct'] = direct_flight_result
    st.session_state['m4_result_transit'] = transit_flight_result


