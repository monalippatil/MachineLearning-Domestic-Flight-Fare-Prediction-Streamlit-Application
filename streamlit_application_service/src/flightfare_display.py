
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, time
from src.flightfare_predictions import predict_adaboost_tyler_model, predict_tf_kesar_monali_model, predict_tf_kesar_michael_model, predict_tf_kesar_charles_model


def flightfare_display_menu():
    """
    --------------------
    Description
    --------------------
    Function to display menu on the streamlit app and extract user inputs
    
    --------------------
    Parameters
    --------------------
    None

    --------------------
    Pseudo-Code
    --------------------
    Display menu to the user
    Extract user provided inputs
            Name of departure airport
            Name of destination airport
            Departure date
            Departure time
            Type of cabin
            Ticket type (Optional)
            Economy ticket (Optional)
    Check if user has selected mandatory inputs
    Process the user inputs and extract additional features from date input
    Store user inputs into session state variables
    Call to function to display prediction results
    
    --------------------
    Returns
    --------------------
    """    
    
    # Streamlit app title
    st.title('US Flights Fare Predictor')

    # Mandatory user inputs
    origin_airport = st.selectbox('Select Origin Airport (Mandatory)', ('ATL - Atlanta Hartsfield International Airport',
                                                            'BOS - Boston, Logan International Airport',
                                                            'CLT - Charlotte/Douglas International Airport',
                                                            'DEN - Denver International Airport',
                                                            'DFW - Dallas/Fort Worth International Airport',
                                                            'DTW - Detroit Metro Wayne County Airport',
                                                            'EWR - Newark Liberty International Airport',
                                                            'IAD - Washington, Dulles International Airport',
                                                            'JFK - New York, John F Kennedy International Airport',
                                                            'LAX - Los Angeles International Airport',
                                                            'LGA - New York, LaGuardia Airport',
                                                            'MIA - Miami International Airport',
                                                            'OAK - Oakland International Airport',
                                                            "ORD - Chicago, O'Hare International Airport Airport",
                                                            'PHL - Philadelphia International Airport',
                                                            'SFO - San Francisco International Airport'), 
                                                            index=None, placeholder='Select Origin Airport...', )

    destination_airport = st.selectbox('Select Destination Airport (Mandatory)', ('ATL - Atlanta Hartsfield International Airport',
                                                                    'BOS - Boston, Logan International Airport',
                                                                    'CLT - Charlotte/Douglas International Airport',
                                                                    'DEN - Denver International Airport',
                                                                    'DFW - Dallas/Fort Worth International Airport',
                                                                    'DTW - Detroit Metro Wayne County Airport',
                                                                    'EWR - Newark Liberty International Airport',
                                                                    'IAD - Washington, Dulles International Airport',
                                                                    'JFK - New York, John F Kennedy International Airport',
                                                                    'LAX - Los Angeles International Airport',
                                                                    'LGA - New York, LaGuardia Airport',
                                                                    'MIA - Miami International Airport',
                                                                    'OAK - Oakland International Airport',
                                                                    "ORD - Chicago, O'Hare International Airport Airport",
                                                                    'PHL - Philadelphia International Airport',
                                                                    'SFO - San Francisco International Airport'), 
                                                                    index=None, placeholder='Select Destination Airport...',)

    departure_date = st.date_input('Select Departure Date (Mandatory)', value=None, format='MM.DD.YYYY')

    departure_time = st.time_input('Select Preferred Time (Mandatory)', value=None)            

    cabin_type = st.selectbox('Select Cabin Type (Mandatory)', ('Couch', 'Premium', 'Business', 'First Class'), index=None, placeholder='Select Cabin Type...',)

    # Optional user input
    cols=st.columns(3)
    with cols[0]:
        isRefundable = st.radio('Select the ticket type?', ('Refundable', 'Non Refundable'), index=0) 
    with cols[2]:
        isBasicEconomy = st.radio('Select whether an economy ticket?', ('Yes', 'No'), index=0)

    # Check if the user has made a selection 
    if st.button('Predict Flightfare'):
        if origin_airport is None:
            st.write('Please select a departure airport.')
        elif destination_airport is None:
            st.write('Please select a destination airport.')
        elif departure_date is None:
            st.write('Please select the departure date.')
        elif departure_time is None:
            st.write('Please select the departure time.')
        elif cabin_type is None:
            st.write('Please select cabin type.')
        else: 
            # Processing mandatory input values
            # Extract only airport name abbrevations 
            startingAirport = origin_airport.split(' - ')[0][:3]
            destinationAirport = destination_airport.split(' - ')[0][:3]

            # Extract month, day and day-of-month features from the date input
            flightDepartureMonth = datetime.strftime(departure_date, '%m')
            flightDepartureDay = datetime.strftime(departure_date, '%d')
            flightDepartureDOW = datetime.strftime(departure_date, '%w')

            # Seperate hour and minute from the time input
            flightDepartureHour = time.strftime(departure_time, '%H')
            flightDepartureMinute = time.strftime(departure_time, '%M')

            # Extract only first letter from the cabin type input
            cabinCode = cabin_type[0][0]

            # Processing optional input values
            if isRefundable == 'Refundable':
                isRefundable = True
            else:
                isRefundable = False

            if isBasicEconomy == 'Yes':
                isBasicEconomy = True
            else:
                isBasicEconomy = False

            # Store user input values into the Session state variables
            st.session_state['flightDepartureMonth'] = flightDepartureMonth
            st.session_state['flightDepartureDay'] = flightDepartureDay
            st.session_state['flightDepartureDOW'] = flightDepartureDOW
            st.session_state['flightDepartureHour'] = flightDepartureHour
            st.session_state['flightDepartureMinute'] = flightDepartureMinute
            st.session_state['startingAirport'] = startingAirport
            st.session_state['destinationAirport'] = destinationAirport
            st.session_state['isBasicEconomy'] = isBasicEconomy
            st.session_state['isRefundable'] = isRefundable
            st.session_state['cabinCode'] = cabinCode

            # Display results to the users
            flightfare_display_results()


def flightfare_display_results():
    """
    --------------------
    Description
    --------------------
    Function to call prediction functions using 4 models and display results to the user on the streamlit app

    --------------------
    Parameters
    --------------------
    None

    --------------------
    Pseudo-Code
    --------------------
    Call to prediction functions using 4 models
    Extract prediction results from session state variables
    Combine prediction results in a table
    Display prediction results to the user on the streamlit app

    --------------------
    Returns
    --------------------
    """        

    # Predict using adaboost model 1
    predict_adaboost_tyler_model()

    # Predict using tf keras model 2
    predict_tf_kesar_monali_model()

    # Predict using tf keras model 3
    predict_tf_kesar_michael_model()

    # Predict using tf keras model 4
    predict_tf_kesar_charles_model()

    # Extract prediction results 
    m1_result_direct = st.session_state['m1_result_direct']
    m1_result_transit = st.session_state['m1_result_transit']
       
    m2_result_direct = st.session_state['m2_result_direct']
    m2_result_transit = st.session_state['m2_result_transit']

    m3_result_direct = st.session_state['m3_result_direct']
    m3_result_transit = st.session_state['m3_result_transit']

    m4_result_direct = st.session_state['m4_result_direct']
    m4_result_transit = st.session_state['m4_result_transit']

    # Combine prediction results into a table
    predict_results = [['Bookings', m1_result_direct, m1_result_transit],
                       ['Flight Center', m2_result_direct, m2_result_transit],
                       ['Expedia', m3_result_direct, m3_result_transit],
                       ['Kayak', m4_result_direct, m4_result_transit],]

    # Display results to the user
    st.markdown('###### Predicted Flight Fares')
    fare_predictions = pd.DataFrame(predict_results, columns=['Sources', 'Direct Flight', 'Connecting Flight'])
    st.table(fare_predictions)



    