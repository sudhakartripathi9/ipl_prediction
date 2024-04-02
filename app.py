import streamlit as st
import pickle
import pandas as pd
# import joblib

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

file = open('pipe.pkl', 'rb')
pipe = pickle.load(file)
st.title('IPL win predictor ')
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Taget score')
c1, c2, c3 = st.columns(3)

with c1:
    score = st.number_input('Current Score')

with c2:
    over = st.number_input('Over left')

with c3:
    wicket = st.number_input('Wicket')

if st.button('Predict probability'):
    run_left = target - score
    ball_left = 120 - over * 6
    crr = score / over
    rrr = (target * 6) / ball_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                             'run_left': [run_left], 'ball_left': [ball_left], 'wicket_left': [wicket],
                             'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    result = pipe.predict_proba(input_df)

    prob1 = result[0][0]
    prob2 = result[0][1]
    
    st.header(batting_team + '-' + str(round(prob2*100)) + '%')
    st.header(bowling_team + '-' + str(round(prob1 * 100)) + '%')
