"""Portfolio dashboard. Run with streamlit run app.py."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import calendar
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from forecasting import ROOT, CATEGORIES, fingerprint, load_bundle, forecast_frame

st.set_page_config(page_title='Sales, in season. | Sebastian Latiphy', page_icon='◈', layout='wide')
st.markdown('''<style>
.stApp {background:radial-gradient(ellipse at 0% 0%,#482316 0%,#211810 23%,#17130f 55%);}
.block-container {max-width:1180px;padding-top:3rem;padding-bottom:3rem;}
h1,h2,h3 {letter-spacing:-.045em;font-weight:500!important;}
h1 {font-size:clamp(2.5rem,6vw,4.8rem)!important;line-height:1.08!important;}
.hero-serif {font-family:Georgia,serif;font-style:italic;color:#ffb875;}
.eyebrow {font-size:.75rem;letter-spacing:.15em;text-transform:uppercase;color:#ffb875;margin:0 0 1.2rem;}
.intro {font-size:1.13rem;max-width:700px;color:#c6b9ac;line-height:1.7;}
[data-testid="stMetric"] {border-top:1px solid #564536;padding-top:1rem;}
[data-testid="stMetricValue"] {font-family:Georgia,serif;}
[data-testid="stRadio"] label {min-height:44px;align-items:center;}
a {color:#ffb875;} button {min-height:44px;} :focus-visible {outline:2px solid #ffb875!important;outline-offset:3px;}
@media(max-width:700px){.block-container{padding:5rem 1rem 2rem;}h1{font-size:2.8rem!important;}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important;}}
</style>''', unsafe_allow_html=True)

@st.cache_resource(show_spinner='Loading the saved forecasting model…')
def bundle(signature):
    return load_bundle()

@st.cache_data(show_spinner=False)
def project(signature, steps):
    df, model, scaler, _ = bundle(signature)
    return forecast_frame(df, model, scaler, steps)

with st.sidebar:
    st.markdown('### ◈ Latiphy')
    st.caption('PHARMACEUTICAL SALES / RESEARCH')
    page = st.radio('Explore the project', ['Overview', 'Sales history', 'Seasonal patterns', 'Forecast', 'Model & evidence'])
    st.divider()
    st.markdown('**A historical forecasting study.**')
    st.caption('2018–2023 observations. Projections begin January 2024, not today.')
    st.link_button('Visit portfolio ↗', 'https://www.latiphy.com/')

try:
    signature = fingerprint()
    df, model, scaler, metadata = bundle(signature)
except Exception as exc:
    st.error(f'The forecasting bundle could not be loaded: {exc}')
    st.info('Check the CSV, pharma_lstm_model.keras, scaler.pkl and forecast_metadata.json. Install the pinned requirements. No fallback model or automatic retraining is used.')
    st.stop()

AMBER = '#FFB875'
COLORS = ['#A8B9FF','#D7997B','#D5BF88','#FFB875','#C5AAD8','#98C6B5','#8CBED0','#D99FB4']

def styled(fig, height=420):
    fig.update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#C6B9AC', family='Arial', size=13),
                      margin=dict(l=12,r=16,t=35,b=12), hovermode='x unified',
                      legend=dict(orientation='h',y=-.22), dragmode=False)
    fig.update_xaxes(title='Week',showgrid=False)
    fig.update_yaxes(title='Weekly sales (Extended Units)',gridcolor='#3B3027',zeroline=False)
    return fig

def plot(fig):
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False, 'scrollZoom': False})

def heading(kicker, title, subtitle):
    st.markdown(f'<p class="eyebrow">{kicker}</p>',unsafe_allow_html=True)
    st.title(title)
    st.write(subtitle)

if page == 'Overview':
    st.markdown('<p class="eyebrow">SEBASTIAN LATIPHY / FORECASTING STUDY</p><h1>Sales,<br><span class="hero-serif">in season.</span></h1><p class="intro">Explore six years of pharmaceutical sales and a recursive LSTM projection. Connect observed patterns with the questions behind purchasing and replenishment plans.</p>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric('Weekly observations',len(df)); b.metric('Sales categories','8'); c.metric('Projection horizon','1–12 weeks')
    fig=go.Figure(go.Scatter(x=df['datum'],y=df['N02BE'],name='N02BE',line=dict(color=AMBER,width=2)))
    plot(styled(fig))
    st.caption('N02BE observed sales. Supplied CSV: January 2018 to December 2023. Units and Northeast US market are source labels; provenance is not independently verified.')
    st.subheader('From a pattern to a planning question')
    a,b,c=st.columns(3)
    with a: st.markdown('**01 / Observe**'); st.write('Compare eight categories and inspect the recurring spring sales peak.')
    with b: st.markdown('**02 / Project**'); st.write('Extend the last 26 N02BE observations by up to 12 weeks using the saved model.')
    with c: st.markdown('**03 / Evaluate**'); st.write('Review matched-origin evidence and the limits of using sales to plan inventory.')

elif page == 'Sales history':
    heading('01 / OBSERVE','Eight categories. Six years.','Compare weekly recorded sales. Category totals represent sales, not unconstrained demand.')
    selected=st.multiselect('Categories',CATEGORIES,default=['N02BE'])
    if not selected: st.info('Choose at least one category to display its sales history.')
    else:
        fig=go.Figure()
        for category in selected:
            fig.add_trace(go.Scatter(x=df['datum'],y=df[category],name=category,line=dict(color=COLORS[CATEGORIES.index(category)],width=2)))
        plot(styled(fig))
        with st.expander('Inspect weekly values'):
            st.dataframe(df[['datum',*selected]],hide_index=True,width='stretch')
    st.caption('Source: data/iqvia_pharma_sales.csv. All eight series use Extended Units. Hover to inspect values, or open the accessible data table.')

elif page == 'Seasonal patterns':
    heading('02 / INTERPRET','A spring signal.','N02BE average weekly sales peak in March–May in this dataset. The pattern does not identify a clinical cause.')
    monthly=df.assign(Year=df.datum.dt.year,Month=df.datum.dt.month).pivot_table(index='Year',columns='Month',values='N02BE',aggfunc='mean')
    fig=go.Figure(go.Heatmap(z=monthly.values,x=[calendar.month_abbr[m] for m in monthly.columns],y=monthly.index.astype(str),colorscale=[[0,'#28201A'],[.5,'#B55228'],[1,'#FFE0AB']],colorbar=dict(title='Units'),hovertemplate='%{y} %{x}<br>%{z:,.0f} units<extra></extra>'))
    fig=styled(fig,380);fig.update_xaxes(title='Month');fig.update_yaxes(title='Year',autorange='reversed')
    plot(fig)
    st.caption('Mean weekly sales within each calendar month, not monthly totals. March 2020 is an exceptional high-sales month.')
    with st.expander('Monthly averages as a table'): st.dataframe(monthly.rename(columns=lambda m:calendar.month_abbr[m]).round(0),width='stretch')
    st.subheader('What this could mean for planning')
    st.markdown('''- **Review spring purchasing:** check supplier capacity and delivery schedules ahead of the observed high-sales period. Confirm the pattern in recent data first.
- **Plan around lead times:** combine projections with stock on hand, expiry dates, delivery delays and service targets.
- **Measure the benefit:** compare stockout days, expiry waste and inventory costs in a replenishment pilot. Savings have not been measured.''')

elif page == 'Forecast':
    heading('03 / PROJECT','The next twelve weeks.','Exploratory N02BE sales projections from the end of this dataset. These are historical projections, not forecasts for today.')
    steps=st.slider('Projection horizon (weeks)',1,12,12)
    try: future=project(signature,steps)
    except Exception as exc:
        st.error(f'Projection could not be generated: {exc}');st.stop()
    start=future['Date'].iloc[0]; end=future['Date'].iloc[-1]
    st.caption(f'Forecast origin: 25 December 2023 · Projection dates: {start:%d %b %Y} to {end:%d %b %Y}')
    if (future.iloc[:,1]<0).any(): st.warning('The model generated negative sales projections. They are shown unchanged and are unsuitable for purchasing decisions.')
    hist=df.tail(52)
    fig=go.Figure(go.Scatter(x=hist.datum,y=hist.N02BE,name='Observed sales',line=dict(color='#A8B9FF',width=2)))
    fig.add_trace(go.Scatter(x=future.Date,y=future.iloc[:,1],name='Exploratory projection',mode='lines+markers',line=dict(color=AMBER,width=2)))
    fig.add_shape(type='line',x0=start,x1=start,y0=0,y1=1,xref='x',yref='paper',line=dict(color='#C6B9AC',dash='dash'))
    fig.add_annotation(x=start,y=1,xref='x',yref='paper',text='Forecast begins',showarrow=False,yshift=14,xanchor='right')
    plot(styled(fig))
    st.caption('Each predicted week becomes an input for the next. No seasonal multipliers or prediction intervals are added. A smooth trajectory does not imply certainty.')
    st.dataframe(future, column_config={'Date': st.column_config.DateColumn(format='DD MMM YYYY'), 'Projected sales (Extended Units)': st.column_config.NumberColumn(format='localized', help='Exploratory weekly sales in Extended Units')}, hide_index=True, width='stretch')
    st.download_button('Download projection CSV',future.to_csv(index=False).encode(),file_name=f'n02be_projection_{steps}_weeks.csv',mime='text/csv')

else:
    heading('04 / EVIDENCE','A model, with boundaries.','A univariate LSTM takes the previous 26 weekly N02BE observations and predicts the next value.')
    st.markdown('**26 observations → saved scaling → LSTM → inverse scaling → next week**')
    st.write('The dashboard loads the all-history model trained for seven epochs. For longer horizons, predictions are fed back into the input window. It does not retrain or use an API.')
    st.subheader('Matched-origin evaluation')
    evidence=json.loads((ROOT/'evaluation.json').read_text())
    horizon=st.selectbox('Recorded evaluation horizon',['First 12 weeks','Full 58-week holdout'])
    st.dataframe(pd.DataFrame(evidence['first_12' if horizon=='First 12 weeks' else 'full']),hide_index=True,width='stretch')
    st.caption('Source: notebook section 6, saved output of cell 29. Origin: 14 November 2022. First 12 weeks: 21 November 2022 to 6 February 2023. Full holdout: 21 November 2022 to 25 December 2023. MAE and RMSE are Extended Units; MAPE is percent.')
    st.write('Every model forecasts from the same cutoff without receiving holdout sales. These scores belong to the pre-test evaluation models, not the all-history dashboard model. One origin cannot establish a best model.')
    st.subheader('Before using it for purchasing')
    st.markdown('''- Evaluate multiple forecast origins at the required decision horizon.
- Validate uncertainty coverage; this app provides point projections only.
- Add inventory availability, lead times, service targets and costs. Observed sales can be constrained by stockouts.
- Refresh the data and reassess the model. The supplied history ends in 2023.''')
    with st.expander('Saved artifact metadata'): st.json(metadata)

st.divider()
st.caption('Sebastian Latiphy · Pharmaceutical sales forecasting · Historical study / Exploratory projections')
