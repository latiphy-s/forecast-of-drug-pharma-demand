import json
import numpy as np
import pandas as pd
import pytest
from forecasting import ROOT, load_bundle, validate_data, forecast_frame, fingerprint

@pytest.fixture(scope='module')
def loaded():
    return load_bundle()

def test_notebook_projection_all_horizons(loaded):
    df,model,scaler,metadata=loaded
    reference=json.loads((ROOT/'tests/notebook_projection.json').read_text())
    expected=np.array([r['value'] for r in reference])
    for horizon in range(1,13):
        result=forecast_frame(df,model,scaler,horizon)
        np.testing.assert_allclose(result.iloc[:,1],expected[:horizon],rtol=0,atol=.1)
        assert result.Date.dt.strftime('%Y-%m-%d').tolist()==[r['date'] for r in reference[:horizon]]
    assert metadata['training_epochs']==7

@pytest.mark.parametrize('kind',['missing','nonnumeric','duplicate','gap','negative','bad_date','short','infinite','column'])
def test_reject_invalid_data(loaded,kind):
    df=loaded[0].copy()
    if kind=='missing': df.loc[0,'N02BE']=np.nan
    elif kind=='nonnumeric': df['N02BE']=df.N02BE.astype(object);df.loc[0,'N02BE']='oops'
    elif kind=='duplicate': df.loc[1,'datum']=df.loc[0,'datum']
    elif kind=='gap': df=df.drop(index=10)
    elif kind=='negative': df.loc[0,'N02BE']=-1
    elif kind=='bad_date': df.loc[0,'datum']=pd.NaT
    elif kind=='short': df=df.head(25)
    elif kind=='infinite': df.loc[0,'N02BE']=np.inf
    elif kind=='column': df=df.drop(columns='N02BE')
    with pytest.raises(ValueError): validate_data(df)

def test_missing_files(tmp_path):
    with pytest.raises(ValueError,match='Missing required files'): fingerprint(tmp_path)

@pytest.mark.parametrize('horizon',[0,13,1.5,True])
def test_bad_horizon(loaded,horizon):
    df,m,s,_=loaded
    with pytest.raises(ValueError): forecast_frame(df,m,s,horizon)

def test_app_pages():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
    assert not app.exception
    assert not app.error
    for page in ['Sales history','Seasonal patterns','Forecast','Model & evidence']:
        app.sidebar.radio[0].set_value(page).run(timeout=60)
        assert not app.exception, page
        assert not app.error, page
        if page=='Forecast':
            app.slider[0].set_value(1).run()
            assert len(app.dataframe[0].value)==1
            app.slider[0].set_value(12).run()
            assert len(app.dataframe[0].value)==12
