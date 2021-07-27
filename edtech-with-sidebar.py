import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import calendar
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from PIL import Image
import base64
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import re


st.set_page_config(
     page_title="EdTech Tool",
     layout="wide",
     initial_sidebar_state="collapsed",
)

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

show_streamlit_style = """
            <style>
            footer:after {
            	content:'Developed By J.A.A.';
            	visibility: visible;}
            </style>
            """
st.markdown(show_streamlit_style, unsafe_allow_html=True)


#function to read the data with caching
@st.cache(allow_output_mutation=True)
def load_data(path):
    df=pd.read_csv(path)
    return df

url1 = 'https://drive.google.com/file/d/1ydzkkv8saB5KGnUt5Z2e6Es1bfIZUdrP/view?usp=sharing'
path1 = 'https://drive.google.com/uc?export=download&id='+url1.split('/')[-2]

url2 = 'https://drive.google.com/file/d/1olz9vzJaWwz4edWXcq2pUMJ4KmDrBq6a/view?usp=sharing'
path2 = 'https://drive.google.com/uc?export=download&id='+url2.split('/')[-2]

url3 = 'https://drive.google.com/file/d/1T4ZURpO1a8pAvRgvBwKBJ4x3bVdrTBTa/view?usp=sharing'
path3 = 'https://drive.google.com/uc?export=download&id='+url3.split('/')[-2]

url4 = 'https://drive.google.com/file/d/13IZgG1_YjityXlP2qiHVRKueYEH3yGhD/view?usp=sharing'
path4 = 'https://drive.google.com/uc?export=download&id='+url4.split('/')[-2]

url5 = 'https://drive.google.com/file/d/1QvrTkBnX6dTjdp14MaQuTvDmxuBMRvJh/view?usp=sharing'
path5 = 'https://drive.google.com/uc?export=download&id='+url5.split('/')[-2]

url6 = 'https://drive.google.com/file/d/1HVs4DzhmTuJLQ54Lq1BkEv2rjFAgMCvE/view?usp=sharing'
path6 = 'https://drive.google.com/uc?export=download&id='+url6.split('/')[-2]

df1=load_data(path1)
df2=load_data(path2)
df3=load_data(path3)
df4=load_data(path4)
df5=load_data(path5)
df6=load_data(path6)
# convert "Founded" column from float to int
# df2.replace(-np.Inf, np.nan)
# st.write(df2)
# col=np.array(df1['Founded'], np.int)
# df1['Founded']=col
# col=np.array(df2['Founded'], np.int)
# df2['Founded']=col

frames = [df1, df2, df3, df4, df5, df6]
final_df = pd.concat(frames)
final_df.reset_index(drop=True, inplace=True)
final_df.drop(['Unnamed: 0','Score'], axis=1,inplace=True)
final_df = final_df.drop_duplicates(subset=['title'], keep='first')
final_df[['Founded']] = final_df[['Founded']].fillna(value=0)
final_df = final_df.astype({"Founded": int})
# st.write(final_df["Founded"].unique())
# for i in range (final_df.shape[0]):
#     # st.write(final_df["Founded"][i])
#     if final_df["Founded"][i] ==-2147483648:
#         final_df["Founded"][i]=0

# st.write(df2)

final_df=final_df.rename(columns={"title": "Company", "about": "About","Industries":"Industry"}, errors="raise")
final_df.reset_index(inplace=True)
final_df.drop(['index'], axis=1,inplace=True)


# writing on the sidebar
url = 'https://drive.google.com/file/d/1qcwhWj62I_t7qApTgIzsbZ_ftLsOHq05/view?usp=sharing'
msba_logo = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
st.sidebar.image(msba_logo, use_column_width='auto')


html = '''
<p style="text-align: center; font-size: 15px">My name is Jad Abou Assaly, a Computer Engineer with a Masters in Business Analytics.
Being a Tech Savvy and a Data Science enthusiast, I strive to deliver <b>Data driven</b> solutions for end users.</p>
<hr style="background-color:#fedf46;class="rounded"">
'''
st.sidebar.markdown(html, unsafe_allow_html=True)

# <span style="color: blue;">word</span>
st.sidebar.subheader("Navigate the App")

option = st.sidebar.radio( '',
        ('Home','Data Exploration', 'Data Filtering', 'Scoring', 'Data Analysis using NLP'))

#adding a line in html
html = '''
<hr hr style="background-color:#fedf46;class="rounded"">
'''
st.sidebar.markdown(html, unsafe_allow_html=True)

html = '''
<p style="text-align: center; font-size: 15px">For additional info or enhancements, Kindly send your request to jsa48@mail.aub.edu</p>
'''
st.sidebar.markdown(html, unsafe_allow_html=True)


url1="https://drive.google.com/file/d/1xWXsrzxFjjOfey7mqREMi4FOcZ-5AaTn/view?usp=sharing"
successfinder_logo='https://drive.google.com/uc?export=download&id='+url1.split('/')[-2]

col1, col2 = st.beta_columns([4,1])
#with col1:
#    st.image(msba_logo)
with col1:
    # st.text("")
    st.text("")
    html ="<h1 style='text-align: center;border: 3px solid #fedf46;font-size: 32px'>EdTech Initiatives Filtering & Scoring Tool</h1>"
    st.markdown(html, unsafe_allow_html=True)

#background-color:#fedf46;
#F0F2F6 --> greybackground-color:#fedf46;
#53565A

with col2:
    st.image(successfinder_logo)

@st.cache
def get_table_download_link_csv(df):
     csv = df.to_csv(index=False)
     b64 = base64.b64encode(csv.encode()).decode()
     href = f'<a href="data:file/csv;base64,{b64}" download="download.csv">Download csv file</a>'
     return href


################################## Home ###########################

if option =="Home":
    html = '''
    <h1 style="font-size: 24px;background-color:#fedf46;text-align: center">Home</h1>
    '''
    st.markdown(html, unsafe_allow_html=True)


    st.write("")
    st.write("")

    col1, col2, col3 = st.beta_columns([1,7,1])
    with col2:
        html="""<h3 style="font-size:20px;line-height:1.5em;">&nbsp;&nbsp;&nbsp;&nbsp;Amidst the coronavirus pandemic, EdTech industry gained an exponential market growth in the world, and it became of interest for many companies
        worldwide that would like to invest in this field. This webapp is designed to dynamically filter, compare and score all educational technology companies that were scraped from LinkedIn
        and recommend the ones that could be of interest to <span style="border: 3px solid #fedf46;">“SuccessFinder”</span> as potential start-ups or scale-ups.</h3>"""
        st.markdown(html, unsafe_allow_html=True)

    st.write("")
    st.write("")

    url = 'https://drive.google.com/file/d/13s744P2WnAa8qh8RwRGJOKoSj_pP3Yjm/view?usp=sharing'
    home_image = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]

    col1, col2, col3 = st.beta_columns([1,5,1])
    with col2:
        st.image(home_image, use_column_width='auto')

    st.write("")
    # st.write("")

    html1 = '''
    <h3 style="text-align: center; font-size: 20px">This web app aims to <span style="color: #fedf46;">ANALYZE</span> and <span style="color: #fedf46;">SCORE</span> the best <span style="color: #fedf46;">EdTech</span>
    initiatives worldwide.</h3>
    '''
    html2 = '''
    <h3 style="text-align: center; font-size: 20px">Kindly expand the sidebar to explore the application.</h3>
    '''
    col1, col2, col3 = st.beta_columns([1,7,1])
    with col2:
        st.markdown(html1, unsafe_allow_html=True)
        st.markdown(html2, unsafe_allow_html=True)

    html = '''
    <hr style="background-color:#fedf46;class="rounded"">
    '''
    st.markdown(html, unsafe_allow_html=True)
################################## Data Exploration ###########################
if option == "Data Exploration":
    html = '''
    <h1 style="font-size: 24px;background-color:#fedf46;text-align: center">Data Exploration</h1>
    '''
    st.markdown(html, unsafe_allow_html=True)

    st.markdown(f"""The dataset is composed of **{final_df.shape[0]}** companies with the following info: {list(final_df.columns)[0]}, {list(final_df.columns)[1]},
    {list(final_df.columns)[2]}, {list(final_df.columns)[3]}, {list(final_df.columns)[4]}, {list(final_df.columns)[5]}, {list(final_df.columns)[6]}, {list(final_df.columns)[7]},
    {list(final_df.columns)[8]}, {list(final_df.columns)[9]}, {list(final_df.columns)[10]}.""")

    col1, col2 = st.beta_columns([1,1])
    with col1:
        x=st.number_input("Select Number of Rows to Show",min_value=final_df.first_valid_index()+1, max_value=final_df.last_valid_index()+1,value=15)
    with col2:
        column_list=final_df.columns.tolist()
        columns_to_aggregate=st.multiselect('Select the Columns to Show',options=column_list,default=['Company','About','Website','Industry','Headquarters'] )
    df1=final_df.head(x)[columns_to_aggregate]
    st.write(df1)
    button =st.button("Download the Full Dataset")
    if button:
        st.markdown(get_table_download_link_csv(final_df), unsafe_allow_html=True)

    html = '''
    <h1 style="font-size: 16px">Click On the Buttons Below for More Insights</h1>
    '''
    st.markdown(html, unsafe_allow_html=True)
    st.write("")

    col2, col3, col4 = st.beta_columns([1,1,1])

    with col2:
        button1=st.button('Show Industry Statistics')
    with col3:
        button2=st.button('Show Company Size Statistics')
    with col4:
        button3=st.button('Show World Distribution')


    if button1:
        st.write("")
        item_counts = final_df["Industry"].value_counts()
        col1,col2,col3=st.beta_columns([1,5,1])
        with col2:
            st.warning(f'**The histogram below shows that the top industry is {item_counts.index[0]} appearing {item_counts[0]} times followed by {item_counts.index[1]} appearing {item_counts[1]} times.**')
        fig1 = px.bar(final_df.Industry.value_counts(),labels={"index":"Industry", "value":"Count"}) #,title='Industry Distribution', labels={"index":"Industry", "value":"Count"})
        fig1.update_traces(marker_color='#fedf46')
        fig1.update_layout(width=1300,height=600,showlegend=False)
        st.plotly_chart(fig1,width=1100,height=600)


    if button2:
        st.write("")
        item_counts_1 = final_df["Company size"].value_counts()
        col1,col2,col3=st.beta_columns([1,5,1])
        with col2:
            st.warning(f'**The histogram below shows {item_counts_1[0]} companies having {item_counts_1.index[0]}  followed by {item_counts_1[1]} companies with {item_counts_1.index[1]}.**')
        fig2 = px.bar(final_df['Company size'].value_counts() ,labels={"index":"Company Size", "value":"Count"})#title='Company Size',
        fig2.update_traces(marker_color='#fedf46')
        fig2.update_layout(width=1300,height=600,showlegend=False)
        st.plotly_chart(fig2,width=1100,height=600)

    if button3:
        st.write("")
        col1,col2,col3=st.beta_columns([1,5,1])
        with col2:
            st.warning("**The map below shows that most companies are spread across North America and Europe.**")
        fig3 = px.scatter_geo(final_df, lon="longitude",lat="latitude",hover_name="Company", hover_data=["Headquarters", "Industry"],projection="natural earth")#,color="Score"
        fig3.update_traces(marker_color='#fedf46')
        fig3.update_layout(width=1300,height=600)
        st.plotly_chart(fig3)


    html = '''
    <hr style="background-color:#fedf46;class="rounded"">
    '''
    st.markdown(html, unsafe_allow_html=True)


###########################  FILTERING  #######################################

if option == "Data Filtering":
    html = '''
    <h1 style="font-size: 24px;background-color:#fedf46;text-align: center">Data Filtering</h1>
    '''
    st.markdown(html, unsafe_allow_html=True)

    mask1=mask2=mask3=mask4=True

    col1, col2, col3 , col4 = st.beta_columns([1,1,1,1])
    industries = final_df["Industry"] .unique()
    with col1:
        html = '''
        <h1 style="font-size: 16px">Select The Industry:</h1>
        '''
        st.markdown(html, unsafe_allow_html=True)
        industry_choice = st.selectbox("", industries)
        mask1 = final_df["Industry"] == industry_choice

# company_size = final_df["Company size"].loc[final_df["Industries"] == industry_choice].unique()
    company_size = final_df["Company size"].loc[(final_df["Industry"] == industry_choice)].unique()
    company_size=list(company_size)
    company_size.insert(0,"All")

    with col2:
        html = '''
        <h1 style="font-size: 16px">Select The Company Size:</h1>
        '''
        st.markdown(html, unsafe_allow_html=True)
        company_size_choice = st.selectbox("", company_size)
        company_size_1=np.delete(company_size,0)
        mask2 = final_df["Company size"] == company_size_choice if company_size_choice in company_size_1 else True

    type = final_df["Type"].loc[(final_df["Industry"] == industry_choice) & (final_df["Company size"] == company_size_choice)].unique()
    type=list(type)
    type.insert(0,"All")

    with col3:
        if company_size_choice != 'All':
            html = '''
            <h1 style="font-size: 16px">Select The Company Type:</h1>
            '''
            st.markdown(html, unsafe_allow_html=True)
            type_choice = st.selectbox("", type)
            type_1=np.delete(type,0)
            mask3 = final_df["Type"] == type_choice if type_choice in type_1 else True
            year = final_df["Founded"].loc[(final_df["Industry"] == industry_choice) & (final_df["Company size"] == company_size_choice) & (final_df["Type"] == type_choice)].unique()
            year=list(year)
            year.insert(0,"All")
            with col4:
                if type_choice != 'All':
                    html = '''
                    <h1 style="font-size: 16px">Select The Year Founded:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                    year_choice = st.selectbox("", year)
                    if year_choice != "All":
                        mask4 = final_df["Founded"] == year_choice
                    else:
                        mask4 = True


    df_filtered_1 = (final_df[mask1 & mask2 & mask3 & mask4])
    st.dataframe(df_filtered_1)

    col1, col2,col3 = st.beta_columns([2,1,2])
    with col1:
        html = '''
        <h1 style="font-size: 16px">Table Exploration:</h1>
        '''
        st.markdown(html, unsafe_allow_html=True)
        box= st.selectbox('', ['Check 1 Company','Compare 2 Companies'])
# button1=st.button("Click To Explore 1 Company")
    if box=='Check 1 Company':
        col1, col2,col3 = st.beta_columns([2,1,2])
        with col1:
            html = '''
            <h1 style="font-size: 16px">Enter the Company Row Number:</h1>
            '''
            st.markdown(html, unsafe_allow_html=True)
            x=st.number_input('',value=df_filtered_1.first_valid_index(), step=1)
            # x=df_filtered_1.first_valid_index()
        if x in df_filtered_1.index:
            html = '''
            <h1 style="text-decoration: underline;font-size: 14px">Company Name:</h1>
            '''
            st.markdown(html, unsafe_allow_html=True)
        # st.write("Company Name:")
            st.write(df_filtered_1['Company'].loc[x])
            html = '''
            <h1 style="text-decoration: underline;font-size: 14px">About:</h1>
            '''
            st.markdown(html, unsafe_allow_html=True)
        # st.write("About:")
            st.write(df_filtered_1['About'].loc[x])
            html = '''
            <h1 style="text-decoration: underline;font-size: 14px">Website:</h1>
            '''
            st.markdown(html, unsafe_allow_html=True)
        # st.write("Website:")
            st.write(df_filtered_1['Website'].loc[x])
            html = '''
            <h1 style="text-decoration: underline;font-size: 14px">Industry:</h1>
            '''
            st.markdown(html, unsafe_allow_html=True)
        # st.write("Industries:")
            st.write(df_filtered_1['Industry'].loc[x])
            html = '''
            <h1 style="text-decoration: underline;font-size: 14px">Company Size:</h1>
            '''
            st.markdown(html, unsafe_allow_html=True)
        # st.write("Company Size:")
            st.write(df_filtered_1['Company size'].loc[x])
            html = '''
            <h1 style="text-decoration: underline;font-size: 14px">Headquarter:</h1>
            '''
            st.markdown(html, unsafe_allow_html=True)
        # st.write("Headquarter:")
            st.write(df_filtered_1['Headquarters'].loc[x])
            html = '''
            <h1 style="text-decoration: underline;font-size: 14px">Type:</h1>
            '''
            st.markdown(html, unsafe_allow_html=True)
        # st.write("Type:")
            st.write(df_filtered_1['Type'].loc[x])
            html = '''
            <h1 style="text-decoration: underline;font-size: 14px">Year Founded:</h1>
            '''
            st.markdown(html, unsafe_allow_html=True)
        # st.write("Year Founded:")
            st.write(df_filtered_1['Founded'].loc[x])
            html = '''
            <h1 style="text-decoration: underline;font-size: 14px">Specialities:</h1>
            '''
            st.markdown(html, unsafe_allow_html=True)
        # st.write("Specialities:")
            st.write(df_filtered_1['Specialties'].loc[x])
        else:
            st.write("Insert a number from the available indices")

# button2=st.button("Click To Compare 2 Companies")
    if box=='Compare 2 Companies':
        if df_filtered_1.shape[0]>1:
            col1, col2,col3 = st.beta_columns([2,1,2])
            with col1:
                html = '''
                <h1 style="font-size: 16px">Enter the Row Number of The 1st Company:</h1>
                '''
                st.markdown(html, unsafe_allow_html=True)
                x=st.number_input('',value=df_filtered_1.first_valid_index(), step=1)
                if  x in df_filtered_1.index:
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">Company Name:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("Company Name:")
                    st.write(df_filtered_1['Company'].loc[x])
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">About:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("About:")
                    st.write(df_filtered_1['About'].loc[x])
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">Website:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("Website:")
                    st.write(df_filtered_1['Website'].loc[x])
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">Industry:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("Industries:")
                    st.write(df_filtered_1['Industry'].loc[x])
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">Company Size:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("Company Size:")
                    st.write(df_filtered_1['Company size'].loc[x])
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">Headquarter:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("Headquarter:")
                    st.write(df_filtered_1['Headquarters'].loc[x])
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">Type:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("Type:")
                    st.write(df_filtered_1['Type'].loc[x])
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">Year Founded:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("Year Founded:")
                    st.write(df_filtered_1['Founded'].loc[x])
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">Specialities:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("Specialities:")
                    st.write(df_filtered_1['Specialties'].loc[x])
                else:
                    st.write("Insert a number from the available indices")

            with col3:
                html = '''
                <h1 style="font-size: 16px">Enter the Row Number of The 2nd Company:</h1>
                '''
                st.markdown(html, unsafe_allow_html=True)
                y=st.number_input('',value=df_filtered_1.last_valid_index(), step=1)
                if y in df_filtered_1.index:
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">Company Name:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("Company Name:")
                    st.write(df_filtered_1['Company'].loc[y])
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">About:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("About:")
                    st.write(df_filtered_1['About'].loc[y])
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">Website:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("Website:")
                    st.write(df_filtered_1['Website'].loc[y])
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">Industry:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("Industries:")
                    st.write(df_filtered_1['Industry'].loc[y])
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">Company Size:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("Company Size:")
                    st.write(df_filtered_1['Company size'].loc[y])
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">Headquarter:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("Headquarter:")
                    st.write(df_filtered_1['Headquarters'].loc[y])
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">Type:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("Type:")
                    st.write(df_filtered_1['Type'].loc[y])
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">Year Founded:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("Year Founded:")
                    st.write(df_filtered_1['Founded'].loc[y])
                    html = '''
                    <h1 style="text-decoration: underline;font-size: 14px">Specialities:</h1>
                    '''
                    st.markdown(html, unsafe_allow_html=True)
                # st.write("Specialities:")
                    st.write(df_filtered_1['Specialties'].loc[y])
                else:
                    st.write("Insert a number from the available indices")

    html = '''
    <hr style="background-color:#fedf46;class="rounded"">
    '''
    st.markdown(html, unsafe_allow_html=True)

###################################### SCORING #################################


if option == "Scoring":
    final_df_1=final_df
    final_df_1['Score']=0
    html = '''
    <h1 style="font-size: 24px;background-color:#fedf46;text-align: center">Scoring</h1>
    '''
    st.markdown(html, unsafe_allow_html=True)
# st.write('The scoring tool is based on four factors: Industry, Company Size, Company Type & Year Founded')
    html = '''
    <h1 style="font-size: 20px">The Scoring Tool is Based on <span style="color: #fedf46;">FOUR</span> Factors: <span style="color: #fedf46;">Industry</span>, <span style="color: #fedf46;">Company Size</span>, <span style="color: #fedf46;">Company Type</span> and <span style="color: #fedf46;">Year Founded</span>.</h1>
    '''
    st.markdown(html, unsafe_allow_html=True)

#Industry
    html = '''<h2 style="text-decoration: underline;font-size: 20px">Industry</h2>'''
    st.markdown(html, unsafe_allow_html=True)
    st.warning("ℹ️ Please select the industries that you would like to score. Then specify the score in the corresponding box. This score will be added to all the corresponding companies. Once done select the Apply box.")
    industry1=industry2=industry3=industry4=industry5=industry6=industry7=industry8=""
    industry_select = st.multiselect('Select the Industries that you wish to score', final_df_1["Industry"].unique().tolist())
# st.write(len(industry_select))
    if len(industry_select)<5:
        col1, col2,col3,col4 = st.beta_columns([1,1,1,1])
        with col1:
            if len(industry_select)-1>=0:
                industry1=industry_select[0]
                n=st.number_input(f"Score for {industry1}",value =0)

        with col2:
            x= len(industry_select)-1
            if x-1>=0:
                industry2=industry_select[1]
                o=st.number_input(f"Score for {industry2}",value =0)

        with col3:
            y= len(industry_select)-2
            if y-1>=0:
                industry3=industry_select[2]
                p=st.number_input(f"Score for {industry3}",value =0)

        with col4:
            z= len(industry_select)-3
            if z-1>=0:
                industry4=industry_select[3]
                q=st.number_input(f"Score for {industry4}",value =0)

    elif len(industry_select)<9:
        col1, col2,col3,col4 = st.beta_columns([1,1,1,1])
        with col1:
            if len(industry_select)-1>=0:
                industry1=industry_select[0]
                n=st.number_input(f"Score for {industry1}",value =0)

        with col2:
            if len(industry_select)-2>=0:
                industry2=industry_select[1]
                o=st.number_input(f"Score for {industry2}",value =0)

        with col3:
            if len(industry_select)-3>=0:
                industry3=industry_select[2]
                p=st.number_input(f"Score for {industry3}",value =0)

        with col4:
            if len(industry_select)-4>=0:
                industry4=industry_select[3]
                q=st.number_input(f"Score for {industry4}",value =0)

        col1, col2,col3,col4 = st.beta_columns([1,1,1,1])
        with col1:
            if len(industry_select)-5>=0:
                industry5=industry_select[4]
                r=st.number_input(f"Score for {industry5}",value =0)

        with col2:
            if len(industry_select)-6>=0:
                industry6=industry_select[5]
                s=st.number_input(f"Score for {industry6}",value =0)

        with col3:
            if len(industry_select)-7>=0:
                industry7=industry_select[6]
                t=st.number_input(f"Score for {industry7}",value =0)

        with col4:
            if len(industry_select)-8>=0:
                industry8=industry_select[7]
                u=st.number_input(f"Score for {industry8}",value =0)

    else:
        st.write("Choose 8 or less industries to score")

    button4=st.checkbox("Apply \"Industry\" Scores")
    if button4:
        for i in range (final_df_1.shape[0]):
            if final_df_1['Industry'][i] == industry1:
                final_df_1['Score'][i]+=n
            elif final_df_1['Industry'][i] == industry2:
                final_df_1['Score'][i]+=o
            elif final_df_1['Industry'][i] == industry3:
                final_df_1['Score'][i]+=p
            elif final_df_1['Industry'][i] == industry4:
                final_df_1['Score'][i]+=q
            elif final_df_1['Industry'][i] == industry5:
                final_df_1['Score'][i]+=r
            elif final_df_1['Industry'][i] == industry6:
                final_df_1['Score'][i]+=s
            elif final_df_1['Industry'][i] == industry7:
                final_df_1['Score'][i]+=t
            elif final_df_1['Industry'][i] == industry8:
                final_df_1['Score'][i]+=u



#Company Size
    html = '''<h2 style="text-decoration: underline;font-size: 20px">Company Size</h2>'''
    st.markdown(html, unsafe_allow_html=True)
    st.warning("ℹ️ Please select the score that you would like to assign to each company size. This score will be added to all the corresponding companies. Once done select the Apply box.")
    col1, col2,col3,col4 = st.beta_columns([1,1,1,1])
    with col1:
        size1=final_df_1["Company size"].value_counts().index[0]
        a=st.number_input(f"Score for {size1}",value =0)

    with col2:
        size2=final_df_1["Company size"].value_counts().index[1]
        b=st.number_input(f"Score for {size2}",value =0)

    with col3:
        size3=final_df_1["Company size"].value_counts().index[2]
        c=st.number_input(f"Score for {size3}",value =0)

    with col4:
        size4=final_df_1["Company size"].value_counts().index[3]
        d=st.number_input(f"Score for {size4}",value =0)


    col1, col2,col3,col4 = st.beta_columns([1,1,1,1])
    with col1:
        size5=final_df_1["Company size"].value_counts().index[4]
        e=st.number_input(f"Score for {size5}",value =0)

    with col2:
        size6=final_df_1["Company size"].value_counts().index[5]
        f=st.number_input(f"Score for {size6}",value =0)

    with col3:
        size7=final_df_1["Company size"].value_counts().index[6]
        g=st.number_input(f"Score for {size7}",value =0)

    with col4:
        size8=final_df_1["Company size"].value_counts().index[7]
        h=st.number_input(f"Score for {size8}",value =0)

    button2=st.checkbox("Apply \"Company Size\" Scores")
    if button2:
        for i in range (final_df_1.shape[0]):
            if final_df_1['Company size'][i] == size1:
                final_df_1['Score'][i]+=a
            elif final_df_1['Company size'][i] == size2:
                final_df_1['Score'][i]+=b
            elif final_df_1['Company size'][i] == size3:
                final_df_1['Score'][i]+=c
            elif final_df_1['Company size'][i] == size4:
                final_df_1['Score'][i]+=d
            elif final_df_1['Company size'][i] == size5:
                final_df_1['Score'][i]+=e
            elif final_df_1['Company size'][i] == size6:
                final_df_1['Score'][i]+=f
            elif final_df_1['Company size'][i] == size7:
                final_df_1['Score'][i]+=g
            elif final_df_1['Company size'][i] == size8:
                final_df_1['Score'][i]+=h

#Company Type
    html = '''<h2 style="text-decoration: underline;font-size: 20px">Company Type</h2>'''
    st.markdown(html, unsafe_allow_html=True)
    st.warning("ℹ️ Please select the score that you would like to assign to each type of companies. This score will be added to all the corresponding companies. Once done select the Apply box.")
    col1, col2,col3,col4 = st.beta_columns([1,1,1,1])
    with col1:
        type1=final_df_1["Type"].value_counts().index[0]
        j=st.number_input(f"Score for {type1}",value =0)

    with col2:
        type2=final_df_1["Type"].value_counts().index[1]
        k=st.number_input(f"Score for {type2}",value =0)

    with col3:
        type3=final_df_1["Type"].value_counts().index[2]
        l=st.number_input(f"Score for {type3}",value =0)

    with col4:
        type4=final_df_1["Type"].value_counts().index[3]
        m=st.number_input(f"Score for {type4}",value =0)


    col1, col2,col3,col4 = st.beta_columns([1,1,1,1])
    with col1:
        type5=final_df_1["Type"].value_counts().index[4]
        n=st.number_input(f"Score for {type5}",value =0)

    with col2:
        type6=final_df_1["Type"].value_counts().index[5]
        o=st.number_input(f"Score for {type6}",value =0)

    with col3:
        type7=final_df_1["Type"].value_counts().index[6]
        p=st.number_input(f"Score for {type7}",value =0)

    with col4:
        type8=final_df_1["Type"].value_counts().index[7]
        q=st.number_input(f"Score for {type8}",value =0)

    button3=st.checkbox("Apply \"Company Type\" Scores")
    if button3:
        for i in range (final_df_1.shape[0]):
            if final_df_1['Type'][i] == type1:
                final_df_1['Score'][i]+=j
            elif final_df_1['Type'][i] == type2:
                final_df_1['Score'][i]+=k
            elif final_df_1['Type'][i] == type3:
                final_df_1['Score'][i]+=l
            elif final_df_1['Type'][i] == type4:
                final_df_1['Score'][i]+=m
            elif final_df_1['Type'][i] == type5:
                final_df_1['Score'][i]+=n
            elif final_df_1['Type'][i] == type6:
                final_df_1['Score'][i]+=o
            elif final_df_1['Type'][i] == type7:
                final_df_1['Score'][i]+=p
            elif final_df_1['Type'][i] == type8:
                final_df_1['Score'][i]+=q

#Year Founded
    html = '''<h2 style="text-decoration: underline;font-size: 20px">Year Founded</h2>'''
    st.markdown(html, unsafe_allow_html=True)
    st.warning("ℹ️ The score will be added to all the companies founded on or after the selected year.")
    col1,col2 = st.beta_columns([1,1])
    with col1:
        l=final_df_1["Founded"].unique().tolist()
        l.sort()
        x=st.selectbox("Select The Year",options=l[1:])

    with col2:
        y=st.number_input("Assign The Score",value=0)

    button1=st.checkbox("Apply \"Year Founded\" Scores")
    if button1:
        for i in range (final_df_1.shape[0]):
            if final_df_1['Founded'][i] >=x:
                final_df_1['Score'][i]+=y

#Show Dataset with scores
    final_df_2=final_df_1[['Company','Industry','Company size','Type','Score']]

    @st.cache
    def showfig(n):
        fig = px.bar(final_df_2.nlargest(n,"Score"), x="Score", y="Company",orientation='h',title=f'Top {n} Scoring Companies')
        fig.update_traces(marker_color='#fedf46')
        fig.update_layout(width=1300,height=600)
        return fig

    st.write("")
    st.write("")
    col1,col2=st.beta_columns([1,1])

    with col1:
        check=st.checkbox("Show Scored Dataset")

    with col2:
        barplot=st.checkbox("Show Top Companies")

    if check:
        st.write(final_df_2)



    if barplot:
        n=st.slider('Slide to select the number of top companies to display', min_value=2, max_value=25,value=5)

    # df_perf=df_year.groupby(['Store'],as_index=False)['Sales'].sum() #group by store and retreiving the sum of sales of each
        # fig1 = px.bar(final_df_2.nlargest(n,"Score"), x="Score", y="Company",orientation='h',title=f'Top {n} Scoring Companies')
        # fig1.update_traces(marker_color='#fedf46')
        # fig1.update_layout(width=1300,height=600)
        fig1=showfig(n)
        st.plotly_chart(fig1,width=1300,height=600)

    html = '''
    <hr style="background-color:#fedf46;class="rounded"">
    '''
    st.markdown(html, unsafe_allow_html=True)

############################### Data Analysis using NLP #######################
if option == 'Data Analysis using NLP':
    html = '''
    <h1 style="font-size: 24px;background-color:#fedf46;text-align: center">Data Analysis using NLP</h1>
    '''
    st.markdown(html, unsafe_allow_html=True)

    st.write("")

    html = '''
    <h1 style="font-size: 20px">In this section, we will use NLP to delve a little deeper into the <span style="color: #fedf46;">"About"</span> column of each company.</h1>
    '''
    st.markdown(html, unsafe_allow_html=True)

    st.write("")

    final_df_3=final_df

    ##Creating a list of stop words
    stop_words = set(stopwords.words("english"))

    corpus = []
    for i in range (final_df_3.shape[0]):
        #Remove punctuations
        text = re.sub('[^a-zA-Z]', ' ', final_df_3['About'][i])

        #Convert to lowercase
        text = text.lower()

        #remove tags
        text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

        # remove special characters and digits
        text=re.sub("(\\d|\\W)+"," ",text)

        ##Convert to list from string
        text = text.split()

        ##Stemming
        ps=PorterStemmer()
        #Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in stop_words]
        text = " ".join(text)
        corpus.append(text)

    final_df_3['Cleaned_About'] = corpus

    @st.cache
    def most_occuring_word(i,k):
        from collections import Counter
        # split() returns list of all the words in the string
        split_it = final_df_3['Cleaned_About'][i].split()

        # Pass the split_it list to instance of Counter class.
        Counter = Counter(split_it)

        # most_common() produces k frequently encountered
        # input values and their respective counts.
        most_occur = Counter.most_common(k)
        return most_occur

    @st.cache
    def createdataframe(x,n):
        df=pd.DataFrame(most_occuring_word(x,n),columns=('Word','Number of Occurence'))
        return df

    @st.cache
    def createbarchart(df):
        fig=px.bar(df,y='Word',x='Number of Occurence',orientation='h')#,title=f'Top {n} most occuring words for {company}.')
        fig.update_traces(marker_color='#fedf46')
        return(fig)
        # st.plotly_chart(fig3)

    st.write("""We will start by cleaning the columns from stopwords, digits,
     punctuations and special characters. Also we will be lemmatizing each word by converting it to its original roots. Then we will be display the related number of words.""")
    st.write("Please input the number of comapnies to show.")
    col1,col2,col3,col4=st.beta_columns([1,1,1,1])
    with col1:
        n=st.number_input(" ",min_value=1,max_value=final_df_3.shape[0],value=5,step=1)
    final_df_3['Word_Count'] = final_df_3['Cleaned_About'].apply(lambda x: len(str(x).split(" ")))
    final_df_3=final_df_3[['Company','About','Cleaned_About','Word_Count']]
    st.write(final_df_3.head(n))


    st.write("")
    st.write("")
    col1,col2,col3,col4=st.beta_columns([1,2,2,1])

    with col2:
        check2=st.checkbox("Most Occuring Word")
        #Fetch wordcount for each "about"
    with col3:
        check3=st.checkbox("Keyword Lookup")
        #Fetch wordcount for each "about"

    st.write("")
    st.write("")

    if check2:
        col1,col2=st.beta_columns([1,1])
        with col1:
            # st.write("Choose the company row number:")
            # x=st.number_input(" ",min_value=0,max_value=final_df_3.shape[0],value=0, step=1)
            # company=final_df_3['Company'][x]
            st.write("Choose the company:")
            comp=st.selectbox(" ",final_df_3['Company'].unique())
            x=final_df_3[final_df_3['Company']==comp].index[0]
        with col2:
            st.write("Slide to choose the top \"n\" words to display")
            n=st.slider(" ",min_value=2,max_value=20,step=1,value=5)#max_value=int(final_df_3["Word_Count"][x])
        # st.dataframe(most_occuring_word(x,y))


        # df=pd.DataFrame(most_occuring_word(x,n),columns=('Word','Number of Occurence'))

        html= f"""<h2 style="text-align: center; font-size: 18px">Top <span style="color: #fedf46;">{n}</span> most occuring words for <span style="color: #fedf46;">{comp}</span></h2>"""
        st.markdown(html,unsafe_allow_html=True)
        col1,col2 = st.beta_columns([1,2])
        with col1:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            df=createdataframe(x,n)
            st.write(df)
        with col2:
            fig3=createbarchart(df)
            # fig3=px.bar(df,y='Word',x='Number of Occurence',orientation='h')#,title=f'Top {n} most occuring words for {company}.')
            # fig3.update_traces(marker_color='#fedf46')
            st.plotly_chart(fig3)

    if check3:
        col1,col2=st.beta_columns([1,1])
        with col1:
            n=st.slider("Choose the number of keywords you want to search for", min_value=1, max_value=4,step=1)
        with col2:
            option=st.selectbox('Select the search option', ["AND","OR"])
        col1,col2,col3,col4=st.beta_columns([1,1,1,1])
        if n ==1:
            with col1:
                s1=st.text_input('Keyword 1')
            index_list=[]
            for i in range (final_df_3.shape[0]):
                if s1 in final_df_3['Cleaned_About'][i]:
                    index_list.append(i)
            st.write(f'The number of companies is {len(index_list)}.')
            st.write(final_df_3.iloc[index_list])

        if n ==2 and option == "AND":
            with col1:
                s1=st.text_input('Keyword 1')
            with col2:
                s2=st.text_input('Keyword 2')
            index_list=[]
            for i in range (final_df_3.shape[0]):
                if (s1 in final_df_3['Cleaned_About'][i] and s2 in final_df_3['Cleaned_About'][i]) :
                    index_list.append(i)
            st.write(f'The number of companies is {len(index_list)}.')
            st.write(final_df_3.iloc[index_list])

        if n ==2 and option == "OR":
            with col1:
                s1=st.text_input('Keyword 1')
            with col2:
                s2=st.text_input('Keyword 2')
            index_list=[]
            for i in range (final_df_3.shape[0]):
                if (s1 in final_df_3['Cleaned_About'][i] or s2 in final_df_3['Cleaned_About'][i]) :
                    index_list.append(i)
            st.write(f'The number of companies is {len(index_list)}.')
            st.write(final_df_3.iloc[index_list])


        if n ==3 and option == "AND":
            with col1:
                s1=st.text_input('Keyword 1')
            with col2:
                s2=st.text_input('Keyword 2')
            with col3:
                s3=st.text_input('Keyword 3')
            index_list=[]
            for i in range (final_df_3.shape[0]):
                if (s1 in final_df_3['Cleaned_About'][i] and s2 in final_df_3['Cleaned_About'][i] and s3 in final_df_3['Cleaned_About'][i]) :
                    index_list.append(i)
            st.write(f'The number of companies is {len(index_list)}.')
            st.write(final_df_3.iloc[index_list])

        if n ==3 and option == "OR":
            with col1:
                s1=st.text_input('Keyword 1')
            with col2:
                s2=st.text_input('Keyword 2')
            with col3:
                s3=st.text_input('Keyword 3')
            index_list=[]
            for i in range (final_df_3.shape[0]):
                if (s1 in final_df_3['Cleaned_About'][i] or s2 in final_df_3['Cleaned_About'][i] or s3 in final_df_3['Cleaned_About'][i]) :
                    index_list.append(i)
            st.write(f'The number of companies is {len(index_list)}.')
            st.write(final_df_3.iloc[index_list])


        if n ==4 and option == "AND":
            with col1:
                s1=st.text_input('Keyword 1')
            with col2:
                s2=st.text_input('Keyword 2')
            with col3:
                s3=st.text_input('Keyword 3')
            with col4:
                s4=st.text_input('Keyword 4')
            index_list=[]
            for i in range (final_df_3.shape[0]):
                if (s1 in final_df_3['Cleaned_About'][i] and s2 in final_df_3['Cleaned_About'][i] and s3 in final_df_3['Cleaned_About'][i] and s4 in final_df_3['Cleaned_About'][i]) :
                    index_list.append(i)
            st.write(f'The number of companies is {len(index_list)}.')
            st.write(final_df_3.iloc[index_list])

        if n ==4 and option == "OR":
            with col1:
                s1=st.text_input('Keyword 1')
            with col2:
                s2=st.text_input('Keyword 2')
            with col3:
                s3=st.text_input('Keyword 3')
            with col4:
                s4=st.text_input('Keyword 4')
            index_list=[]
            for i in range (final_df_3.shape[0]):
                if (s1 in final_df_3['Cleaned_About'][i] or s2 in final_df_3['Cleaned_About'][i] or s3 in final_df_3['Cleaned_About'][i] or s4 in final_df_3['Cleaned_About'][i]) :
                    index_list.append(i)
            st.write(f'The number of companies is {len(index_list)}.')
            st.write(final_df_3.iloc[index_list])
