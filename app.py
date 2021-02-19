#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd


# In[6]:


final = pd.read_csv("cluster.csv",index_col=0)


def cluster(title):
        return final[(final.title!=title) & (final.segment==int(final[final.title==title]["segment"]))]["title"].values
    
def link(title):
        return final[(final.title!=title) & (final.segment==int(final[final.title==title]["segment"]))]["link"].values
    
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:gold;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Similar Movie App</h1> 
    </div> 
    """
    
    page_bg_img = '''
    <style>
    body{
    background-image: url("https://mocah.org/uploads/posts/4533539-pulp-fiction-movies-simple-background-skull-drawing-white-background.jpg");
    background-size: cover;}
    </style>
    '''
      
    # display the front end aspect
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown(html_temp,unsafe_allow_html=True) 
    st.sidebar.markdown("![Alt Text](https://media.giphy.com/media/RlH7GlyiGKJ1ciAbO0/giphy.gif)")
    
    # select boxes for movie title
    title = st.selectbox('Pick Your Movie',(final["title"]))
    result = ""
    links = ""
    
    # click on "Find me Similar Movies" and end up with similar movie and its IMDb link
    if st.button("Find Similar Movie"):
        result = cluster(title)
        links = link(title)
        for i in range(len(result)):   
            st.success("[{}]({})".format(result[i],links[i]))

        
if __name__=='__main__': 
    main()


# In[3]:


# convert streamlit to app.py to run on jupyter notebook


# In[ ]:


# run streamlit app

