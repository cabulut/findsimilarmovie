#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd


# In[2]:


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
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True) 
      
    # select boxes for movie title
    title = st.selectbox('Movie',(final["title"]))
    result = ""
    links = ""
    
    # click on "Find me Similar Movies" and end up with similar movie and its IMDb link
    if st.button("Find me Similar Movies"):
        result = cluster(title)
        links = link(title)
        for i in range(len(result)):   
            st.success("[{}]({})".format(result[i],links[i]))

        
if __name__=='__main__': 
    main()


# In[5]:


# convert streamlit to app.py to run on jupyter notebook


# In[ ]:


# run streamlit app

