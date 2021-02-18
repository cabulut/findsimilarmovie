#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, colorConverter


# In[4]:


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
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    title = st.selectbox('Movie',(final["title"]))
    result = ""
    links = ""
    
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Find me Similar Movies"):
        result = cluster(title)
        links = link(title)
        for i in range(len(result)):   
            st.success("[{}]({})".format(result[i],links[i]))

        
if __name__=='__main__': 
    main()


# In[3]:




# In[ ]:




# In[3]:


#final = pd.merge(final,den,on=["title"])
#final = final.sort_values("title").reset_index(drop=True)


# In[31]:


stemmer = SnowballStemmer("english")

# Define a function to perform both stemming and tokenization
def token_and_stem(text):
    
    # Tokenize by sentence, then by word
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    # Filter out raw tokens to remove noise
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    
    # Stem the filtered_tokens
    stems = [stemmer.stem(t) for t in filtered_tokens]
    
    return stems


# In[32]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.99, max_features=200000,
                                 min_df=0.01, stop_words='english',
                                 use_idf=True, tokenizer=token_and_stem,
                                 ngram_range=(1,3))


# In[33]:


tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in final["plot"]])

print(tfidf_matrix.shape)


# In[52]:


similarity_distance = 1 - cosine_similarity(tfidf_matrix)

# Create mergings matrix 
Z = linkage(similarity_distance, method='ward')

# Plot the dendrogram, using title as label column
dendrogram_ = dendrogram(Z,
               labels=[x for x in final["title"]],
               leaf_rotation=90,
               leaf_font_size=15,
)

# Adjust the plot
fig = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(108, 42)

# Show the plotted dendrogram
plt.show()


# In[21]:


#dendrogram_


# from collections import defaultdict
# cluster_idxs = defaultdict(list)
# for c, pi in zip(dendrogram_['color_list'], dendrogram_['icoord']):
#     for leg in pi[1:3]:
#         i = (leg - 5.0) / 10.0
#         if abs(i - int(i)) < 1e-5:
#             cluster_idxs[c].append(int(i))

# In[ ]:


#den = pd.DataFrame(dendrogram_["ivl"],columns=["title"])


# """lst = [1,1,2,2,2,3,3,4,4,4,5,5,6,6,6,7,7,8,8,9,9,9,10,10,11,11,11,12,12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,19,20,20,
#       21,21,21,22,22,22,23,23,24,24,25,25,25,25,26,26,27,27,28,28,28,28,29,29,30,30,30,31,31,31,32,32,33,33,33,33,
#       34,34,34,34,35,35,36,36,37,37,38,38,39,39,39,40,40,41,41,41,42,42,43,43,44,44,45,45,46,46,47,47,48,48,48,49,49,50,50,51,51,52,52,53,53,
#       54,54,54,54,55,55,56,56,57,57,58,58,59,59,59,60,60,60,61,61,62,62,63,63,64,64,64,64,65,65,66,66,67,67,68,68,69,69,69,
#       70,70,71,71,71,72,72,73,73,73,74,74,75,75,75,76,76,77,77,78,78,79,79,79,80,80,81,81,82,82,82,
#       83,83,83,84,84,84,85,85,86,86,86,87,87,88,88,89,89,90,90,90,90,91,91,91,92,92,93,93,93,94,94,95,95,96,96,97,97,
#       98,98,98,99,99,100,100,101,101,102,102,102,103,103,104,104]"""

# In[6]:


#final = pd.merge(final,den,on=["title"])


# In[7]:


#final = final.sort_values("title").reset_index(drop=True)


# In[8]:


#den["segment"] = lst


# In[2]:


#cluster_idxs


# class Clusters(dict):
#     def _repr_html_(self):
#         html = '<table style="border: 0;">'
#         for c in self:
#             hx = rgb2hex(colorConverter.to_rgb(c))
#             html += '<tr style="border: 0;">' \
#             '<td style="background-color: {0}; ' \
#                        'border: 0;">' \
#             '<code style="background-color: {0};">'.format(hx)
#             html += c + '</code></td>'
#             html += '<td style="border: 0"><code>' 
#             html += repr(self[c]) + '</code>'
#             html += '</td></tr>'
# 
#         html += '</table>'
# 
#         return html

# cluster_classes = Clusters()
# for c, l in cluster_idxs.items():
#     i_l = [dendrogram_['ivl'][i] for i in l]
#     cluster_classes[c] = i_l
# 
# cluster_classes
