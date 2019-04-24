
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt


# In[2]:


word2vec_df = pd.read_csv('word2vec.csv', header=None)
word2vec_df.head()


# In[3]:


# mean imputation
mean_vec = np.mean(word2vec_df.dropna(axis=0).values, axis=0)
word2vec_df.iloc[32618, :] = mean_vec
word2vec_df.iloc[87646, :] = mean_vec

# C-style contiguous
word2vec_arr = np.ascontiguousarray(word2vec_df.values.astype(np.float32))

# l2 normalization
faiss.normalize_L2(word2vec_arr)


# In[4]:


def random_pairs(X, n_sample):
    # sample index pairs
    random_id = np.random.randint(low=0, high=len(X), size=(n_sample, 2))
    return random_id

def cosine_similarity(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


# In[5]:


# dinstance distribution
pairs = random_pairs(word2vec_arr, 5000)
sim_sample = [cosine_similarity(word2vec_arr[id1, :], word2vec_arr[id2, :]) for (id1, id2) in pairs]
plt.hist(sim_sample, bins=int(np.sqrt(len(sim_sample))))
plt.show()


# In[6]:


index = faiss.IndexFlatIP(word2vec_arr.shape[1])
print(index.is_trained)
index.add(word2vec_arr)
print(index.ntotal)


# In[7]:


k = 20
xq = word2vec_arr
D, I = index.search(xq, k)


# In[8]:


print(I[:5])
print('...')
print(I[-5:])


# In[9]:


print(D[:5])
print('...')
print(D[-5:])


# In[10]:


# filtered distance distribution
plt.hist(D[:, 1:].ravel(), bins=int(np.sqrt(len(D[:, 1:].ravel()))))
plt.show()


# In[11]:


plt.hist(sim_sample, bins=int(np.sqrt(len(sim_sample))), density=True)
plt.hist(D[:, 1:].ravel(), bins=int(np.sqrt(len(D[:, 1:].ravel()))), density=True)
plt.show()

