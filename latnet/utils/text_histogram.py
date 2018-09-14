
import numpy as np

def vector_to_text_hist(vector, bins=10):
  hist_vector, _ = np.histogram(vector, bins=bins, density=True)
  hist_vector = 100*hist_vector/np.sum(hist_vector)
  string = ''
  
  for i in range(bins):
    string += str(int(hist_vector[i])).zfill(2) + ', '
  return string
  


