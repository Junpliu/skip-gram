# skip-gram
A simple implementation of Skip-Gram model in PyTorch.
## Files organization
main.py —— the training process

model_def.py —— model's definition

getData.py —— pre-processing and organizing data(import torch.utils.data.DataLoader to enable batch)

text8、simtext2 —— the data files, "simtext2" is smaller. 

If you encounter the problem _"RuntimeWarning: divide by zero encountered in true_divide
  sampling_p = (np.sqrt(fre_np / 0.001) + 1) * 0.001 / fre_np"_, you should probably consider decreasing the value of vacabulary_size(for example 1000), because you may be using smaller dataset. 
## Results
Without much hyper-parameter optimization, the loss value of the model decreases to 0.1, you are free to optimize them to lower loss value. 


## Reference paper
**Distributed Representations of Words and Phrases and their Compositionality**

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean
[**NIPS 2015 paper**](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)