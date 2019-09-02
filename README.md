# skip-gram
A simple implementation of Skip-Gram model in PyTorch.
## Files organization
main.py —— the training process

model.py —— model's definition

getData.py —— pre-processing and organizing data(import torch.utils.data.DataLoader to enable batch)

text8、simtext2 —— the data files, "simtext2" is smaller. 

If you encounter the problem _"RuntimeWarning: divide by zero encountered in true_divide
  sampling_p = (np.sqrt(fre_np / 0.001) + 1) * 0.001 / fre_np"_, you should probably consider decreasing the value of vacabulary_size(for example 1000), because you may be using smaller dataset. 
## Results

The results of english text are as follow, the chinese word vectors are still be training. 

| task        |   this repo   |  [**CCL2017 paper**](https://link.springer.com/chapter/10.1007/978-3-319-69005-6_4)  |
| :---:   | :-----:  | :----:  |
|   word relatedness    |  69.88%  |   69.36%     |
|  syntactic question   |   16.84%   |   54.24%   |
| semantic question     |        |  15.59%  |

## References
[**Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems. 2013.**](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

[**Li, Fang and Xiaojie Wang. “Improving Word Embeddings for Low Frequency Words by Pseudo Contexts.” CCL (2017).**](https://link.springer.com/chapter/10.1007/978-3-319-69005-6_4)