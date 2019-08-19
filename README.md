# SQAIR-pytorch
PyTorch implementation of Sequential Attend Infer Repeat. I made several modifications to the original model so it is simpler and easy to understand.

The implementation still needs to be improved. In particular, I did not implement the glimpse extraction step in propagation, because I am not able to get it work currently. And that causes ID swap as reported in the paper. Anyway, the model works to some extent. Some results on sequence of length 5:

![f2](../../../../../Desktop/f2.png)

We can see that ID swap happens very frequently. I will present more results when I have addressed this issue.