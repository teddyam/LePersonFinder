read

attempted fixes for vit: 
- reduce batch size, increase batch size
- test on single image/single json and check overfitting (it does overfit wihtin like 100 epoch)
- play around w learning rate 
- play around w number of dense/dropout layers in various feedforward layers throughout the transformer model
- increase/decrease number of epochs 
- change the optimizer (SGD -> nan loss -> dk why tf), Adam trains the best but its still outputting the exact same predictions across training ex

- overlying issue = prediction of same shit across training ex WHY????
- printed out model weights at each epoch and checked to see if they're the same btwn examples
    - see printline in trian method 

- remaining shit to test: 
    - idfk