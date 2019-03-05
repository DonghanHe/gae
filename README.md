Graph Auto-Encoders For Adajcency matrix and Node feature reconstruction

In this project I used T Kipf's graph auto-encoder (VAE) version to do node reconstruction as well as adjacency matrix recovering. The encoder is consisted of 2 layers of graph convolutional layer, and the decoder first generates the features, then the adjacency matrix. The model achieves great accuracy for a simple two layers encoding-two layers decoding autoencoder. 

For more information on the Graph Auto-Encoder, please see Thomas Kipf's original [github page for graph auto-encoders](https://github.com/tkipf/gae). To train the model, simply run 

```bash
python train.py
```



Reference:

```
@article{kipf2016variational,
  title={Variational Graph Auto-Encoders},
  author={Kipf, Thomas N and Welling, Max},
  journal={NIPS Workshop on Bayesian Deep Learning},
  year={2016}
}
```
