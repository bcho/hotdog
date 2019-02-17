# Train / Change logs

## 2019-02-17

in my local machine, with CPU complication

### train

```
Train on 21600 samples, validate on 2400 samples
Epoch 1/10
21600/21600 [==============================] - 53s 2ms/step - loss: 0.6721 - acc: 0.5980 - val_loss: 0.6764 - val_acc: 0.6171
Epoch 2/10
21600/21600 [==============================] - 48s 2ms/step - loss: 0.6409 - acc: 0.6274 - val_loss: 0.6471 - val_acc: 0.6196
Epoch 3/10
21600/21600 [==============================] - 41s 2ms/step - loss: 0.6349 - acc: 0.6327 - val_loss: 0.6402 - val_acc: 0.6417
Epoch 4/10
21600/21600 [==============================] - 50s 2ms/step - loss: 0.6304 - acc: 0.6411 - val_loss: 0.6552 - val_acc: 0.6154
Epoch 5/10
21600/21600 [==============================] - 46s 2ms/step - loss: 0.6247 - acc: 0.6481 - val_loss: 0.6851 - val_acc: 0.6229
Epoch 6/10
21600/21600 [==============================] - 49s 2ms/step - loss: 0.6082 - acc: 0.6609 - val_loss: 0.6802 - val_acc: 0.6288
Epoch 7/10
21600/21600 [==============================] - 53s 2ms/step - loss: 0.5934 - acc: 0.6787 - val_loss: 0.6822 - val_acc: 0.6267
Epoch 8/10
21600/21600 [==============================] - 50s 2ms/step - loss: 0.5645 - acc: 0.6991 - val_loss: 0.7128 - val_acc: 0.6367
Epoch 9/10
21600/21600 [==============================] - 49s 2ms/step - loss: 0.5323 - acc: 0.7256 - val_loss: 0.8013 - val_acc: 0.6429
Epoch 10/10
21600/21600 [==============================] - 49s 2ms/step - loss: 0.4867 - acc: 0.7579 - val_loss: 0.7722 - val_acc: 0.6396
6000/6000 [==============================] - 2s 408us/step
INFO 2019-02-17 18:27:46 +0800 steering loss: 0.784000692208608
INFO 2019-02-17 18:27:46 +0800 steering acc: 0.6318333333333334
```

### predict

```
python steering_model.py predict data/image/hotdog-00bcb4e1
image: data/image/hotdog-00bcb4e1
is a hotdog: 0.670391857624054
is not a hotdog: 0.32960814237594604
```

```
python steering_model.py predict data/image/pets-6041f8c6
image: data/image/pets-6041f8c6
is a hotdog: 0.08674793690443039
is not a hotdog: 0.913252055644989
```
