keras
===

- description
  - test code for [keras](http://keras.io/) with tensorflow backend
    - codes mostly from `examples` directory in [keras git](https://github.com/fchollet/keras.git)
  - [guide to keras sequential model](http://keras.io/getting-started/sequential-model-guide/)

- set backend to tensorflow
```
(after installing keras and tensorflow)
$ cp keras.json ~/.keras/keras.json
$ cat keras.json
{"epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow"}
```

- test code
```
mlp.py
```

- mnist
```
mnist_cnn.py  mnist_irnn.py  mnist_mlp.py
```

- reuters
```
reuters_mlp.py
```
