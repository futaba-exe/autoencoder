console.log("Hello Autoencoder!")

import * as tf from '@tensorflow/tfjs-node'

const autoencoder = tf.sequential();

const encoder = tf.layers.dense({
    units: 32,
    inputShape: [784],
    activation: 'relu'
});

const decoder = tf.layers.dense({
    units: 784,
    activation: 'sigmoid'
    // inputShape: [32]
});

autoencoder.add(encoder);
autoencoder.add(decoder);

const learningRate = 0.15;
const optimiser = tf.train.adam(learningRate);
autoencoder.compile({
    optimizer: optimiser,
    loss: 'binaryCrossentropy',
    //metrics: ['accuracy'],
});

function generateImage(){
    const img = [];
    for (let i = 0; i < 784; i++) {
        img[i] = Math.random();
    }
    return img;
}

const x_inputs = [];
for (let i = 0; i < 1000; i++) {
    x_inputs[i] = generateImage();
}

const x_train = tf.tensor2d(x_inputs);
x_train.print();

trainModel();

async function trainModel() {
    await autoencoder.fit(x_train, x_train, {
        epochs: 50,
        batchSize: 256,
        shuffle: true,
        verbose: true,
    });
}




