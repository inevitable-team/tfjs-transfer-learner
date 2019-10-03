# TensorFlowJS Transfer Learner

Retrain the MobileNet model via transfer learning using TensorFlow.js in NodeJS.

Using the popular MobileNet model as a feature extractor, this package allows a user to target a source folder of images, then produce a model to be trained or tested.

## Example Usages

### Basic

To use this package and run it on the test data provided, the only code that needs to be wrote is:

```js
    let config = {};
    let transferLearnerTester: new transferLearner(config);
    await transferLearnerTester.setup();
    await transferLearnerTester.train();
    await transferLearnerTester.evaluate();
    console.table(transferLearnerTester.prettyConfusionMatrix());
    console.log("Accuracy: ", transferLearnerTester.accuracy() + "%");
```

Which will then display a result of:

```
    ┌────────────────┬───────────────────┬────────────────────┬──────────────────┐
    │    (index)     │ "blue" Prediction │ "green" Prediction │ "red" Prediction │
    ├────────────────┼───────────────────┼────────────────────┼──────────────────┤
    │ "blue" Actual  │         1         │         0          │        0         │
    │ "green" Actual │         1         │         2          │        0         │
    │  "red" Actual  │         0         │         0          │        3         │
    └────────────────┴───────────────────┴────────────────────┴──────────────────┘
    Accuracy:  85.71%
```

### Use Model to Predict a new Image 

Using the trained model from above, the following code can be added to see the prediction of a given image.

```js
    let onePrediction = await transferLearnerTester.predictOne(`${__dirname}/example_dataset/green/1.png`);
    console.log("Predict One Prediction: ", onePrediction);
```

Which will then log the following result:

```
    Predict One Prediction:  green
```

## Source Folder Structure

In the `imagesUrl` parameter you specify for the source folder, there are two acceptable structure, one where you can manually split the training and testing data, then another where it is randomly determined by the `split` parameter.

### Manual Split 

```
example_dataset
+-- training
|   +-- blue
|   |   +-- image_x.jpeg
|   |   +-- image_z.png
|   |   +-- image_l.png
|   +-- green
|   |   +-- image_q.png
|   |   +-- image_a.jpg
|   |   +-- image_d.jpeg
+-- testing
|   +-- blue
|   |   +-- image_er.png
|   |   +-- image_do.jpg
|   |   +-- image_b.jpg
|   +-- green
|   |   +-- ima_0.jpeg
|   |   +-- imatg.jpg
```

### Auto Split 

```
example_dataset
+-- blue
|   +-- image_x.jpeg
|   +-- image_z.png
|   +-- image_do.jpg
|   +-- image_b.jpg
+-- green
|   +-- image_q.png
|   +-- image_a.jpg
|   +-- image_d.jpeg
```

## Parameters

The `imagesUrl` has been mentioned to specify the source folder, but below are all of the possible parameters that can be used when instantiating a new transferLearner Object.

```js
    let config: {
        onlyTesting: false, // Optional: Boolean, true if you want to test via other means and use the "predictOne" function
        split: 0.75, // Optional: Float, vary the difference in training and testing data, 0.75: 75% of the images will be used for training
        oldModel: null,  // Optional: tf.model(), Only pass if you do not wish to download and use the model from the oldModelUrl
        oldModelUrl: 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json',  // Optional: URL / String
        oldModelLayer: 'conv_pw_13_relu',  // Optional: String, which layer of the old model to be used as the feature extractor
        oldModelImageSize: 224,  // Optional: Number, specifiy the input width/height of the old model
        oldModelImageShape: [this.oldModelImageSize, this.oldModelImageSize, 3],  // Optional, using the input size to get the shape
        imagesUrl: `${__dirname}/example_dataset`,  // Optional: String, specify the location of where the source folder is of the images
        lossFunction: 'categoricalCrossentropy',  // Optional: String, loss function for the models training phase
        optimizer: tf.train.adam(),  // Optional: tf.train / String, optimizer for the models training phase
        epochs: 5,  // Optional: Number, specify the amount of epoches to be run during the training phase
        batchSize: 8 // Optional: Number, specify the size of batchs to be run during the training phase
    }
```