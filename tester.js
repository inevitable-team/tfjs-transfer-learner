const transferLearner = require("./transferLearner.js");

runner(1, { epochs: 20 });
runner(2, { epochs: 20, imageLimiter: 0.5 });

async function runner(id, config) {
    let transferLearnerTester = new transferLearner(config);
    await transferLearnerTester.setup();
    console.log(`${id} Training Data Size: `, transferLearnerTester.trainingData.length);
    console.log(`${id} Testing Data Size: `, transferLearnerTester.testingData.length);
    await transferLearnerTester.train();
    await transferLearnerTester.evaluate();
    console.table(transferLearnerTester.prettyConfusionMatrix());
    console.log(`${id} Accuracy: `, transferLearnerTester.accuracy() + "%");

    console.log(`${id} Benchmark: `, transferLearnerTester.benchmarkResults());

    let onePrediction = await transferLearnerTester.predictOne(`${__dirname}/example_dataset/green/1.png`);
    console.log(`${id} Predict One Prediction: `, onePrediction);
}