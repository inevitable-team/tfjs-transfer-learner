const transferLearner = require("./transferLearner.js");

runner({ epochs: 20 });

async function runner(config) {
    let transferLearnerTester = new transferLearner(config);
    await transferLearnerTester.setup();
    await transferLearnerTester.train();
    await transferLearnerTester.evaluate();
    console.table(transferLearnerTester.prettyConfusionMatrix());
    console.log("Accuracy: ", transferLearnerTester.accuracy() + "%");

    let onePrediction = await transferLearnerTester.predictOne(`${__dirname}/example_dataset/green/1.png`);
    console.log("Predict One Prediction: ", onePrediction);
}