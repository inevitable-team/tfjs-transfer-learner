const tf = require("@tensorflow/tfjs-node")

class datasetWrapper {
  constructor() {
      this.example;
      this.value;
  }

  addExample(example, value, units) {
    const y = tf.tidy(() => tf.tensor2d(value, [1,units]).toFloat());

    if (this.xs == null) {
      this.xs = tf.keep(example);
      this.ys = tf.keep(y);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));

      const oldY = this.ys;
      this.ys = tf.keep(oldY.concat(y, 0));

      oldX.dispose();
      oldY.dispose();
      y.dispose();
    }
  }
}

module.exports = datasetWrapper;