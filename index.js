import {TRAINING_DATA} from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js"

const PREDICTION_ELEMENT = document.getElementById('prediction')

const INPUTS = TRAINING_DATA.inputs
const OUTPUTS = TRAINING_DATA.outputs
tf.util.shuffleCombo(INPUTS, OUTPUTS)

function normalize(tensor, min, max){
  const results = tf.tidy(function(){
    const MIN_VALUES = tf.scalar(min)
    const MAX_VALUES = tf.scalar(max)

    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES)
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES)
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE)

    return NORMALIZED_VALUES
  })
  return results
}
const INPUTS_TENSOR = normalize(tf.tensor2d(INPUTS), 0, 255);

const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);

const LOOKUP = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

const model = tf.sequential()

model.add(tf.layers.conv2d({
  inputShape: [28,28,1],
  filters: 16,
  kernelSize: 3,
  strides: 1,
  padding: 'same',
  activation: 'relu'
}))
model.add(tf.layers.maxPooling2d({poolSize:2, strides: 2}))

model.add(tf.layers.conv2d({
  filters: 32,
  kernelSize: 3,
  strides: 1,
  padding: 'same',
  activation: 'relu'
}))
model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}))

model.add(tf.layers.flatten())
model.add(tf.layers.dense({units: 128, activation: 'relu'}))
model.add(tf.layers.dense({units: 10, activation: 'softmax'}))
model.summary()

train()

async function train(){
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })

  const RESHAPED_INPUTS = INPUTS_TENSOR.reshape([INPUTS.length, 28, 28, 1])

  let results = await model.fit(RESHAPED_INPUTS, OUTPUTS_TENSOR, {
    shuffle: true,
    validationSplit: 0.15,
    epochs: 30,
    batchSize: 256,
    callbacks: {onEpochEnd: logProgress}
  })

  RESHAPED_INPUTS.dispose()
  OUTPUTS_TENSOR.dispose()
  INPUTS_TENSOR.dispose()

  evaluate()
}

function logProgress(epoch, logs) {
  console.log('Data for epoch ' + epoch, logs);
}

function evaluate(){
  const OFFSET = Math.floor((Math.random() * INPUTS.length))
  
  let answer = tf.tidy(function(){
    let newInput = normalize(tf.tensor1d(INPUTS[OFFSET]), 0, 255)
    let output = model.predict(newInput.reshape([1, 28, 28, 1]))

    output.print()
    return output.squeeze().argMax()
  })

  answer.array().then(function(index){
    PREDICTION_ELEMENT.innerText = LOOKUP[index]
    PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET])? 'correct': 'wrong')

    answer.dispose()
    drawImage(INPUTS[OFFSET])
  })
}

const CANVAS = document.getElementById('canvas');
const CTX = CANVAS.getContext('2d');

function drawImage(digit){
  var imageData = CTX.getImageData(0, 0, 28, 28);

  for(let i=0; i < digit.length; i++){
    imageData.data[i * 4] = digit[i] * 255
    imageData.data[i * 4 + 1] = digit[i] * 255
    imageData.data[i * 4 + 2] = digit[i] * 255
    imageData.data[i * 4 + 3] = 255
  }

  CTX.putImageData(imageData, 0, 0)
  setTimeout(evaluate, interval)
}

var interval = 2000;
const RANGER = document.getElementById('ranger');
const DOM_SPEED = document.getElementById('domSpeed');

RANGER.addEventListener('input', function(e) {
  interval = this.value;
  DOM_SPEED.innerText = 'Change speed of classification! Currently: ' + interval + 'ms';
});
