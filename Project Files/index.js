// Load model in browser
const model = await tf.loadLayersModel('model.json');
const imgTensor = tf.browser.fromPixels(image).resizeNearestNeighbor([224, 224]).expandDims();
const prediction = model.predict(imgTensor);
