// Import required packages
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const natural = require('natural');
const tokenizer = new natural.WordTokenizer();
const SentimentAnalyzer = require('natural').SentimentAnalyzer;
const stemmer = require('natural').PorterStemmer;
const path = require('path');
const express = require('express');
const cors = require("cors");

// Create a new express app
const app = express();

// Define some constants
const maxSequenceLength = 512; // Define the maximum sequence length
const trainingData = JSON.parse(fs.readFileSync('../training-data.json'));
const modelPath = 'file://../model/model.json';
const vocabPath = '../vocab.json';

// Load the saved model
const modelPromise = tf.loadLayersModel(modelPath);

// Create a sentiment analyzer object
const analyzer = new SentimentAnalyzer('English', stemmer, 'afinn');

// Define a function to get the sentiment of an input string
function getSentiment(input) {
    const tokens = tokenizer.tokenize(input);
    const score = analyzer.getSentiment(tokens);
    if (score === 0) {
        return 'neutral';
    } else if (score > 0) {
        return 'positive';
    } else {
        return 'negative';
    }
}

// Load the vocab object from vocab.json
const vocabPromise = new Promise((resolve, reject) => {
    fs.readFile(vocabPath, (err, data) => {
        if (err) {
            reject(err);
        } else {
            resolve(JSON.parse(data));
        }
    });
});

// Define a function to preprocess an input string
async function preprocessInput(input) {
    // Use a regex tokenizer to split the input string into keywords
    const tokenizer = new natural.RegexpTokenizer({ pattern: /[\s,.']+/ });
    const inputKeywords = tokenizer.tokenize(input);
    // Load the vocabulary from the vocabPromise
    const vocab = await vocabPromise;
    // Convert the keywords to indices in the vocabulary
    const inputArray = [];
    for (let i = 0; i < maxSequenceLength; i++) {
        const keyword = inputKeywords[i];
        if (keyword in vocab) {
            inputArray.push(vocab[keyword]);
        }
    }
    // If there are no valid indices, return null
    if (inputArray.length === 0) {
        return null;
    }
    // Remove any padding zeros and convert to a tensor
    const filteredArray = inputArray.filter(value => value > 0);
    let newArray = new Array(maxSequenceLength).fill(0);
    for (let i = 0; i < filteredArray.length; i++) {
        newArray[i] = filteredArray[i];
    }
    return tf.tensor2d([newArray], [1, maxSequenceLength]);
}

// Define a function to predict the output category for an input string
async function predictOutput(input) {
    // Load the model from the modelPromise
    const model = await modelPromise;
    // Preprocess the input string
    const inputTensor = await preprocessInput(input);
    // If the input is null, return a default message
    if (inputTensor === null) {
        const defaultMessage = 'I am not sure what you mean. Could you please rephrase your question?';
        return defaultMessage;
    }
    // Make a prediction with the model and find the output category
    const prediction = model.predict(inputTensor);
    const outputIndex = tf.argMax(prediction, axis = -1).dataSync()[0];
    const outputCategory = trainingData[outputIndex]?.output;

    // If outputCategory is not found, return defaultMessage
    const defaultMessage = 'I am not sure what you mean. Could you please rephrase your question?';
    return outputCategory !== undefined ? outputCategory : defaultMessage;
}

// Serve the static files in the public folder
app.use(cors());

// Handle POST requests to /predict
app.post('/predict', async (req, res) => {
    let body = '';
    req.on('data', chunk => {
        body += chunk.toString();
    });
    req.on('end', async () => {
        const input = JSON.parse(body).input;
        const output = await predictOutput(input);
        const sentiment = getSentiment(input);
        res.setHeader('Content-Type', 'application/json');
        res.write(JSON.stringify({ output, sentiment }));
        res.end();
    });
});

// Handle GET requests to /
app.get('/', (req, res) => {
    const filePath = path.join(__dirname, 'index.html');
    fs.readFile(filePath, (err, data) => {
        if (err) {
            res.writeHead(500, { 'Content-Type': 'text/plain' });
            res.write('Error loading index.html');
            res.end();
        } else {
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.write(data);
            res.end();
        }
    });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
