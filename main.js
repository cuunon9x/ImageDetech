const fs = require('fs');
const path = require('path');

// Dataset directory
const datasetDir = "./dataset";
const outputCSV = "labels.csv";

// List of labels based on directories
const labels = ["scratch", "torn_seat", "worn_tire", "dented_door"];

// Array to hold labeled data
let labelData = [];

// Iterate through each directory and label
labels.forEach(label => {
    const labelPath = path.join(datasetDir, label);
    if (fs.existsSync(labelPath)) {
        const images = fs.readdirSync(labelPath).filter(file => file.endsWith('.jpg') || file.endsWith('.png'));
        images.forEach(image => {
            labelData.push(`${image},${label}`);
        });
    }
});

// Write to CSV file
fs.writeFileSync(outputCSV, "image,label\n" + labelData.join("\n"));

console.log(`Labeled! File: ${outputCSV}`);
