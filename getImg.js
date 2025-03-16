const fs = require('fs');
const axios = require('axios');
const SerpApi = require('google-search-results-nodejs');

// API configuration
const API_KEY = ""; // Register at https://serpapi.com/
const searchQueries = {
    "scratch": "car scratch damage",
    "yellow_light": "yellowed car headlights",
    "torn_seat": "torn car seats",
    "worn_tire": "worn car tires",
    "dented_door": "dented car doors"
};

const outputDir = "./downloaded_images";
if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir);

// Function to download images
const downloadImage = async (url, folder, filename) => {
    const folderPath = `${outputDir}/${folder}`;
    if (!fs.existsSync(folderPath)) fs.mkdirSync(folderPath);
    
    try {
        const response = await axios({ url, responseType: 'stream' });
        const writer = fs.createWriteStream(`${folderPath}/${filename}`);
        response.data.pipe(writer);
        console.log(`Downloaded successfully: ${filename}`);
    } catch (error) {
        console.log(`Download error: ${filename}`);
    }
};

// Search and download images
const search = new SerpApi.GoogleSearch(API_KEY);

Object.entries(searchQueries).forEach(([label, query]) => {
    console.log(`Downloading images for: ${query}`);
    search.json({
        q: query,
        tbm: "isch",
        num: 1500, // Number of images to download
        api_key: API_KEY
    }, async (data) => {
        const images = data.images_results || [];
        for (let i = 0; i < images.length; i++) {
            const imgUrl = images[i].original;
            if (imgUrl) {
                await downloadImage(imgUrl, label, `${label}_${i}.jpg`);
            }
        }
    });
});

console.log("Image download complete!");
