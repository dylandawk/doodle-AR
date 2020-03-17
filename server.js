const express = require("express");
const path = require("path");
const fs = require("fs");

// machine learning model requirements
const tf = require("@tensorflow/tfjs");
const tfn = require("@tensorflow/tfjs-node@1.2.9");
const handler = tfn.io.fileSystem(__dirname + '/model/model.json');
const IMAGE_SIZE = 784;
const CLASSES = ['flashlight', 'belt', 'mushroom', 'pond', 'strawberry', 'pineapple', 'sun', 'cow', 'ear', 'bush', 'pliers', 'watermelon', 'apple', 'baseball', 'feather', 'shoe', 'leaf', 'lollipop', 'crown', 'ocean', 'horse', 'mountain', 'mosquito', 'mug', 'hospital', 'saw', 'castle', 'angel', 'underwear', 'traffic_light', 'cruise_ship', 'marker', 'blueberry', 'flamingo', 'face', 'hockey_stick', 'bucket', 'campfire', 'asparagus', 'skateboard', 'door', 'suitcase', 'skull', 'cloud', 'paint_can', 'hockey_puck', 'steak', 'house_plant', 'sleeping_bag', 'bench', 'snowman', 'arm', 'crayon', 'fan', 'shovel', 'leg', 'washing_machine', 'harp', 'toothbrush', 'tree', 'bear', 'rake', 'megaphone', 'knee', 'guitar', 'calculator', 'hurricane', 'grapes', 'paintbrush', 'couch', 'nose', 'square', 'wristwatch', 'penguin', 'bridge', 'octagon', 'submarine', 'screwdriver', 'rollerskates', 'ladder', 'wine_bottle', 'cake', 'bracelet', 'broom', 'yoga', 'finger', 'fish', 'line', 'truck', 'snake', 'bus', 'stitches', 'snorkel', 'shorts', 'bowtie', 'pickup_truck', 'tooth', 'snail', 'foot', 'crab', 'school_bus', 'train', 'dresser', 'sock', 'tractor', 'map', 'hedgehog', 'coffee_cup', 'computer', 'matches', 'beard', 'frog', 'crocodile', 'bathtub', 'rain', 'moon', 'bee', 'knife', 'boomerang', 'lighthouse', 'chandelier', 'jail', 'pool', 'stethoscope', 'frying_pan', 'cell_phone', 'binoculars', 'purse', 'lantern', 'birthday_cake', 'clarinet', 'palm_tree', 'aircraft_carrier', 'vase', 'eraser', 'shark', 'skyscraper', 'bicycle', 'sink', 'teapot', 'circle', 'tornado', 'bird', 'stereo', 'mouth', 'key', 'hot_dog', 'spoon', 'laptop', 'cup', 'bottlecap', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'smiley_face', 'waterslide', 'eyeglasses', 'ceiling_fan', 'lobster', 'moustache', 'carrot', 'garden', 'police_car', 'postcard', 'necklace', 'helmet', 'blackberry', 'beach', 'golf_club', 'car', 'panda', 'alarm_clock', 't-shirt', 'dog', 'bread', 'wine_glass', 'lighter', 'flower', 'bandage', 'drill', 'butterfly', 'swan', 'owl', 'raccoon', 'squiggle', 'calendar', 'giraffe', 'elephant', 'trumpet', 'rabbit', 'trombone', 'sheep', 'onion', 'church', 'flip_flops', 'spreadsheet', 'pear', 'clock', 'roller_coaster', 'parachute', 'kangaroo', 'duck', 'remote_control', 'compass', 'monkey', 'rainbow', 'tennis_racquet', 'lion', 'pencil', 'string_bean', 'oven', 'star', 'cat', 'pizza', 'soccer_ball', 'syringe', 'flying_saucer', 'eye', 'cookie', 'floor_lamp', 'mouse', 'toilet', 'toaster', 'The_Eiffel_Tower', 'airplane', 'stove', 'cello', 'stop_sign', 'tent', 'diving_board', 'light_bulb', 'hammer', 'scorpion', 'headphones', 'basket', 'spider', 'paper_clip', 'sweater', 'ice_cream', 'envelope', 'sea_turtle', 'donut', 'hat', 'hourglass', 'broccoli', 'jacket', 'backpack', 'book', 'lightning', 'drums', 'snowflake', 'radio', 'banana', 'camel', 'canoe', 'toothpaste', 'chair', 'picture_frame', 'parrot', 'sandwich', 'lipstick', 'pants', 'violin', 'brain', 'power_outlet', 'triangle', 'hamburger', 'dragon', 'bulldozer', 'cannon', 'dolphin', 'zebra', 'animal_migration', 'camouflage', 'scissors', 'basketball', 'elbow', 'umbrella', 'windmill', 'table', 'rifle', 'hexagon', 'potato', 'anvil', 'sword', 'peanut', 'axe', 'television', 'rhinoceros', 'baseball_bat', 'speedboat', 'sailboat', 'zigzag', 'garden_hose', 'river', 'house', 'pillow', 'ant', 'tiger', 'stairs', 'cooler', 'see_saw', 'piano', 'fireplace', 'popsicle', 'dumbbell', 'mailbox', 'barn', 'hot_tub', 'teddy-bear', 'fork', 'dishwasher', 'peas', 'hot_air_balloon', 'keyboard', 'microwave', 'wheel', 'fire_hydrant', 'van', 'camera', 'whale', 'candle', 'octopus', 'pig', 'swing_set', 'helicopter', 'saxophone', 'passport', 'bat', 'ambulance', 'diamond', 'goatee', 'fence', 'grass', 'mermaid', 'motorbike', 'microphone', 'toe', 'cactus', 'nail', 'telephone', 'hand', 'squirrel', 'streetlight', 'bed', 'firetruck'];
const k = 10;

//parsing the HTTP POST
const multer = require("multer");
const bodyparser = require("body-parser");
const morgan = require("morgan");
const crypto = require("crypto");

// image manipulation
const PNG = require("png-js");
const {promisify} = require("util");
const port = process.env.PORT || 3000;
const app = express();

app.use(bodyparser.json());
app.use(bodyparser.urlencoded({extended:true}));
app.use(morgan('dev'));

const testPath = "/Users/dylandawkins/Documents/GitHub/doodle-AR/images/1788a705cff2766e12d63dd1cb4a708d.png"

//select destination to save uploaded file and rename using crypto
const storage = multer.diskStorage({
    destination: `${__dirname}/images`,
    filename: function(req, file, callback) {
        crypto.pseudoRandomBytes(16, function(err, raw) {
            if (err) return callback(err);
          
            callback(null, raw.toString('hex') + path.extname(file.originalname));
          });
    }
})

const upload = multer({ storage: storage });

let model;

async function loadMyModel() {
    try{
        model = await tf.loadLayersModel(handler);
        model.summary();
    } catch (e) {
        console.dir(e)
    }
}
loadMyModel();

function guess(inputs) {

    //console.log(inputs);

    // Predict
    let guess = model.predict(tf.tensor([inputs]));
  
    // Format res to an array
    const rawProb = Array.from(guess.dataSync());
  
    // Get top K res with index and probability
    const rawProbWIndex = rawProb.map((probability, index) => {
      return {
        index,
        probability
      }
    });
    const sortProb = rawProbWIndex.sort((a, b) => b.probability - a.probability);
    const topKClassWIndex = sortProb.slice(0, k);
    const topKRes = topKClassWIndex.map(i => {
        console.log(`Classes: ${CLASSES[i.index]}, Probability: ${i.probability.toFixed(2) * 100}%`);
    });
}

async function getInputImage(path){
    
    let inputs = [];
    var prom;
    let decoded = 0;
    PNG.decode(path, function(pixels){
        return new Promise(function(resolve,reject){
            let oneRow = [];
            for (let i = 0; i < IMAGE_SIZE; i++) {
                let bright = pixels[i * 4];
                let onePix = [parseFloat((255 - bright) / 255)];
                oneRow.push(onePix);
                if (oneRow.length === 28) {
                    inputs.push(oneRow);
                    oneRow = [];
                }
            }
            if(inputs.length){
                resolve("Success");
            } else {
                reject("Failure");
            }
        }).then(function(result){
            console.log(`${result}: ${inputs.length}`);
            guess(inputs);
            decoded = 1;
        }).catch(function(err){
            console.log(err);
        });
    }); 
}


app.get("/", async (req,res) =>{
    res.send("model loaded"); 
});

// app.get("/test", async (req,res) =>{
//     res.send("pic sent"); 
//     getInputImage(testPath);
// });

app.get("/api/images", (req,res) => {

});

app.post("/api/image", upload.single('avatar'), (req,res) => {
    if(!req.file){
        console.log("No file received");
        return res.send({
            success: false
        });
    } else {
        const host = req.hostname;
        const filePath = req.protocol + "://" + host + '/' + req.file.path;
        console.log(`File: ${filePath} recieved`);
        console.log(req.file.path);
        getInputImage(req.file.path);
        return res.send({
            success: true
        });
    }
});


app.listen(port, () => {
    console.log(`Server listening on port: ${port}`);
})