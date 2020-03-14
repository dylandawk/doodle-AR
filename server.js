const express = require("express");
const path = require("path");
const fs = require("fs");

// machine learning model requirements
const tf = require("@tensorflow/tfjs");
const tfn = require("@tensorflow/tfjs-node");
const handler = tfn.io.fileSystem(__dirname + '/model/model.json');

//parsing the HTTP POST
const multer = require("multer");
const bodyparser = require("body-parser");
const morgan = require("morgan");
const crypto = require("crypto");

const port = process.env.PORT || 3000;
const app = express();

app.use(bodyparser.json());
app.use(bodyparser.urlencoded({extended:true}));
app.use(morgan('dev'));

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

//loadMyModel();


app.get("/", async (req,res) =>{
    res.send("Model has loaded")   
});

app.get("/api/images", (req,res) => {

});

app.post("/api/image", upload.single('avatar'), (req,res) => {
    if(!req.file){
        console.log("No file received");
        return res.send({
            success: false
        });
    } else {
        const host = req.host;
        const filePath = req.protocol + "://" + host + '/' + req.file.path;
        console.log(`File: ${filePath} recieved`);
        return res.send({
            success: true
        });
    }
});


app.listen(port, () => {
    console.log(`Server listening on port: ${port}`);
})