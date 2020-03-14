const express = require("express");
const tf = require("@tensorflow/tfjs");
const tfn = require("@tensorflow/tfjs-node");
const handler = tfn.io.fileSystem(__dirname + '/model/model.json');


const path = require("path");
const fs = require("fs");

//parsing the HTTP POST
const multer = require("multer");

const app = express();
const port = process.env.PORT || 3000;

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

app.get("/api", (req,res) => {

});

app.listen(port, () => {
    console.log(`Server listening on port: ${port}`);
})