const express = require("express");
const tf = require("@tensorflow/tfjs");
const app = express();
const port = process.env.PORT || 3000;

async function loadMyModel() {
    model = await tf.loadLayersModel('model/model.json');
    model.summary();
}


app.get("/", function(req,res){
    res.sendFile(__dirname + "/index.html");
});

app.listen(port, () => {
    console.log(`Server listening on port: ${port}`);
})