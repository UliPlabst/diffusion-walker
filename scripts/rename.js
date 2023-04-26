const fs = require("fs")
const path = require("path")
let dir = "./data/upscaled/landscapes";
images = fs.readdirSync(dir);
images
  .forEach(e => {
    let s = e.split("."); 
    let n = Number.parseInt(s[0]); 
    let pad = ('00000000' + n).slice(-8); 
    let name = `${pad}.jpg`; 
    fs.renameSync(
      path.join(dir, e), 
      path.join(dir, name)
    );
  })
