const fs = require("fs")
images = fs.readdirSync("./data");
images
  .forEach(e => {
    let s = e.split("."); 
    let n = Number.parseInt(s[0]); 
    let pad = ('00000000' + n).slice(-8); 
    let name = `${pad}.jpg`; 
    fs.renameSync(e, name);
  })
