var express = require('express')
var app = express()
var fs = require('fs')
var path = require('path');
var low = require('lowdb')
const db = low('db.json')
const state = db.getState()

state.ids = state.ids||{}

var r = (i)=>Math.random()*(i||1)

var imaginations = path.resolve( __dirname + '/../generated')
var indexhtml = path.resolve(__dirname + '/index.html')

app.get('/', (req,res)=>{
  state.visited = state.visited||0 + 1
  db.write()
  res.sendFile(indexhtml)
})

app.get('/generated/:id',(req,res)=>{
  var id = req.params.id
  state.served = state.served||0 + 1

  console.log(id);
  fullpath = imaginations +'/'+ id + '.jpg'
  res.sendFile(fullpath)
})

var keys = {}

app.get('/random',(req,res,next)=>{
  // list all imaginations
  state.asked = state.asked||0 + 1

  fs.readdir(imaginations,(err,files)=>{
    if(err)next(err);
    else{
      //files is list of filenames
      var index = Math.floor(r(files.length))
      var id = files[index].split('.')[0]

      var key = Math.floor(Math.random()*256*16777216).toString()
      keys[key] = 1

      // return the id
      res.json({
        id,
        key,
      })
    }
  })
})

// scoring
app.get('/score/:id/:score/:key',(req,res)=>{
  var id = req.params.id
  var sc = req.params.score
  var key = req.params.key
  sc = Math.floor(Number(sc))
  if((sc!==0 && sc!==1)||id.length!=50||keys[key]!==1){
    throw('bad param')
  }

  //reset the key to prevent reuse
  keys[key]=undefined

  if(state.ids[id]){
    state.ids[id][(sc==1?'p':'n')] += 1
  }else{
    state.ids[id] = {p:sc,n:1-sc}
  }

  state.scored = state.scored||0 + 1

  db.write()
  res.json({message:'success'})
})

app.get('/score/:id',(req,res)=>{
  var id = req.params.id
  var score = state.ids[id]
  res.json(score) //{p:1,n:0}
})

app.listen(3000,()=>{
  console.log('listening on 3000');
})
