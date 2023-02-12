var express = require('express');
var router = express.Router();
var _ = require('lodash');
var path = require('path');
var Multer = require('multer');
var request = require('request');


// google cloud storage
const Storage = require('@google-cloud/storage');
const storage = Storage({
  projectId: 'tejasxrayapp'
});

const CLOUD_BUCKET = 'demo-xray1'
const bucket = storage.bucket(CLOUD_BUCKET);
function getPublicUrl (filename) {
  return `https://storage.googleapis.com/${CLOUD_BUCKET}/${filename}`;
}

function sendUploadToGCS (req, res, next) {
  if (!req.file) {
    return next();
  }

  const gcsname = Date.now() + req.file.originalname;
  const file = bucket.file(`upload/${gcsname}`);

  const stream = file.createWriteStream({
    metadata: {
      contentType: req.file.mimetype
    },
    resumable: false
  });

  stream.on('error', (err) => {
    req.file.cloudStorageError = err;
    next(err);
  });

  stream.on('finish', () => {
    req.file.cloudStorageObject = gcsname;
    file.makePublic().then(() => {
      req.file.cloudStoragePublicUrl = getPublicUrl(gcsname);
      next();
    });
  });

  stream.end(req.file.buffer);
}

// mutter
var fileFilter = function(req, file, cb) {
	var allowedMimes = ['image/jpeg', 'image/pjpeg', 'image/png', 'image/gif'];

	if (_.includes(allowedMimes, file.mimetype)) {
		cb(null, true);
	} else {
		cb(new Error('Invalid file type. Only jpg, png and gif image files are allowed.'));
	}
};

var mutter = Multer({
  storage: Multer.MemoryStorage,
  limits: {
    files: 1,
    fileSize: 5 * 1024 * 1024 // no larger than 5mb
  },
  fileFilter: fileFilter
})

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Chest X-ray', cxr_field: 'cxr'});
});


router.post(
  '/upload',
  mutter.single('cxr'),
  sendUploadToGCS,
  function(req, res, next) {

    console.log(req.file)

    options = {
      uri: 'https://backend123-4kc43x6ysq-an.a.run.app/cxr',
      method: 'POST',
      json: {
        image_name: req.file.cloudStorageObject
      }
    }

	  request(options, (err, result, body) => {
	    if (err) { return console.log(err);  }
	   	  res.render('upload', {title: 'Result', body: body})
	  })

});

module.exports = router;
