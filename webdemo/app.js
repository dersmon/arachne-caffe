var exampleImagePath = 'images/demo/';

var currentSelection = 0;

var img = document.querySelector('#displayedImage');
var nameLabel = document.querySelector('#imageName');
var labelList = document.querySelector('#labelList');
var pagination = document.querySelector('.pagination');

var imageData = null;

function updateSelection() {
	updateImage();
	updateInfo();
}

function updateImage() {
	img.src = exampleImagePath + imageData[currentSelection][0];
}

function updateInfo() {
   glyphSpan = document.createElement('span');
   glyphSpan.className = "glyphicon glyphicon-link"
   glyphSpan.setAttribute("aria-hidden", true)

   arachneLink = document.createElement('a');
   arachneLink.href = "http://arachne.dainst.org/entity/" + imageData[currentSelection][0].slice(0, -4);
   arachneLink.appendChild(glyphSpan)
   arachneLink.innerHTML += "Arachne entity " + imageData[currentSelection][0].slice(0, -4) + " ";

   while (nameLabel.firstChild) {
       nameLabel.removeChild(nameLabel.firstChild);
   }
   nameLabel.appendChild(arachneLink);
   labelList.innerHTML = "";

	for(var i = 0; i < imageData[currentSelection][1][0].length; i++)
	{
      labelList.innerHTML += imageData[currentSelection][1][0][i] + ': ' + Math.round(imageData[currentSelection][1][1][i] * 10000) * 0.01 + '%'

      divObject = document.createElement('div');
      divObject.className = 'progress'

      spanObject = document.createElement('span')
      spanObject.className = 'progress-bar'
      spanObject.setAttribute('role', 'progressbar');
      spanObject.setAttribute('aria-valuenow', 60);
      spanObject.setAttribute('aria-valuemin', 0);
      spanObject.setAttribute('aria-valuemax', 100);
      spanObject.style.width = (imageData[currentSelection][1][1][i] * 100) + '%';

      divObject.appendChild(spanObject)
      labelList.appendChild(divObject)
	}
}

function lastImage() {
	if(currentSelection == 0){
		return;
	} else {
		currentSelection--;
		updateSelection();
	}
}

function nextImage() {
	if(currentSelection == imageData.length - 1){
		return;
	} else {
		currentSelection++;
		updateSelection();
	}
}


$(function() {
   $.getJSON('image_predictions/demo.json', function(data) {
	   imageData = data;
	   currentSelection = 0;
	   updateSelection();
   });
})
