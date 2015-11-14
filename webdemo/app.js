var exampleImagePath = '../examples/images/';

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
	nameLabel.innerHTML = imageData[currentSelection][0];
	labelList.innerHTML = "";
	
	for(var i = 0; i < imageData[currentSelection][1][0].length; i++)
	{		
		labelList.innerHTML += '<li>' 
			+ imageData[currentSelection][1][0][i] + ': ' + Math.round(imageData[currentSelection][1][1][i] * 10000) * 0.01 + '%'
			+ '<div class="progress">'
				+ '<span class="progress-bar" role="progressbar" aria-valuenow="60" aria-valuemin="0" aria-valuemax="100" style="width: '+ imageData[currentSelection][1][1][i] * 100 + '%;">' 	
				+ '</span>'
			+ '</div>'
		+ '</li>';
	}
		//+ ", labels: " + imageData[currentSelection][1][0] 
	//	+ ", probabilities: " + imageData[currentSelection][1][1];
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
   $.getJSON('results.json', function(data) {	 
	   imageData = data;	   
	   currentSelection = 0;   
	   updateSelection();
	
	   // imageContainer.innerHTML = data[0][0] + ", labels: " + data[0][1][0];
	   // console.log(imageContainer);
	   // imageContainer.appendChild(img);
   });
})
