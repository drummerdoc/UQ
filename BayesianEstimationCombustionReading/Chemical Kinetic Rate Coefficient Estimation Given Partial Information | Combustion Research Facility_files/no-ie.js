// Hover Zoom
jQuery(document).ready(function($){
	$(".hover-zoom").hover(function(){
			$(this).children("span").hide();										   
			$(this).children("span").stop().fadeTo(300, 1); 
		},function(){
			$(this).children("span").stop().fadeTo(300, 0);
	});
}); 

// Hover Zoom
jQuery(document).ready(function($){
	$(".gallery-zoom").hover(function(){
			$(this).children("span").hide();										   
			$(this).children("span").stop().fadeTo(300, 1); 
		},function(){
			$(this).children("span").stop().fadeTo(300, 0);
	});
});

// Hover Zoom
jQuery(document).ready(function($){
	$(".gallery-zoom-video").hover(function(){
			$(this).children("span").hide();										   
			$(this).children("span").stop().fadeTo(300, 1); 
		},function(){
			$(this).children("span").stop().fadeTo(300, 0);
	});
});

// Hover Zoom
jQuery(document).ready(function($){
	$(".comdisp a, .comdisp-home a, a.read-tiny").hover(function(){
			$(this).children("span").hide();										   
			$(this).children("span").stop().fadeTo(300, 1); 
		},function(){
			$(this).children("span").stop().fadeTo(300, 0);
	});
});