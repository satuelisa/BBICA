#!/bin/zsh
loc="/Volumes/dropbox/Dropbox/Research/Topics/Arboles/CA/RAW/aug27b/images" # already done for jul25b
for file in `ls -1 ${loc}/*.JPG`; do
    frame=`basename $file .JPG`
    echo Enhancing $frame
    if [ ! -f ${loc}/enhanced/normalized/$frame.png ]; then
	convert $file -separate -normalize -combine ${loc}/enhanced/normalized/$frame.png
	rm ${loc}/enhanced/equalized/$frame.png
    fi
    if [ ! -f ${loc}/enhanced/equalized/$frame.png ]; then    
	convert -equalize ${loc}/enhanced/normalized/$frame.png ${loc}/enhanced/equalized/$frame.png
	rm ${loc}/enhanced/uniform/$frame.png	
    fi
    if [ ! -f ${loc}/enhanced/uniform/$frame.png ]; then    
	redist -s uniform ${loc}/enhanced/equalized/$frame.png ${loc}/enhanced/uniform/$frame.png
	# the script redist is from http://www.fmwconcepts.com/imagemagick/redist/n
	rm ${loc}/enhanced/oriented/$frame.png	
    fi
    # NOTE: redist uniform ROTATES THE IMAGE 180 DEGREES!!!
    if [ ! -f ${loc}/enhanced/oriented/$frame.png ]; then    
	convert ${loc}/enhanced/uniform/$frame.png -rotate 180  ${loc}/enhanced/oriented/$frame.png
	rm ${loc}/enhanced/modulated/$frame.png		
    fi
    if [ ! -f ${loc}/enhanced/modulated/$frame.png ]; then        
    # 50,220 ground too red, whites turn blues
    # same with 80,180
    # 100,150 is pretty good, a little white
	convert ${loc}/enhanced/oriented/$frame.png -modulate 100,140 ${loc}/enhanced/modulated/$frame.png
    fi
    if [ ! -f ${loc}/enhanced/$frame.png ]; then        
	convert -transparent black ${loc}/enhanced/modulated/$frame.png ${loc}/enhanced/$frame.png
    fi
done
