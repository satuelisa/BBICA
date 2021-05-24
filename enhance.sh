#!/bin/zsh
loc="/Users/elisa/Dropbox/Research/Topics/Arboles/CA/RAW/jul25b/images"
for file in `ls -1 ${loc}/*.JPG`; do
    frame=`basename $file .JPG`
    echo Enhancing $frame
    if [ ! -f ${loc}/enhanced/normalized/$frame.png ]; then
	convert $file -separate -normalize -combine ${loc}/enhanced/normalized/$frame.png
    fi
    if [ ! -f ${loc}/enhanced/equalized/$frame.png ]; then    
	convert -equalize ${loc}/enhanced/normalized/$frame.png ${loc}/enhanced/equalized/$frame.png
    fi
    if [ ! -f ${loc}/enhanced/uniform/$frame.png ]; then    
	redist -s uniform ${loc}/enhanced/equalized/$frame.png ${loc}/enhanced/uniform/$frame.png
	# the script redist is from http://www.fmwconcepts.com/imagemagick/redist/n
    fi
    # NOTEL redist uniform ROTATES THE IMAGE 180 DEGREES!!!
    convert ${loc}/enhanced/uniform/$frame.png -rotate 180  ${loc}/enhanced/oriented/$frame.png  
    # 50,220 ground too red, whites turn blues
    # same with 80,180
    # 100,150 is pretty good, a little white
    convert ${loc}/enhanced/oriented/$frame.png -modulate 100,140 ${loc}/enhanced/modulated/$frame.png 
    convert -transparent black ${loc}/enhanced/modulated/$frame.png ${loc}/enhanced/$frame.png
done
